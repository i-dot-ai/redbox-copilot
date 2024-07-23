import json
import logging
from asyncio import CancelledError
from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from dataclasses_json import Undefined, dataclass_json
from django.conf import settings
from django.utils import timezone
from websockets import ConnectionClosedError, WebSocketClientProtocol
from websockets.client import connect
from yarl import URL

from redbox_app.redbox_core import error_messages
from redbox_app.redbox_core.models import ChatHistory, ChatMessage, ChatRoleEnum, Citation, File, User

OptFileSeq = Sequence[File] | None
logger = logging.getLogger(__name__)
logger.info("WEBSOCKET_SCHEME is: %s", settings.WEBSOCKET_SCHEME)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class CoreChatResponseDoc:
    file_uuid: UUID
    page_content: str | None = None
    page_numbers: list[int] | None = None


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class ErrorDetail:
    code: str
    message: str


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class CoreChatResponse:
    # Needs to match ClientResponse in core_api/src/routes/chat.py
    resource_type: Literal["text", "documents", "route_name", "end", "error"]
    data: list[CoreChatResponseDoc] | str | ErrorDetail | None = None


class ChatConsumer(AsyncWebsocketConsumer):
    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data or bytes_data)
        logger.debug("received %s from browser", data)
        user_message_text: str = data.get("message", "")
        session_id: str | None = data.get("sessionId", None)
        selected_file_uuids: Sequence[UUID] = [UUID(u) for u in data.get("selectedFiles", [])]
        user: User = self.scope.get("user", None)

        session: ChatHistory = await self.get_session(session_id, user, user_message_text)

        # save user message
        selected_files = await self.get_files_by_id(selected_file_uuids, user)
        await self.save_message(session, user_message_text, ChatRoleEnum.user, selected_files=selected_files)

        await self.llm_conversation(selected_files, session, user, user_message_text)
        await self.close()

    async def llm_conversation(
        self, selected_files: Sequence[File], session: ChatHistory, user: User, title: str
    ) -> None:
        session_messages = await self.get_messages(session)
        message_history: Sequence[Mapping[str, str]] = [
            {"role": message.role, "text": message.text} for message in session_messages
        ]
        url = URL.build(scheme="ws", host=settings.CORE_API_HOST, port=settings.CORE_API_PORT) / "chat/rag"
        try:
            async with connect(str(url), extra_headers={"Authorization": user.get_bearer_token()}) as core_websocket:
                message = {
                    "message_history": message_history,
                    "selected_files": [{"uuid": f.core_file_uuid} for f in selected_files],
                }
                await self.send_to_server(core_websocket, message)
                await self.send_to_client("session-id", session.id)
                reply, citations, route = await self.receive_llm_responses(user, core_websocket)
            message = await self.save_message(session, reply, ChatRoleEnum.ai, sources=citations, route=route)
            await self.send_to_client("end", {"message_id": message.id, "title": title, "session_id": session.id})

            for file, _ in citations:
                file.last_referenced = timezone.now()
                await self.file_save(file)
        except (TimeoutError, ConnectionClosedError, CancelledError, CoreError) as e:
            logger.exception("Error from core.", exc_info=e)
            await self.send_to_client("error", error_messages.CORE_ERROR_MESSAGE)

    async def receive_llm_responses(
        self, user: User, core_websocket: WebSocketClientProtocol
    ) -> tuple[str, Sequence[tuple[File, CoreChatResponseDoc]], str]:
        full_reply: MutableSequence[str] = []
        citations: MutableSequence[tuple[File, CoreChatResponseDoc]] = []
        route: str | None = None
        async for raw_message in core_websocket:
            response: CoreChatResponse = CoreChatResponse.schema().loads(raw_message)
            logger.debug("received %s from core-api", response)
            if response.resource_type == "text":
                full_reply.append(await self.handle_text(response))
            elif response.resource_type == "documents":
                citations += await self.handle_documents(response, user)
            elif response.resource_type == "route_name":
                route = await self.handle_route(response, user.is_staff)
            elif response.resource_type == "error":
                full_reply.append(await self.handle_error(response))
        return "".join(full_reply), citations, route

    async def handle_documents(
        self, response: CoreChatResponse, user: User
    ) -> Sequence[tuple[File, CoreChatResponseDoc]]:
        source_files, citations = await self.get_sources_with_files(response.data, user)
        for file in source_files:
            await self.send_to_client("source", {"url": str(file.url), "original_file_name": file.original_file_name})
        return citations

    async def handle_text(self, response: CoreChatResponse) -> str:
        await self.send_to_client("text", response.data)
        return response.data

    async def handle_route(self, response: CoreChatResponse, show_route: bool) -> str:
        # TODO(@rachaelcodes): remove is_staff conditional and hidden-route with new route design
        # https://technologyprogramme.atlassian.net/browse/REDBOX-419
        if show_route:
            await self.send_to_client("route", response.data)
        else:
            await self.send_to_client("hidden-route", response.data)
        return response.data

    async def handle_error(self, response: CoreChatResponse) -> str:
        match response.data.code:
            case "no-document-selected":
                message = error_messages.SELECT_DOCUMENT
                await self.send_to_client("text", message)
                return message
            case "question-too-long":
                message = error_messages.QUESTION_TOO_LONG
                await self.send_to_client("text", message)
                return message
            case _:
                message = f"{response.data.code}: {response.data.message}"
                raise CoreError(message)

    async def send_to_client(self, message_type: str, data: str | Mapping[str, Any] | None = None) -> None:
        message = {"type": message_type, "data": data}
        logger.debug("sending %s to browser", message)
        await self.send(json.dumps(message, default=str))

    @staticmethod
    async def send_to_server(websocket, data):
        logger.debug("sending %s to core-api", data)
        return await websocket.send(json.dumps(data, default=str))

    @staticmethod
    @database_sync_to_async
    def get_session(session_id: str, user: User, user_message_text: str) -> ChatHistory:
        if session_id:
            session = ChatHistory.objects.get(id=session_id)
        else:
            session_name = user_message_text[0 : settings.CHAT_TITLE_LENGTH]
            session = ChatHistory(name=session_name, users=user)
            session.save()
        return session

    @staticmethod
    @database_sync_to_async
    def get_messages(session: ChatHistory) -> Sequence[ChatMessage]:
        return list(ChatMessage.objects.filter(chat_history=session).order_by("created_at"))

    @staticmethod
    @database_sync_to_async
    def save_message(
        session: ChatHistory,
        user_message_text: str,
        role: ChatRoleEnum,
        sources: Sequence[tuple[File, CoreChatResponseDoc]] | None = None,
        selected_files: Sequence[File] | None = None,
        route: str | None = None,
    ) -> ChatMessage:
        chat_message = ChatMessage(chat_history=session, text=user_message_text, role=role, route=route)
        chat_message.save()
        if sources:
            for file, citations in sources:
                for citation in citations:
                    Citation.objects.create(
                        chat_message=chat_message,
                        file=file,
                        text=citation.page_content,
                        page_numbers=citation.page_numbers,
                    )
        if selected_files:
            chat_message.selected_files.set(selected_files)
        return chat_message

    @staticmethod
    @database_sync_to_async
    def get_files_by_id(docs: Sequence[UUID], user: User) -> Sequence[File]:
        return list(File.objects.filter(id__in=docs, user=user))

    @staticmethod
    @database_sync_to_async
    def get_sources_with_files(
        docs: Sequence[CoreChatResponseDoc], user: User
    ) -> tuple[Sequence[File], Sequence[tuple[File, CoreChatResponseDoc]]]:
        uuids = [doc.file_uuid for doc in docs]
        files = File.objects.filter(core_file_uuid__in=uuids, user=user)

        return files, [(file, [doc for doc in docs if doc.file_uuid == file.core_file_uuid]) for file in files]

    @staticmethod
    @database_sync_to_async
    def file_save(file):
        return file.save()


class CoreError(Exception):
    message: str
