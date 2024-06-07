import json
import logging
from collections.abc import Mapping, MutableSequence, Sequence
from types import SimpleNamespace
from uuid import UUID

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from redbox_app.redbox_core.models import ChatHistory, ChatMessage, ChatRoleEnum, File, User
from websockets.client import connect
from yarl import URL

OptFileSeq = Sequence[File] | None
logger = logging.getLogger(__name__)
logger.info("WEBSOCKET_SCHEME is: %s", settings.WEBSOCKET_SCHEME)


class ChatConsumer(AsyncWebsocketConsumer):
    async def receive(self, text_data):
        data = json.loads(text_data)
        user_message_text: str = data.get("message", "")
        session_id: str | None = data.get("sessionId", None)
        selected_file_uuids: Sequence[UUID] = [UUID(u) for u in data.get("selectedFiles", [])]
        user: User = self.scope.get("user", None)

        session = await self.get_session(session_id, user, user_message_text)

        # save user message
        selected_files = await self.get_files_by_id(selected_file_uuids, user)
        await self.save_message(session, user_message_text, ChatRoleEnum.user, selected_files=selected_files)

        # get LLM response
        session_messages = await self.get_messages(session)
        message_history: Sequence[Mapping[str, str]] = [
            {"role": message.role, "text": message.text} for message in session_messages
        ]
        url = URL.build(scheme="ws", host=settings.CORE_API_HOST, port=settings.CORE_API_PORT) / "chat/rag"
        async with connect(str(url), extra_headers={"Authorization": user.get_bearer_token()}) as websocket:
            await websocket.send(json.dumps({"message_history": message_history}))
            await websocket.send(json.dumps({"selected_files": [f.core_file_uuid for f in selected_files]}))
            await self.send_json({"type": "session-id", "data": str(session.id)})
            full_reply: MutableSequence[str] = []
            source_files: Sequence[File] = []
            async for raw_message in websocket:
                message = json.loads(raw_message, object_hook=lambda d: SimpleNamespace(**d))
                logger.debug("Received: %s", message)
                if message.resource_type == "text":
                    await self.send_json({"type": "text", "data": message.data})
                    full_reply.append(message.data)
                elif message.resource_type == "documents":
                    doc_uuids: Sequence[UUID] = [UUID(doc.file_uuid) for doc in message.data]
                    source_files = await self.get_files_by_core_uuid(doc_uuids, user)
                    for source in source_files:
                        await self.send_json(
                            {
                                "type": "source",
                                "data": {"url": str(source.url), "original_file_name": source.original_file_name},
                            }
                        )
                elif message.resource_type == "end":
                    await self.save_message(session, "".join(full_reply), ChatRoleEnum.ai, source_files=source_files)

    async def send_json(self, data):
        await self.send(json.dumps(data, default=str))

    @staticmethod
    @database_sync_to_async
    def get_session(session_id: str, user: User, user_message_text: str) -> ChatHistory:
        if session_id:
            session = ChatHistory.objects.get(id=session_id)
        else:
            session_name = user_message_text[0:20]
            session = ChatHistory(name=session_name, users=user)
            session.save()
        return session

    @staticmethod
    @database_sync_to_async
    def get_messages(session: ChatHistory) -> Sequence[ChatMessage]:
        return list(ChatMessage.objects.filter(chat_history=session))

    @staticmethod
    @database_sync_to_async
    def save_message(
        session: ChatHistory,
        user_message_text: str,
        role: ChatRoleEnum,
        source_files: OptFileSeq = None,
        selected_files: OptFileSeq = None,
    ) -> ChatMessage:
        chat_message = ChatMessage(chat_history=session, text=user_message_text, role=role)
        chat_message.save()
        if source_files:
            chat_message.source_files.set(source_files)
        if selected_files:
            chat_message.selected_files.set(selected_files)
        return chat_message

    @staticmethod
    @database_sync_to_async
    def get_files_by_id(uuids: Sequence[UUID], user: User) -> Sequence[File]:
        return list(File.objects.filter(id__in=uuids, user=user))

    @staticmethod
    @database_sync_to_async
    def get_files_by_core_uuid(uuids: Sequence[UUID], user: User) -> Sequence[File]:
        return list(File.objects.filter(core_file_uuid__in=uuids, user=user))
