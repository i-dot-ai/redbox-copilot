import logging
import re
from typing import Annotated
from uuid import UUID

from core_api.auth import get_user_uuid, get_ws_user_uuid
<<<<<<< HEAD
from core_api.semantic_routes import get_routable_chains
=======
from core_api.semantic_routes import get_routable_chains, get_semantic_route_layer
>>>>>>> 96d5faf850a87bd7d9bb8a696a5b2343b6a47ba3
from fastapi import Depends, FastAPI, WebSocket
from fastapi.encoders import jsonable_encoder
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from openai import APIError
<<<<<<< HEAD

from redbox.api.runnables import map_to_chat_response
from redbox.models.chain import ChainInput, ChainChatMessage
from redbox.models.chat import ChatRequest, ChatResponse, SourceDocument, ClientResponse, ErrorDetail
=======
from pydantic import BaseModel
from redbox.api.runnables import map_to_chat_response
from redbox.models.chain import ChainInput
from redbox.models.chat import ChatRequest, ChatResponse, SourceDocument
>>>>>>> 96d5faf850a87bd7d9bb8a696a5b2343b6a47ba3
from redbox.models.errors import NoDocumentSelected, QuestionLengthError
from redbox.transform import map_document_to_source_document
from semantic_router import RouteLayer

# === Logging ===

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

re_keyword_pattern = re.compile(r"@(\w+)")


chat_app = FastAPI(
    title="Core Chat API",
    description="Redbox Core Chat API",
    version="0.1.0",
    openapi_tags=[
        {"name": "chat", "description": "Chat interactions with LLM and RAG backend"},
        {"name": "llm", "description": "LLM information and parameters"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


<<<<<<< HEAD
async def route_chat(
    chat_request: ChatRequest, user_uuid: UUID, routable_chains: dict[str, Tool]
=======
async def semantic_router_to_chain(
    chat_request: ChatRequest,
    user_uuid: UUID,
    routable_chains: dict[str, Runnable],
    route_layer: RouteLayer,
>>>>>>> 96d5faf850a87bd7d9bb8a696a5b2343b6a47ba3
) -> tuple[Runnable, ChainInput]:
    question = chat_request.message_history[-1].text

    selected_chain = None

    def select_chat_chain(chat_request: ChatRequest, routable_chains: dict[str, Runnable]) -> Runnable:
        if chat_request.selected_files:
            return routable_chains.get("chat/documents")
        else:
            return routable_chains.get("chat")

    # Match keyword
    route_match = re_keyword_pattern.search(question)
    route_name = route_match.group()[1:] if route_match else None
    selected_chain = routable_chains.get(route_name, select_chat_chain(chat_request, routable_chains))

    params = ChainInput(
        question=chat_request.message_history[-1].text,
        file_uuids=[str(f.uuid) for f in chat_request.selected_files],
        user_uuid=str(user_uuid),
        chat_history=[
            ChainChatMessage(role=message.role, text=message.text) for message in chat_request.message_history[:-1]
        ],
    )

    log.info("Routed to %s", route_name)
    log.info("Selected files: %s", chat_request.selected_files)

    return selected_chain, params


@chat_app.post("/rag", tags=["chat"])
async def rag_chat(
    chat_request: ChatRequest,
    user_uuid: Annotated[UUID, Depends(get_user_uuid)],
    routable_chains: Annotated[dict[str, Tool], Depends(get_routable_chains)],
) -> ChatResponse:
    """REST endpoint. Get a LLM response to a question history and file."""
<<<<<<< HEAD
    selected_chain, params = await route_chat(chat_request, user_uuid, routable_chains)
    return (selected_chain | map_to_chat_response).invoke(params.dict())


@chat_app.get("/tools", tags=["chat"])
async def available_tools(
    routable_chains: Annotated[dict[str, Tool], Depends(get_routable_chains)],
):
    """REST endpoint. Get a mapping of all tools available via chat."""
    return [{"name": chat_tool.name, "description": chat_tool.description} for chat_tool in routable_chains.values()]
=======
    selected_chain, params = await semantic_router_to_chain(
        chat_request, user_uuid, routable_chains, route_layer
    )
    return (selected_chain | map_to_chat_response).invoke(params.model_dump())


class ErrorDetail(BaseModel):
    code: str
    message: str


class ClientResponse(BaseModel):
    # Needs to match CoreChatResponse in django-app/redbox_app/redbox_core/consumers.py
    resource_type: Literal["text", "documents", "route_name", "end", "error"]
    data: list[SourceDocument] | str | ErrorDetail | None = None
>>>>>>> 96d5faf850a87bd7d9bb8a696a5b2343b6a47ba3


@chat_app.websocket("/rag")
async def rag_chat_streamed(
    websocket: WebSocket,
    routable_chains: Annotated[dict[str, Tool], Depends(get_routable_chains)],
):
    """Websocket. Get a LLM response to a question history and file."""
    await websocket.accept()

    user_uuid = await get_ws_user_uuid(websocket)

    request = await websocket.receive_text()
    chat_request = ChatRequest.model_validate_json(request)

<<<<<<< HEAD
    selected_chain, params = await route_chat(chat_request, user_uuid, routable_chains)
=======
    selected_chain, params = await semantic_router_to_chain(
        chat_request, user_uuid, routable_chains, route_layer
    )
>>>>>>> 96d5faf850a87bd7d9bb8a696a5b2343b6a47ba3

    try:
        async for event in selected_chain.astream(params.dict()):
            response: str = event.get("response", "")
            source_documents: list[SourceDocument] = [
                map_document_to_source_document(doc)
                for doc in event.get("source_documents", [])
            ]
            route_name: str = event.get("route_name", "")
            if response:
                await send_to_client(
                    ClientResponse(resource_type="text", data=response), websocket
                )
            if source_documents:
                await send_to_client(
                    ClientResponse(resource_type="documents", data=source_documents),
                    websocket,
                )
            if route_name:
                await send_to_client(
                    ClientResponse(resource_type="route_name", data=route_name),
                    websocket,
                )
    except NoDocumentSelected as e:
        log.info("No documents have been selected to summarise", exc_info=e)
        await send_to_client(
            ClientResponse(
                resource_type="error",
                data=ErrorDetail(code="no-document-selected", message=type(e).__name__),
            ),
            websocket,
        )
    except QuestionLengthError as e:
        log.info("Question is too long", exc_info=e)
        await send_to_client(
            ClientResponse(
                resource_type="error",
                data=ErrorDetail(code="question-too-long", message=type(e).__name__),
            ),
            websocket,
        )
    except APIError as e:
        log.exception("Unhandled exception.", exc_info=e)
        await send_to_client(
            ClientResponse(
                resource_type="error",
                data=ErrorDetail(code="unexpected", message=type(e).__name__),
            ),
            websocket,
        )
    finally:
        await send_to_client(ClientResponse(resource_type="end"), websocket)
        await websocket.close()


async def send_to_client(response: ClientResponse, websocket: WebSocket) -> None:
    await websocket.send_json(jsonable_encoder(response))
