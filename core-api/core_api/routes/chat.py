import logging
from typing import Annotated
from uuid import UUID

from redbox import Redbox

from core_api.auth import get_user_uuid, get_ws_user_uuid
from fastapi import Depends, FastAPI, WebSocket
from fastapi.encoders import jsonable_encoder
from openai import APIError
from langchain_core.documents import Document

from redbox.models.chain import RedboxQuery, RedboxState, ChainChatMessage
from redbox.models.chat import ChatRequest, ChatResponse, ClientResponse, ErrorDetail
from redbox.transform import map_document_to_source_document

from core_api.dependencies import get_redbox
from core_api.runnables import map_to_chat_response

# === Logging ===

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


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


@chat_app.post("/rag", tags=["chat"])
async def rag_chat(
    chat_request: ChatRequest,
    user_uuid: Annotated[UUID, Depends(get_user_uuid)],
    redbox: Annotated[Redbox, Depends(get_redbox)],
) -> ChatResponse:
    """REST endpoint. Get a LLM response to a question history and file."""
    state = RedboxState(
        request=RedboxQuery(
            question=chat_request.message_history[-1].text,
            file_uuids=[f.uuid for f in chat_request.selected_files],
            user_uuid=user_uuid,
            chat_history=[
                ChainChatMessage(role=message.role, text=message.text) for message in chat_request.message_history[:-1]
            ],
            ai_settings=chat_request.ai_settings,
        ),
    )
    return await (redbox.graph | map_to_chat_response).ainvoke(state)


@chat_app.get("/tools", tags=["chat"])
async def available_tools(
    redbox: Annotated[Redbox, Depends(get_redbox)],
):
    """REST endpoint. Get a mapping of all tools available via chat."""
    return [{"name": name, "description": description} for name, description in redbox.get_available_keywords().items()]


@chat_app.websocket("/rag")
async def rag_chat_streamed(
    websocket: WebSocket,
    redbox: Annotated[Redbox, Depends(get_redbox)],
):
    """Websocket. Get a LLM response to a question history and file."""
    await websocket.accept()

    user_uuid = await get_ws_user_uuid(websocket)

    request = await websocket.receive_text()
    chat_request = ChatRequest.model_validate_json(request)

    state = RedboxState(
        request=RedboxQuery(
            question=chat_request.message_history[-1].text,
            file_uuids=[f.uuid for f in chat_request.selected_files],
            user_uuid=user_uuid,
            chat_history=[
                ChainChatMessage(role=message.role, text=message.text) for message in chat_request.message_history[:-1]
            ],
            ai_settings=chat_request.ai_settings,
        ),
    )

    async def on_llm_response(tokens: str):
        await send_to_client(ClientResponse(resource_type="text", data=tokens), websocket)

    async def on_route_choice(route_name: str):
        await send_to_client(ClientResponse(resource_type="route_name", data=route_name), websocket)

    async def on_documents_available(docs: list[Document]):
        await send_to_client(
            ClientResponse(resource_type="documents", data=[map_document_to_source_document(d) for d in docs]),
            websocket,
        )

    try:
        await redbox.run(
            state,
            response_tokens_callback=on_llm_response,
            route_name_callback=on_route_choice,
            documents_callback=on_documents_available,
        )
    except APIError as e:
        log.exception("Unhandled exception.", exc_info=e)
        await send_to_client(
            ClientResponse(resource_type="error", data=ErrorDetail(code="unexpected", message=type(e).__name__)),
            websocket,
        )
    finally:
        await send_to_client(ClientResponse(resource_type="end"), websocket)
        await websocket.close()


async def send_to_client(response: ClientResponse, websocket: WebSocket) -> None:
    await websocket.send_json(jsonable_encoder(response))
