import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.schema import StreamEvent
from semantic_router.encoders import TfidfEncoder
from starlette.websockets import WebSocketDisconnect

from core_api.src import dependencies, semantic_routes
from core_api.src.app import app as application
from core_api.src.routes.chat import chat_app
from redbox.models.chat import ChatResponse, ChatRoute

if TYPE_CHECKING:
    from collections.abc import Iterable

system_chat = {"text": "test", "role": "system"}
user_chat = {"text": "test", "role": "user"}

RAG_LLM_RESPONSE = "Based on your documents the answer to your question is 7"
UPLOADED_FILE_UUID = "9aa1aa15-dde0-471f-ab27-fd410612025b"


def mock_chat_prompt():
    return ChatPromptValue(
        messages=[
            SystemMessage(content="You are a helpful AI bot."),
            HumanMessage(content="Hello, how are you doing?"),
        ]
    )


def embedding_model_dim(embedding_model) -> int:
    return len(embedding_model.embed_query("foo"))


def mock_get_llm(llm_responses):
    def wrapped():
        return FakeStreamingListLLM(responses=llm_responses)

    return wrapped


def mock_parameterised_retriever(alice):
    docs = [
        Document(
            page_content="some text that doesn't actually matter " * 10,
            metadata={"index": i, "parent_file_uuid": UPLOADED_FILE_UUID, "creator_user_uuid": str(alice)},
        )
        for i in range(12)
    ]
    return RunnableLambda(lambda _: docs)


def mock_all_chunks_retriever(alice):
    docs = [
        Document(
            page_content="some text that doesn't actually matter " * 10,
            metadata={"index": i, "parent_file_uuid": UPLOADED_FILE_UUID, "creator_user_uuid": str(alice)},
        )
        for i in range(12)
    ]
    return RunnableLambda(lambda _: docs)


def mock_semantic_route_encoder():
    encoder = TfidfEncoder()
    routes = semantic_routes.get_semantic_routes()
    encoder.fit(routes)
    return encoder


@pytest.fixture(scope="session")
def mock_client(alice):
    chat_app.dependency_overrides[dependencies.get_llm] = mock_get_llm([RAG_LLM_RESPONSE] * 32)
    chat_app.dependency_overrides[dependencies.get_all_chunks_retriever] = lambda: mock_all_chunks_retriever(alice)
    chat_app.dependency_overrides[dependencies.get_parameterised_retriever] = lambda: mock_parameterised_retriever(
        alice
    )
    chat_app.dependency_overrides[semantic_routes.get_semantic_routing_encoder] = mock_semantic_route_encoder
    yield TestClient(application)
    chat_app.dependency_overrides = {}


@pytest.fixture()
def mock_streaming_client():
    """
    This client mocks the retrieval pipeline to just produce events for astream_events.
    This tests only the streaming functionality as no pipeline is run
    """
    events: Iterable[StreamEvent] = [
        StreamEvent(
            event="on_chat_model_stream",
            name=f"event-{i}",
            data={"chunk": SimpleNamespace(content=response_chunk)},
            run_id="run_id",
        )
        for i, response_chunk in enumerate(RAG_LLM_RESPONSE.split(" "))
    ]
    event_iterable = MagicMock(name="event_iterable")
    event_iterable.__aiter__.return_value = events
    astream_events = MagicMock(name="astream_events", return_value=event_iterable)
    retrieval_chain = AsyncMock(spec=Runnable, name="retrieval_chain")
    retrieval_chain.astream_events = astream_events
    chat_app.dependency_overrides[semantic_routes.get_routable_chains] = lambda: {"retrieval": retrieval_chain}
    yield TestClient(application)
    chat_app.dependency_overrides = {}


def test_rag_chat_rest_gratitude(mock_client, headers):
    response = mock_client.post(
        "/chat/rag",
        json={"message_history": [{"role": "user", "text": "Thank you"}]},
        headers=headers,
    )
    chat_response = ChatResponse.model_validate(response.json())
    assert chat_response.output_text == "You're welcome!"
    assert chat_response.route_name == ChatRoute.gratitude


def test_rag(mock_client, headers):
    response = mock_client.post(
        "/chat/rag",
        headers=headers,
        json={
            "message_history": [{"text": "Who put the ram in the rama lama ding dong?", "role": "user"}],
            "selected_files": [{"uuid": UPLOADED_FILE_UUID}],
        },
    )
    assert response.status_code == 200, response.text
    chat_response = ChatResponse.model_validate(response.json())
    assert chat_response.output_text == RAG_LLM_RESPONSE
    assert chat_response.route_name == ChatRoute.search


def test_summary(mock_client, headers):
    response = mock_client.post(
        "/chat/rag",
        headers=headers,
        json={
            "message_history": [
                {"text": "Summarise the documents", "role": "user"},
            ],
            "selected_files": [{"uuid": UPLOADED_FILE_UUID}],
        },
    )
    assert response.status_code == 200
    chat_response = ChatResponse.model_validate(response.json())
    assert chat_response.output_text == RAG_LLM_RESPONSE
    assert chat_response.route_name in {ChatRoute.map_reduce_summarise, ChatRoute.stuff_summarise}


def test_keyword(mock_client, headers):
    """Given a history that should summarise, force retrieval."""
    response = mock_client.post(
        "/chat/rag",
        headers=headers,
        json={
            "message_history": [
                {"text": "What can I do for you?", "role": "system"},
                {"text": "Summarise the provided docs? @search", "role": "user"},
            ],
            "selected_files": [{"uuid": UPLOADED_FILE_UUID}],
        },
    )
    assert response.status_code == 200
    chat_response = ChatResponse.model_validate(response.json())
    assert chat_response.output_text == RAG_LLM_RESPONSE
    assert chat_response.route_name == ChatRoute.search


def test_rag_chat_streamed(mock_client, headers):
    # Given
    message_history = [
        {"text": "What can I do for you?", "role": "system"},
        {"text": "Who put the ram in the rama lama ding dong?", "role": "user"},
    ]
    selected_files = [{"uuid": UPLOADED_FILE_UUID}]

    with mock_client.websocket_connect("/chat/rag", headers=headers) as websocket:
        # When
        websocket.send_text(json.dumps({"message_history": message_history, "selected_files": selected_files}))

        all_text, docs, route_name = [], [], ""
        while True:
            try:
                actual = websocket.receive_json()
                if actual["resource_type"] == "text":
                    all_text.append(actual["data"])
                if actual["resource_type"] == "documents":
                    docs.append(actual["data"])
                if actual["resource_type"] == "route_name":
                    route_name = actual["data"]
            except WebSocketDisconnect:
                break

        # Then
        text = "".join(all_text)
        assert text == RAG_LLM_RESPONSE
        assert route_name == ChatRoute.search
