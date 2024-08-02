import json
from typing import TYPE_CHECKING
from uuid import uuid4
from jose import jwt

from langchain_elasticsearch import ElasticsearchStore
import pytest
from core_api import dependencies
from core_api.app import app as application
from core_api.routes.chat import chat_app
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from redbox.models.chain import ChainInput
from redbox.models.chat import ChatResponse, ChatRoute
from redbox.test.data import RedboxChatTestCase, generate_test_cases, TestData

if TYPE_CHECKING:
    pass

system_chat = {"text": "test", "role": "system"}
user_chat = {"text": "test", "role": "user"}

RAG_LLM_RESPONSE = "Based on your documents the answer to your question is 7"
UPLOADED_FILE_UUID = "9aa1aa15-dde0-471f-ab27-fd410612025b"

EXPECTED_AVAILABLE_ROUTES = {"search"}

TEST_CASES = [
    test_case
    for generated_cases in [
        generate_test_cases(
            query=ChainInput(question="What is AI?", file_uuids=[], user_uuid=uuid4(), chat_history=[]),
            test_data=[
                TestData(0, 0, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat),
                TestData(1, 100, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat),
                TestData(10, 1200, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat),
            ],
            test_id="Basic Chat",
        ),
        generate_test_cases(
            query=ChainInput(question="What is AI?", file_uuids=[uuid4()], user_uuid=uuid4(), chat_history=[]),
            test_data=[
                TestData(
                    1, 1000, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat_with_docs
                ),
                TestData(
                    1, 50000, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat_with_docs
                ),
                TestData(
                    1, 200_000, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat_with_docs
                ),
            ],
            test_id="Chat with single doc",
        ),
        generate_test_cases(
            query=ChainInput(question="What is AI?", file_uuids=[uuid4(), uuid4()], user_uuid=uuid4(), chat_history=[]),
            test_data=[
                TestData(
                    2,
                    40000,
                    expected_llm_response=["Map Response"] * 2 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
                TestData(
                    2,
                    100_000,
                    expected_llm_response=["Map Response"] * 2 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
                TestData(
                    4,
                    200_000,
                    expected_llm_response=["Map Response"] * 4 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
            ],
            test_id="Chat with multiple docs",
        ),
        generate_test_cases(
            query=ChainInput(question="What is AI?", file_uuids=[uuid4()], user_uuid=uuid4(), chat_history=[]),
            test_data=[
                TestData(
                    2,
                    200_000,
                    expected_llm_response=["Map Response"] * 2 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
            ],
            test_id="Chat with large doc",
        ),
    ]
    for test_case in generated_cases
]


@pytest.fixture(params=TEST_CASES, ids=[t.test_id for t in TEST_CASES])
def test_case(request):
    return request.param


@pytest.fixture
def client(test_case: RedboxChatTestCase, embedding_model):
    chat_app.dependency_overrides[dependencies.get_embedding_model] = lambda: embedding_model
    yield TestClient(application)
    chat_app.dependency_overrides = {}


@pytest.fixture
def uploaded_docs(test_case: RedboxChatTestCase, elasticsearch_store: ElasticsearchStore):
    docs_ids = elasticsearch_store.add_documents(test_case.docs)
    yield
    if docs_ids:
        elasticsearch_store.delete(docs_ids)


@pytest.fixture
def query_headers(test_case: RedboxChatTestCase):
    return {"Authorization": f"Bearer {jwt.encode({"user_uuid": str(test_case.query.user_uuid)}, key="nvjkernd")}"}


def test_rag(test_case: RedboxChatTestCase, client, uploaded_docs, query_headers):
    response = client.post(
        "/chat/rag",
        headers=query_headers,
        json={
            "message_history": [
                {"role": message.role, "text": message.text} for message in test_case.query.chat_history
            ]
            + [{"role": "user", "text": test_case.query.question}],
            "selected_files": [{"uuid": str(file_uuid)} for file_uuid in test_case.query.file_uuids],
        },
    )
    assert response.status_code == 200, response.text
    chat_response = ChatResponse.model_validate(response.json())

    assert (
        chat_response.output_text == test_case.test_data.expected_llm_response[-1]
    ), f"Expected response [{test_case.test_data.expected_llm_response}] received [{chat_response.output_text}]"
    assert (
        chat_response.route_name == test_case.test_data.expected_route
    ), f"Expected route [{test_case.test_data.expected_route}] received [{chat_response.route_name}]"
    returned_document_texts = set([d.page_content for d in chat_response.source_documents])
    test_query_matching_document_texts = set([d.page_content for d in test_case.get_docs_matching_query()])
    unexpected_returned_documents = list(
        filter(
            lambda d: d.page_content in returned_document_texts - test_query_matching_document_texts,
            chat_response.source_documents,
        )
    )
    assert len(unexpected_returned_documents) == 0, f"Unexpected source docs in result {unexpected_returned_documents}"


def test_rag_chat_streamed(test_case: RedboxChatTestCase, client, uploaded_docs, query_headers):
    with client.websocket_connect("/chat/rag", headers=query_headers) as websocket:
        # When
        websocket.send_text(
            json.dumps(
                {
                    "message_history": [
                        {"role": message.role, "text": message.text} for message in test_case.query.chat_history
                    ]
                    + [{"role": "user", "text": test_case.query.question}],
                    "selected_files": [{"uuid": str(file_uuid)} for file_uuid in test_case.query.file_uuids],
                }
            )
        )

        all_text, docs, route_name = [], [], ""
        while True:
            try:
                actual = websocket.receive_json()
                if actual["resource_type"] == "text":
                    all_text.append(actual["data"])
                if actual["resource_type"] == "documents":
                    docs.extend(actual["data"])
                if actual["resource_type"] == "route_name":
                    route_name = actual["data"]
            except WebSocketDisconnect:
                break

        # Then
        text = "".join(all_text)
        assert (
            text == test_case.test_data.expected_llm_response[-1]
        ), f"Expected response [{test_case.test_data.expected_llm_response}] received [{text}]"
        assert (
            route_name == test_case.test_data.expected_route
        ), f"Expected route [{test_case.test_data.expected_route}] received [{route_name}]"
        returned_document_texts = set([d["page_content"] for d in docs])
        test_query_matching_document_texts = set([d.page_content for d in test_case.get_docs_matching_query()])
        unexpected_returned_documents = list(
            filter(lambda d: d["page_content"] in returned_document_texts - test_query_matching_document_texts, docs)
        )
        assert (
            len(unexpected_returned_documents) == 0
        ), f"Unexpected source docs in result {unexpected_returned_documents}"


def test_available_tools(client, query_headers):
    response = client.get("/chat/tools", headers=query_headers)
    assert response.status_code == 200
    tool_definitions = response.json()
    assert len(tool_definitions) > 0
    assert EXPECTED_AVAILABLE_ROUTES == {item["name"] for item in tool_definitions}
    for tool_definition in tool_definitions:
        assert "name" in tool_definition
        assert "description" in tool_definition
