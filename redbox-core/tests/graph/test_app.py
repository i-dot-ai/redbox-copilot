from uuid import uuid4
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import RunnableLambda
import tiktoken

from redbox.models.chain import ChainInput, ChainState
from redbox import Redbox
from redbox.models.chat import ChatRoute
from redbox.models.settings import Settings
from redbox.test.data import TestData, RedboxChatTestCase, generate_test_cases


LANGGRAPH_DEBUG = False

test_env = Settings()

TEST_CASES = [
    test_case
    for generated_cases in [
        generate_test_cases(
            query=ChainInput(
                question="What is AI?",
                file_uuids=[],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings={"chat_backend": "fake", "fake_backend_responses": ["Testing Response 1"]},
            ),
            test_data=[
                TestData(0, 0, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat),
                TestData(1, 100, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat),
                TestData(10, 1200, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat),
            ],
            test_id="Basic Chat",
        ),
        generate_test_cases(
            query=ChainInput(
                question="What is AI?",
                file_uuids=[uuid4()],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings={"chat_backend": "fake", "fake_backend_responses": ["Testing Response 1"]},
            ),
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
            query=ChainInput(
                question="What is AI?",
                file_uuids=[uuid4(), uuid4()],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings={"chat_backend": "fake", "fake_backend_responses": ["Testing Response 1"]},
            ),
            test_data=[
                TestData(
                    2,
                    40000,
                    expected_llm_response=["Map Step Response"] * 2 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
                TestData(
                    2,
                    100_000,
                    expected_llm_response=["Map Step Response"] * 2 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
                TestData(
                    2,
                    200_000,
                    expected_llm_response=["Map Step Response"] * 2 + ["Testing Response 1"],
                    expected_route=ChatRoute.chat_with_docs,
                ),
            ],
            test_id="Chat with multiple docs",
        ),
        generate_test_cases(
            query=ChainInput(
                question="What is AI?",
                file_uuids=[uuid4()],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings={"chat_backend": "fake", "fake_backend_responses": ["Testing Response 1"]},
            ),
            test_data=[
                TestData(
                    1, 200_000, expected_llm_response=["Testing Response 1"], expected_route=ChatRoute.chat_with_docs
                ),
            ],
            test_id="Chat with large doc",
        ),
        generate_test_cases(
            query=ChainInput(
                question="@search What is AI?",
                file_uuids=[uuid4()],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings={"chat_backend": "fake", "fake_backend_responses": ["The cake is a lie"]},
            ),
            test_data=[
                TestData(
                    1,
                    10000,
                    expected_llm_response=["Condense response", "The cake is a lie"],
                    expected_route=ChatRoute.search,
                ),
                TestData(
                    5,
                    10000,
                    expected_llm_response=["Condense response", "The cake is a lie"],
                    expected_route=ChatRoute.search,
                ),
            ],
            test_id="Search",
        ),
        generate_test_cases(
            query=ChainInput(
                question="@nosuchkeyword What is AI?",
                file_uuids=[uuid4()],
                user_uuid=uuid4(),
                chat_history=[],
                ai_settings={"chat_backend": "fake", "fake_backend_responses": ["Testing Response 1"]},
            ),
            test_data=[
                TestData(
                    2,
                    200_000,
                    expected_llm_response=[test_env.response_no_such_keyword],
                    expected_route=ChatRoute.error_no_keyword,
                ),
            ],
            test_id="No Such Keyword",
        ),
    ]
    for test_case in generated_cases
]


@pytest.fixture(scope="session")
def env():
    return Settings()


@pytest.fixture(scope="session")
def tokeniser():
    return tiktoken.get_encoding("cl100k_base")


def all_chunks_retriever(docs):
    def mock_retrieve(query):
        return docs

    return RunnableLambda(mock_retrieve)


def parameterised_retriever(docs):
    def mock_retrieve(query):
        return docs

    return RunnableLambda(mock_retrieve)


@pytest.mark.asyncio
@pytest.mark.parametrize(("test_case"), TEST_CASES, ids=[t.test_id for t in TEST_CASES])
async def test_chat(test_case: RedboxChatTestCase, env, tokeniser):
    app = Redbox(
        # llm=GenericFakeChatModel(messages=iter(test_case.test_data.expected_llm_response)),
        all_chunks_retriever=all_chunks_retriever(test_case.docs),
        parameterised_retriever=parameterised_retriever(test_case.docs),
        tokeniser=tokeniser,
        env=env,
        debug=LANGGRAPH_DEBUG,
    )
    response = await app.run(
        input=ChainState(query=test_case.query),
    )
    final_state = ChainState(response)
    assert (
        final_state["response"] == test_case.test_data.expected_llm_response[-1]
    ), f"Expected LLM response: '{test_case.test_data.expected_llm_response[-1]}'. Received '{final_state["response"]}'"
    assert (
        final_state["route_name"] == test_case.test_data.expected_route
    ), f"Expected Route: '{ test_case.test_data.expected_route}'. Received '{final_state["route_name"]}'"


@pytest.mark.asyncio
@pytest.mark.parametrize(("test_case"), TEST_CASES, ids=[t.test_id for t in TEST_CASES])
async def test_streaming(test_case: RedboxChatTestCase, env, tokeniser):
    app = Redbox(
        # llm=GenericFakeChatModel(messages=iter(test_case.test_data.expected_llm_response)),
        all_chunks_retriever=all_chunks_retriever(test_case.docs),
        parameterised_retriever=parameterised_retriever(test_case.docs),
        tokeniser=tokeniser,
        env=env,
        debug=LANGGRAPH_DEBUG,
    )

    token_events = []

    async def streaming_response_handler(tokens: str):
        token_events.append(tokens)

    response = await app.run(
        input=ChainState(query=test_case.query), response_tokens_callback=streaming_response_handler
    )
    final_state = ChainState(response)

    # Bit of a bodge to retain the ability to check that the LLM streaming is working in most cases
    if not final_state["route_name"].startswith("error"):
        assert len(token_events) > 1, f"Expected tokens as a stream. Received: {token_events}"
    llm_response = "".join(token_events)
    assert (
        final_state["response"] == llm_response
    ), f"Expected LLM response: '{llm_response}'. Received '{final_state["response"]}'"
    assert (
        final_state["route_name"] == test_case.test_data.expected_route
    ), f"Expected Route: '{ test_case.test_data.expected_route}'. Received '{final_state["route_name"]}'"


def get_available_keywords():
    app = Redbox(
        llm=GenericFakeChatModel(messages=iter([])),
        all_chunks_retriever=all_chunks_retriever([]),
        parameterised_retriever=parameterised_retriever([]),
        tokeniser=tokeniser,
        env=env,
        ai=None,
        debug=LANGGRAPH_DEBUG,
    )
    keywords = {ChatRoute.search}

    assert keywords == set(app.get_available_keywords().keys())
