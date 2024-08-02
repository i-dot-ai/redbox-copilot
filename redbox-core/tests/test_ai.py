from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import jsonlines
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from elasticsearch.helpers import bulk, scan
from pydantic import BaseModel, Field

from redbox.chains.components import get_parameterised_retriever, get_tokeniser, get_llm
from redbox.graph.search import get_search_graph
from redbox.models.chain import ChainInput, ChainState, AISettings

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from elasticsearch import Elasticsearch
    from langchain_community.chat_models import ChatLiteLLM

    from redbox.models import Settings


logging.getLogger().setLevel(logging.CRITICAL)

ROOT = Path(__file__).parents[2]
DATA = ROOT / "notebooks/evaluation/data/0.2.3"


class ExperimentData(BaseModel):
    """Test required a versioned CSV of evaluation questions and a pre-embedded index."""

    data: Path
    embeddings: Path
    test_cases: list[LLMTestCase] = Field(default_factory=list)


RAG_EXPERIMENT_DATA = ExperimentData(
    data=DATA / "synthetic/rag.json", embeddings=DATA / "embeddings/text-embedding-3-large.jsonl"
)
"""
Experiment data should follow the following pattern.

user_scenario: A distinct user action with a consistent capability and difficulty
id: An index of tests within that scenario
notes: A short, plain-English explanation of what you're testing and why
input: The input to the LLM
content: A list of the chunks from the document that an expected answer would use
expected_output: The expected answer
"""

RAG_TESTS: list[tuple[str, str, list[str]]] = []

for testcase in json.load(RAG_EXPERIMENT_DATA.data.open()):
    # Test only one case per user scenario here
    if testcase["id"] == 1:
        raw_pytest = pytest.param(
            testcase["input"],
            testcase["expected_output"],
            testcase["context"],
            id=testcase["user_scenario"],
        )
        RAG_TESTS.append(raw_pytest)

RAG_TESTS = RAG_TESTS[:2]


def clear_index(index: str, es: Elasticsearch) -> None:
    if es.indices.exists(index=index):
        documents = scan(es, index=index, query={"query": {"match_all": {}}})
        bulk_data = [{"_op_type": "delete", "_index": doc["_index"], "_id": doc["_id"]} for doc in documents]
        bulk(es, bulk_data, request_timeout=300)


@pytest.fixture(scope="session")
def ai_experiment_data() -> ExperimentData:
    return RAG_EXPERIMENT_DATA


@pytest.fixture(scope="session")
def ai_settings():
    return AISettings()


@pytest.fixture(scope="session")
def llm(ai_settings) -> ChatLiteLLM:
    return get_llm(ai_settings)


@pytest.fixture(scope="session")
def eval_llm(ai_settings) -> DeepEvalBaseLLM:
    """Creates LLM for evaluating our data.

    Note in its current form this hard-codes the same model for generation
    and evaluation, giving a high risk of overfit in these unit tests.
    This should be changed as soon as possible.
    """

    class ChatLiteLLMDeepEval(DeepEvalBaseLLM):
        def __init__(self, model):
            self.model = model

        def load_model(self):
            return self.model

        def generate(self, prompt: str) -> str:
            chat_model = self.load_model()
            return chat_model.invoke(prompt).content

        async def a_generate(self, prompt: str) -> str:
            chat_model = self.load_model()
            res = await chat_model.ainvoke(prompt)
            return res.content

        def get_model_name(self):
            return "Custom LiteLLM Model"

    return ChatLiteLLMDeepEval(model=get_llm(ai_settings))


@pytest.fixture(scope="session")
def elastic_index_and_user(
    ai_experiment_data: ExperimentData, es_client: Elasticsearch
) -> Generator[tuple[str, str], None, None]:
    index_name = ai_experiment_data.embeddings.stem

    # Clear embeddings from index (in case previous crash stopped teardown)
    clear_index(index=index_name, es=es_client)

    user_uuids: set[UUID] = set()

    # Load embeddings to index
    with jsonlines.open(ai_experiment_data.embeddings, mode="r") as reader:
        for chunk_raw in reader:
            chunk = json.loads(chunk_raw)
            user_uuids.add(UUID(chunk["creator_user_uuid"]))
            es_client.index(
                index=index_name,
                id=chunk["uuid"],
                body=chunk,
            )

    if len(user_uuids) > 1:
        msg = "Embeddings have more than one creator_user_uuid"
        raise ValueError(msg)

    yield index_name, next(iter(user_uuids))

    # Delete embeddings from index
    clear_index(index=index_name, es=es_client)


@pytest.fixture()
def make_test_case(
    llm: ChatLiteLLM, elastic_index_and_user: tuple[str, str], env: Settings
) -> Callable[[str, str, list[str], ExperimentData], LLMTestCase]:
    """
    Returns a factory for making LLMTestCases based on a row of ExperimentData.

    This solves the central engineering problem of these tests:

    1. The embeddings need to be in a fixture for teardown
    2. The LLMTestCases need to be calculated based on these
    3. The LLMTestCases are expensive and should be calculated once
        a. And the normal solution to this is a fixture
    4. The LLMTestCases should be parameters in the tests
    5. You can't use fixtures as parameters
    6. Each test should cover a single metric

    The solution is twofold:

    * Using a factory means we can use a fixture for our embeddings,
    but have the data that creates the test cases be a parameter, because
    the LLMTestCase is made inside the test itself (1, 2, 3, 4, 5)
    * To ensure calculating once and having a test per metric, we use
    ExperimentData as a cache, solving 6 while not violating 3
    """
    _, user_uuid = elastic_index_and_user

    retriever = get_parameterised_retriever(env=env)

    rag_chain = get_search_graph(
        retriever=retriever,
        tokeniser=get_tokeniser(),
    )

    def _make_test_case(
        prompt: str, expected_output: str, context: list[str], experiment: ExperimentData
    ) -> LLMTestCase:
        for case in experiment.test_cases:
            if case.input == prompt:
                return case
        answer: ChainState = rag_chain.invoke(
            input=ChainState(
                query=ChainInput(
                    question=prompt,
                    file_uuids=[],
                    user_uuid=str(user_uuid),
                    chat_history=[],
                )
            )
        )

        return LLMTestCase(
            input=prompt,
            actual_output=answer["response"],
            expected_output=expected_output,
            context=context,
            retrieval_context=[doc.page_content for doc in answer["source_documents"]],
        )

    return _make_test_case


@pytest.mark.ai()
@pytest.mark.parametrize(("prompt", "expected_output", "context"), RAG_TESTS)
def test_contextual_precision(
    make_test_case: Callable, eval_llm: DeepEvalBaseLLM, prompt: str, expected_output: str, context: list[str]
) -> None:
    """Are relevant retrieval_context nodes ranked higher than irrelevant?
    Retrieval metric. Higher is better.
    https://docs.confident-ai.com/docs/metrics-contextual-precision
    """
    deepeval_test_case = make_test_case(
        prompt=prompt, expected_output=expected_output, context=context, experiment=RAG_EXPERIMENT_DATA
    )
    if prompt not in [case.input for case in RAG_EXPERIMENT_DATA.test_cases]:
        RAG_EXPERIMENT_DATA.test_cases.append(deepeval_test_case)

    contextual_precision = ContextualPrecisionMetric(
        threshold=0.5,  # default is 0.5
        model=eval_llm,
        include_reason=True,
    )
    assert_test(deepeval_test_case, [contextual_precision])


@pytest.mark.ai()
@pytest.mark.parametrize(("prompt", "expected_output", "context"), RAG_TESTS)
def test_contextual_recall(
    make_test_case: Callable, eval_llm: DeepEvalBaseLLM, prompt: str, expected_output: str, context: list[str]
) -> None:
    """How much does retrieval_context align with expected_output?
    Retrieval metric. Higher is better.
    https://docs.confident-ai.com/docs/metrics-contextual-recall
    """
    deepeval_test_case = make_test_case(
        prompt=prompt, expected_output=expected_output, context=context, experiment=RAG_EXPERIMENT_DATA
    )
    if prompt not in [case.input for case in RAG_EXPERIMENT_DATA.test_cases]:
        RAG_EXPERIMENT_DATA.test_cases.append(deepeval_test_case)

    contextual_recall = ContextualRecallMetric(
        threshold=0.5,  # default is 0.5
        model=eval_llm,
        include_reason=True,
    )
    assert_test(deepeval_test_case, [contextual_recall])


@pytest.mark.ai()
@pytest.mark.parametrize(("prompt", "expected_output", "context"), RAG_TESTS)
def test_contextual_relevancy(
    make_test_case: Callable, eval_llm: DeepEvalBaseLLM, prompt: str, expected_output: str, context: list[str]
) -> None:
    """How relevant is the retrieval_context to the input?
    Retrieval metric. Higher is better.
    https://docs.confident-ai.com/docs/metrics-contextual-relevancy
    """
    deepeval_test_case = make_test_case(
        prompt=prompt, expected_output=expected_output, context=context, experiment=RAG_EXPERIMENT_DATA
    )
    if prompt not in [case.input for case in RAG_EXPERIMENT_DATA.test_cases]:
        RAG_EXPERIMENT_DATA.test_cases.append(deepeval_test_case)

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.5,  # default is 0.5
        model=eval_llm,
        include_reason=True,
    )
    assert_test(deepeval_test_case, [contextual_relevancy])


@pytest.mark.ai()
@pytest.mark.parametrize(("prompt", "expected_output", "context"), RAG_TESTS)
def test_answer_relevancy(
    make_test_case: Callable, eval_llm: DeepEvalBaseLLM, prompt: str, expected_output: str, context: list[str]
) -> None:
    """How relevant is the actual_answer to the input?
    Generation metric. Higher is better.
    https://docs.confident-ai.com/docs/metrics-answer-relevancy
    """
    deepeval_test_case = make_test_case(
        prompt=prompt, expected_output=expected_output, context=context, experiment=RAG_EXPERIMENT_DATA
    )
    if prompt not in [case.input for case in RAG_EXPERIMENT_DATA.test_cases]:
        RAG_EXPERIMENT_DATA.test_cases.append(deepeval_test_case)

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5,  # default is 0.5
        model=eval_llm,
        include_reason=True,
    )
    assert_test(deepeval_test_case, [answer_relevancy])


@pytest.mark.ai()
@pytest.mark.parametrize(("prompt", "expected_output", "context"), RAG_TESTS)
def test_faithfulness(
    make_test_case: Callable, eval_llm: DeepEvalBaseLLM, prompt: str, expected_output: str, context: list[str]
) -> None:
    """How factually faithful is the actual_output to the retrieval_context?
    Generation metric. Higher is better.
    https://docs.confident-ai.com/docs/metrics-faithfulness
    """
    deepeval_test_case = make_test_case(
        prompt=prompt, expected_output=expected_output, context=context, experiment=RAG_EXPERIMENT_DATA
    )
    if prompt not in [case.input for case in RAG_EXPERIMENT_DATA.test_cases]:
        RAG_EXPERIMENT_DATA.test_cases.append(deepeval_test_case)

    faithfulness = FaithfulnessMetric(
        threshold=0.5,  # default is 0.5
        model=eval_llm,
        include_reason=True,
    )
    assert_test(deepeval_test_case, [faithfulness])


@pytest.mark.ai()
@pytest.mark.parametrize(("prompt", "expected_output", "context"), RAG_TESTS)
def test_hallucination(
    make_test_case: Callable, eval_llm: DeepEvalBaseLLM, prompt: str, expected_output: str, context: list[str]
) -> None:
    """How factually accurate is the output, comparing actual_output to context?
    Generation metric. Lower is better.
    https://docs.confident-ai.com/docs/metrics-hallucination
    """
    deepeval_test_case = make_test_case(
        prompt=prompt, expected_output=expected_output, context=context, experiment=RAG_EXPERIMENT_DATA
    )
    if prompt not in [case.input for case in RAG_EXPERIMENT_DATA.test_cases]:
        RAG_EXPERIMENT_DATA.test_cases.append(deepeval_test_case)

    hallucination = HallucinationMetric(
        threshold=0.5,  # default is 0.5
        model=eval_llm,
        include_reason=True,
    )
    assert_test(deepeval_test_case, [hallucination])
