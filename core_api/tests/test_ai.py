from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import jsonlines
import pandas as pd
import pytest
from deepeval import evaluate
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
from pydantic import BaseModel

from core_api.src.build_chains import build_retrieval_chain
from core_api.src.dependencies import get_llm, get_parameterised_retriever, get_tokeniser
from redbox.models.chain import ChainInput

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from deepeval.evaluate import TestResult
    from elasticsearch import Elasticsearch
    from langchain_community.chat_models import ChatLiteLLM

    from redbox.models import Settings


logging.getLogger("elastic_transport.transport").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

ROOT = Path(__file__).parents[2]
DATA = ROOT / "notebooks/evaluation/data/0.2.0"

CSV = DATA / "synthetic/ragas_synthetic_data.csv"
EMBEDDINGS = DATA / "embeddings/all-mpnet-base-v2.jsonl"


class ExperimentData(BaseModel):
    """Test required a versioned CSV of evaluation questions and a pre-embedded index."""

    csv: Path
    embeddings: Path


EXPERIMENT_DATA = ExperimentData(csv=CSV, embeddings=EMBEDDINGS)
RAW_TESTS: list[tuple[str, str, list[str]]] = []

for testcase in pd.read_csv(EXPERIMENT_DATA.csv).itertuples(index=False):
    RAW_TESTS.append((testcase.input, testcase.expected_output, ast.literal_eval(testcase.context)))


def clear_index(index: str, es: Elasticsearch) -> None:
    documents = scan(es, index=index, query={"query": {"match_all": {}}})
    bulk_data = [{"_op_type": "delete", "_index": doc["_index"], "_id": doc["_id"]} for doc in documents]
    bulk(es, bulk_data, request_timeout=300)


@pytest.fixture(scope="session")
def ai_experiment_data() -> ExperimentData:
    return EXPERIMENT_DATA


@pytest.fixture(scope="session")
def llm(env: Settings) -> ChatLiteLLM:
    return get_llm(env)


@pytest.fixture(scope="session")
def eval_llm(env: Settings) -> DeepEvalBaseLLM:
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

    return ChatLiteLLMDeepEval(model=get_llm(env))


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
    else:
        user_uuid = next(iter(user_uuids))

    yield index_name, user_uuid

    # Delete embeddings from index
    clear_index(index=index_name, es=es_client)


@pytest.fixture()
def make_test_case(
    llm: ChatLiteLLM, es_client: Elasticsearch, elastic_index_and_user: tuple[str, str], env: Settings
) -> Callable[[str, str, list[str]], LLMTestCase]:
    index_name, user_uuid = elastic_index_and_user

    retriever = get_parameterised_retriever(
        env=env,
        es=es_client,
        index_name=index_name,
    )

    rag_chain = build_retrieval_chain(
        llm=llm,
        retriever=retriever,
        tokeniser=get_tokeniser(),
        env=env,
    )

    def _make_test_case(prompt: str, expected_output: str, context: list[str]) -> LLMTestCase:
        prompt = ChainInput(
            question=prompt,
            file_uuids=[],
            user_uuid=str(user_uuid),
            chat_history=[],
        )
        answer = rag_chain.invoke(input=prompt.model_dump(mode="json"))

        return LLMTestCase(
            input=prompt,
            actual_output=answer["response"],
            expected_output=expected_output,
            context=context,
            retrieval_context=[doc.page_content for doc in answer["source_documents"]],
        )

    return _make_test_case


@pytest.mark.ai()
@pytest.mark.parametrize("test_data", RAW_TESTS)
def test_ai(make_test_case: Callable, eval_llm: DeepEvalBaseLLM, test_data: tuple[str, str, list[str]]):
    prompt, expected_output, context = test_data
    deepeval_test_case = make_test_case(prompt=prompt, expected_output=expected_output, context=context)

    contextual_precision = ContextualPrecisionMetric(
        threshold=0.5,  # default is 0.5, higher is better
        model=eval_llm,
    )
    contextual_recall = ContextualRecallMetric(
        threshold=0.5,  # default is 0.5, higher is better
        model=eval_llm,
    )
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.5,  # default is 0.5, higher is better
        model=eval_llm,
    )
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5,  # default is 0.5, higher is better
        model=eval_llm,
    )
    faithfulness = FaithfulnessMetric(
        threshold=0.5,  # default is 0.5, higher is better
        model=eval_llm,
    )
    hallucination = HallucinationMetric(
        threshold=0.5,  # default is 0.5, lower is better
        model=eval_llm,
    )

    evaluation_results: list[TestResult] = evaluate(
        test_cases=[deepeval_test_case],
        metrics=[
            contextual_recall,
            contextual_precision,
            contextual_relevancy,
            answer_relevancy,
            faithfulness,
            hallucination,
        ],
    )

    assert all(result.success for result in evaluation_results), evaluation_results
