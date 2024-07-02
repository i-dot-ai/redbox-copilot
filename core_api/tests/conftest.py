import time
from collections.abc import Generator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from botocore.exceptions import ClientError
from elasticsearch import Elasticsearch
from fastapi.testclient import TestClient
from jose import jwt
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.fake import FakeListLLM
from langchain_core.documents.base import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.runnables import ConfigurableField
from langchain_elasticsearch import ElasticsearchStore

from core_api.src.app import app as application
from core_api.src.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever
from core_api.tests.retriever.data import ALL_CHUNKS_RETRIEVER_DOCUMENTS, PARAMETERISED_RETRIEVER_DOCUMENTS
from redbox.model_db import MODEL_PATH
from redbox.models import Chunk, File, Settings
from redbox.storage import ElasticsearchStorageHandler


@pytest.fixture(scope="session")
def env():
    return Settings()


@pytest.fixture(scope="session")
def s3_client(env):
    _client = env.s3_client()
    try:
        _client.create_bucket(
            Bucket=env.bucket_name,
            CreateBucketConfiguration={"LocationConstraint": env.aws_region},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise

    return _client


@pytest.fixture(scope="session")
def es_client(env) -> Elasticsearch:
    return env.elasticsearch_client()


@pytest.fixture()
def app_client() -> TestClient:
    return TestClient(application)


@pytest.fixture()
def alice() -> UUID:
    return uuid4()


@pytest.fixture()
def headers(alice):
    bearer_token = jwt.encode({"user_uuid": str(alice)}, key="nvjkernd")
    return {"Authorization": f"Bearer {bearer_token}"}


@pytest.fixture()
def elasticsearch_storage_handler(es_client, env):
    return ElasticsearchStorageHandler(es_client=es_client, root_index=env.elastic_root_index)


@pytest.fixture()
def file(s3_client, file_pdf_path: Path, alice, env) -> File:
    file_name = file_pdf_path.name
    file_type = file_pdf_path.suffix

    with file_pdf_path.open("rb") as f:
        s3_client.put_object(
            Bucket=env.bucket_name,
            Body=f.read(),
            Key=file_name,
            Tagging=f"file_type={file_type}",
        )

    return File(key=file_name, bucket=env.bucket_name, creator_user_uuid=alice)


@pytest.fixture()
def large_stored_file(elasticsearch_storage_handler, file) -> File:
    elasticsearch_storage_handler.write_item(file)
    elasticsearch_storage_handler.refresh()
    return file


@pytest.fixture()
def stored_file_1(elasticsearch_storage_handler, file) -> File:
    elasticsearch_storage_handler.write_item(file)
    elasticsearch_storage_handler.refresh()
    return file


@pytest.fixture()
def embedding_model_dim(embedding_model) -> int:
    return len(embedding_model.embed_query("foo"))


@pytest.fixture()
def stored_file_chunks(stored_file_1, embedding_model_dim) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i in range(5):
        chunks.append(
            Chunk(
                text="hello",
                index=i,
                embedding=[1] * embedding_model_dim,
                parent_file_uuid=stored_file_1.uuid,
                creator_user_uuid=stored_file_1.creator_user_uuid,
                metadata={"parent_doc_uuid": str(stored_file_1.uuid)},
            )
        )
    return chunks


@pytest.fixture()
def stored_large_file_chunks(stored_file_1, embedding_model_dim) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i in range(5):
        chunks.append(
            Chunk(
                text="hello " * 100,
                index=i,
                embedding=[1] * embedding_model_dim,
                parent_file_uuid=stored_file_1.uuid,
                creator_user_uuid=stored_file_1.creator_user_uuid,
                metadata={"parent_doc_uuid": str(stored_file_1.uuid)},
            )
        )
    return chunks


@pytest.fixture(params=ALL_CHUNKS_RETRIEVER_DOCUMENTS)
def stored_file_all_chunks(request, env, es_client) -> Generator[list[Document], None, None]:
    store = ElasticsearchStore(
        index_name=env.elastic_root_index + "-chunk", es_connection=es_client, query_field="text"
    )
    documents = list(map(Document.parse_obj, request.param))
    doc_ids = store.add_documents(documents)
    yield documents
    store.delete(doc_ids)


@pytest.fixture(params=PARAMETERISED_RETRIEVER_DOCUMENTS)
def stored_file_parameterised(request, env, es_client) -> Generator[list[Document], None, None]:
    store = ElasticsearchStore(
        index_name=env.elastic_root_index + "-chunk", es_connection=es_client, query_field="text"
    )
    documents = list(map(Document.parse_obj, request.param))
    doc_ids = store.add_documents(documents)
    yield documents
    store.delete(doc_ids)


@pytest.fixture()
def other_stored_file_chunks(stored_file_1) -> list[Chunk]:
    new_uuid = uuid4()
    chunks: list[Chunk] = []
    for i in range(5):
        chunks.append(
            Chunk(
                text="hello",
                index=i,
                parent_file_uuid=new_uuid,
                creator_user_uuid=stored_file_1.creator_user_uuid,
                embedding=[1] * 768,
                metadata={"parent_doc_uuid": str(new_uuid)},
            )
        )
    return chunks


@pytest.fixture()
def chunked_file(elasticsearch_storage_handler, stored_file_chunks, stored_file_1) -> File:
    for chunk in stored_file_chunks:
        elasticsearch_storage_handler.write_item(chunk)
    elasticsearch_storage_handler.refresh()
    time.sleep(1)
    return stored_file_1


@pytest.fixture()
def large_chunked_file(elasticsearch_storage_handler, stored_large_file_chunks, stored_file_1) -> File:
    for chunk in stored_large_file_chunks:
        elasticsearch_storage_handler.write_item(chunk)
    elasticsearch_storage_handler.refresh()
    time.sleep(1)
    return stored_file_1


@pytest.fixture()
def file_pdf_path() -> Path:
    return Path(__file__).parents[2] / "tests" / "data" / "pdf" / "Cabinet Office - Wikipedia.pdf"


@pytest.fixture()
def mock_llm():
    return FakeListLLM(responses=["<<TESTING>>"] * 128)


@pytest.fixture()
def embedding_model(env) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=env.embedding_model, cache_folder=MODEL_PATH)


@pytest.fixture()
def chunk_index_name(env):
    return f"{env.elastic_root_index}-chunk"


@pytest.fixture()
def all_chunks_retriever(env, es_client) -> AllElasticsearchRetriever:
    return AllElasticsearchRetriever(
        es_client=es_client,
        index_name=f"{env.elastic_root_index}-chunk",
    )


@pytest.fixture()
def parameterised_retriever(env, es_client, embedding_model_dim) -> ParameterisedElasticsearchRetriever:
    default_params = {
        "size": env.ai.rag_k,
        "num_candidates": env.ai.rag_num_candidates,
        "match_boost": 1,
        "knn_boost": 1,
        "similarity_threshold": 0,
    }
    return ParameterisedElasticsearchRetriever(
        es_client=es_client,
        index_name=f"{env.elastic_root_index}-chunk",
        params=default_params,
        embedding_model=FakeEmbeddings(size=embedding_model_dim),
    ).configurable_fields(
        params=ConfigurableField(
            id="params", name="Retriever parameters", description="A dictionary of parameters to use for the retriever."
        )
    )


# Pytest internal functions for skipping tests unless selected
# See https://docs.pytest.org/en/latest/example/simple.html
# #control-skipping-of-tests-according-to-command-line-option


def pytest_addoption(parser):
    parser.addoption("--ai", action="store_true", default=False, help="run ai tests that use live LLMs")


def pytest_configure(config):
    config.addinivalue_line("markers", "ai: mark test as using live LLMs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ai"):
        # --ai given in cli: do not skip live LLM tests
        return
    skip_ai = pytest.mark.skip(reason="need --ai option to run")
    for item in items:
        if "ai" in item.keywords:
            item.add_marker(skip_ai)
