from pathlib import Path
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch
import tiktoken

from redbox.models import File, Settings
from redbox.storage.elasticsearch import ElasticsearchStorageHandler

from collections.abc import Generator

from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_elasticsearch import ElasticsearchStore

from redbox.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever
from redbox.test.data import RedboxChatTestCase
from tests.retriever.data import ALL_CHUNKS_RETRIEVER_CASES, PARAMETERISED_RETRIEVER_CASES


@pytest.fixture(scope="session")
def env():
    return Settings(django_secret_key="", postgres_password="")


@pytest.fixture()
def alice():
    return uuid4()


@pytest.fixture()
def bob():
    return uuid4()


@pytest.fixture()
def claire():
    return uuid4()


@pytest.fixture()
def file_belonging_to_alice(
    file_pdf_path, alice, env, elasticsearch_storage_handler: ElasticsearchStorageHandler
) -> File:
    f = File(
        key=file_pdf_path.name,
        bucket=env.bucket_name,
        creator_user_uuid=alice,
    )
    elasticsearch_storage_handler.write_item(f)
    elasticsearch_storage_handler.refresh()
    return f


@pytest.fixture()
def file_belonging_to_bob(file_pdf_path, bob, env, elasticsearch_storage_handler: ElasticsearchStorageHandler) -> File:
    f = File(
        key=file_pdf_path.name,
        bucket=env.bucket_name,
        creator_user_uuid=bob,
    )
    elasticsearch_storage_handler.write_item(f)
    elasticsearch_storage_handler.refresh()
    return f


@pytest.fixture()
def file_belonging_to_claire(
    file_pdf_path, claire, env, elasticsearch_storage_handler: ElasticsearchStorageHandler
) -> File:
    f = File(
        key=file_pdf_path.name,
        bucket=env.bucket_name,
        creator_user_uuid=claire,
    )
    elasticsearch_storage_handler.write_item(f)
    elasticsearch_storage_handler.refresh()
    return f


@pytest.fixture
def file_pdf_path() -> Path:
    return Path(__file__).parents[2] / "tests" / "data" / "pdf" / "Cabinet Office - Wikipedia.pdf"


@pytest.fixture()
def elasticsearch_client(env) -> Elasticsearch:
    return env.elasticsearch_client()


@pytest.fixture()
def elasticsearch_storage_handler(elasticsearch_client, env) -> ElasticsearchStorageHandler:
    return ElasticsearchStorageHandler(es_client=elasticsearch_client, root_index=env.elastic_root_index)


@pytest.fixture(scope="session")
def es_index(env) -> str:
    return f"{env.elastic_root_index}-chunk"


@pytest.fixture(scope="session")
def es_index_file(env) -> str:
    return f"{env.elastic_root_index}-file"


@pytest.fixture(autouse=True, scope="session")
def create_index(env, es_index, es_index_file):
    es = env.elasticsearch_client()
    if not es.indices.exists(index=es_index):
        es.indices.create(index=es_index)
    if not es.indices.exists(index=es_index_file):
        es.indices.create(index=es_index_file)
    yield
    es.indices.delete(index=es_index)
    es.indices.delete(index=es_index_file)


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


@pytest.fixture(params=ALL_CHUNKS_RETRIEVER_CASES)
def stored_file_all_chunks(
    request, elasticsearch_client, es_index, embedding_model_dim
) -> Generator[RedboxChatTestCase, None, None]:
    test_case: RedboxChatTestCase = request.param
    store = ElasticsearchStore(
        index_name=es_index,
        es_connection=elasticsearch_client,
        query_field="text",
        embedding=FakeEmbeddings(size=embedding_model_dim),
    )
    doc_ids = store.add_documents(test_case.docs)
    yield test_case
    store.delete(doc_ids)


@pytest.fixture(params=PARAMETERISED_RETRIEVER_CASES)
def stored_file_parameterised(
    request, elasticsearch_client, es_index, embedding_model, env: Settings
) -> Generator[RedboxChatTestCase, None, None]:
    test_case: RedboxChatTestCase = request.param
    store = ElasticsearchStore(
        index_name=es_index,
        es_connection=elasticsearch_client,
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
        embedding=embedding_model,
    )
    doc_ids = store.add_documents(test_case.docs)
    yield test_case
    store.delete(doc_ids)


@pytest.fixture()
def all_chunks_retriever(elasticsearch_client, es_index) -> AllElasticsearchRetriever:
    return AllElasticsearchRetriever(
        es_client=elasticsearch_client,
        index_name=es_index,
    )


@pytest.fixture()
def parameterised_retriever(
    env, elasticsearch_client, es_index, embedding_model_dim
) -> ParameterisedElasticsearchRetriever:
    return ParameterisedElasticsearchRetriever(
        es_client=elasticsearch_client,
        index_name=es_index,
        embedding_model=FakeEmbeddings(size=embedding_model_dim),
        embedding_field_name=env.embedding_document_field_name,
    )


@pytest.fixture(scope="session")
def embedding_model_dim() -> int:
    return 3072  # 3-large default size


@pytest.fixture(scope="session")
def embedding_model(embedding_model_dim):
    return FakeEmbeddings(size=embedding_model_dim)


@pytest.fixture(scope="session")
def tokeniser():
    return tiktoken.get_encoding("cl100k_base")
