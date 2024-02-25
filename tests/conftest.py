import os
from typing import Generator, TypeVar

import pytest
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from redbox.models import Chunk, Settings
from redbox.storage.elasticsearch import ElasticsearchStorageHandler
from fastapi.testclient import TestClient


T = TypeVar("T")

YieldFixture = Generator[T, None, None]

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env.test")


env = Settings(_env_file=env_path)  # type: ignore


@pytest.fixture
def chunk() -> Chunk:
    test_chunk = Chunk(
        uuid="aaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        parent_file_uuid="test_uuid",
        index=1,
        text="test_text",
        metadata={},
        creator_user_uuid="test",
    )
    return test_chunk


def file_pdf_path() -> str:
    return "tests/data/pdf/Cabinet Office - Wikipedia.pdf"


# def setup_elasticsearch():
#     es = Elasticsearch(env.elastic_host)
#     for index_name, schema in schemas.items():
#         body = {
#             "settings": {
#                 "number_of_shards": 1,
#                 "number_of_replicas": 1,
#                 "index.store.type": "mmapfs",
#             },
#             "mappings": schema,
#         }
#         es.indices.create(index=index_name, body=body)
#
#
# def teardown_elasticsearch():
#     es = Elasticsearch(env.elastic_host)
#     for index_name in schemas.keys():
#         es.indices.delete(index=index_name)


@pytest.fixture
def elasticsearch_client() -> YieldFixture[Elasticsearch]:
    # setup_elasticsearch()
    yield env.elasticsearch_client()
    # teardown_elasticsearch()


@pytest.fixture
def elasticsearch_storage_handler(elasticsearch_client):
    yield ElasticsearchStorageHandler(
        es_client=elasticsearch_client, root_index="redbox-test-data"
    )


@pytest.fixture
def client():
    from app.workers.embed.app import app as application

    yield TestClient(application)


@pytest.fixture
def example_modes():
    from app.workers.embed.app import models as db

    db["paraphrase-albert-small-v2"] = SentenceTransformer(
        model_name_or_path="paraphrase-albert-small-v2",
        cache_folder="./models",
    )
    yield db
