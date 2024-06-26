from pathlib import Path
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError
from elasticsearch import Elasticsearch
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer

from redbox.models import Chunk, EmbedQueueItem, File
from redbox.storage import ElasticsearchStorageHandler
from worker.src.app import app, env


@pytest.fixture(scope="session")
def s3_client():
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
def es_client() -> Elasticsearch:
    return env.elasticsearch_client()


@pytest.fixture()
def embedding_model() -> SentenceTransformer:
    return SentenceTransformer(env.embedding_model)


@pytest.fixture(scope="session")
def file_pdf_path() -> Path:
    return Path(__file__).parents[2] / "tests" / "data" / "pdf" / "Cabinet Office - Wikipedia.pdf"


@pytest.fixture(scope="session")
def file(s3_client, file_pdf_path: Path):
    file_name = file_pdf_path.name
    file_type = file_pdf_path.suffix

    with file_pdf_path.open("rb") as f:
        s3_client.put_object(
            Bucket=env.bucket_name,
            Body=f.read(),
            Key=file_name,
            Tagging=f"file_type={file_type}",
        )

    return File(key=file_name, bucket=env.bucket_name, creator_user_uuid=uuid4())


@pytest.fixture()
def app_client():
    return TestClient(app)


@pytest.fixture()
def elasticsearch_storage_handler(
    es_client,
) -> ElasticsearchStorageHandler:
    return ElasticsearchStorageHandler(es_client=es_client, root_index=env.elastic_root_index)


@pytest.fixture()
def chunk() -> Chunk:
    return Chunk(parent_file_uuid=uuid4(), index=1, text="test_text", creator_user_uuid=uuid4())


@pytest.fixture()
def stored_chunk(chunk, elasticsearch_storage_handler) -> Chunk:
    elasticsearch_storage_handler.write_item(chunk)
    return chunk


@pytest.fixture()
def embed_queue_item(stored_chunk) -> EmbedQueueItem:
    return EmbedQueueItem(chunk_uuid=stored_chunk.uuid)
