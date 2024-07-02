#!/usr/bin/env python

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch
from faststream import Context, ContextRepo, FastStream
from faststream.redis import RedisBroker
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnableLambda, chain
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_openai.embeddings import AzureOpenAIEmbeddings

from redbox.models import File, ProcessingStatusEnum, Settings
from redbox.storage.elasticsearch import ElasticsearchStorageHandler
from worker.src.loader import UnstructuredDocumentLoader

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

start_time = datetime.now(UTC)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

env = Settings()


broker = RedisBroker(url=env.redis_url)

publisher = broker.publisher(list=env.embed_queue_name)


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=env.azure_openai_endpoint,
        api_version=env.azure_api_version_embeddings,
        model=env.azure_embedding_model,
        max_retries=env.embedding_max_retries,
        retry_min_seconds=env.embedding_retry_min_seconds,
        retry_max_seconds=env.embedding_retry_max_seconds,
        chunk_size=env.embedding_max_batch_size,
    )


def get_elasticsearch_store(es: Elasticsearch, es_index_name: str):
    return ElasticsearchStore(
        index_name=es_index_name,
        embedding=get_embeddings(),
        es_connection=es,
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
    )


def get_elasticsearch_storage_handler(es: Elasticsearch):
    return ElasticsearchStorageHandler(es, env.elastic_root_index)


@asynccontextmanager
async def lifespan(context: ContextRepo):
    es = env.elasticsearch_client()
    es_index_name = f"{env.elastic_root_index}-chunk"

    es.indices.create(index=es_index_name, ignore=[400])
    context.set_global("vectorstore", get_elasticsearch_store(es, es_index_name))
    context.set_global("s3_client", env.s3_client())
    context.set_global("storage_handler", get_elasticsearch_storage_handler(es))
    yield


def document_loader(s3_client: S3Client, env: Settings):
    @chain
    def wrapped(file: File):
        file_raw = BytesIO()
        s3_client.download_fileobj(Bucket=file.bucket, Key=file.key, Fileobj=file_raw)
        file_raw.seek(0)
        return UnstructuredDocumentLoader(file=file, file_bytes=file_raw, env=env).lazy_load()

    return wrapped


@chain
def log_chunks(chunks: list[Document]):
    log.info("Processing %s chunks", len(chunks))
    return chunks


@broker.subscriber(list=env.ingest_queue_name)
async def ingest(
    file: File,
    s3_client: S3Client = Context(),
    vectorstore: VectorStore = Context(),
    storage_handler: ElasticsearchStorageHandler = Context(),
):
    logging.info("Ingesting file: %s", file)

    file.ingest_status = ProcessingStatusEnum.embedding
    storage_handler.update_item(file)

    try:
        new_ids = await (
            document_loader(s3_client=s3_client, env=env)
            | RunnableLambda(list)
            | log_chunks
            | RunnableLambda(vectorstore.aadd_documents)  # type: ignore[arg-type]
        ).ainvoke(file)
        file.ingest_status = ProcessingStatusEnum.complete
        logging.info("File: %s [%s] chunks ingested", file, len(new_ids))
    except Exception:
        logging.exception("Error while processing file [%s]", file)
        file.ingest_status = ProcessingStatusEnum.failed
    finally:
        storage_handler.update_item(file)


app = FastStream(broker=broker, lifespan=lifespan)
