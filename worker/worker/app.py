#!/usr/bin/env python

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch
from faststream import Context, ContextRepo, FastStream
from faststream.redis import RedisBroker
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_elasticsearch.vectorstores import ElasticsearchStore, BM25RetrievalStrategy

from redbox.chains.components import get_embeddings
from redbox.models import File, ProcessingStatusEnum, Settings
from redbox.storage.elasticsearch import ElasticsearchStorageHandler
from redbox.loader import UnstructuredLargeChunkLoader, UnstructuredTitleLoader
from redbox.chains.ingest import ingest_from_loader

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


def get_elasticsearch_store(es: Elasticsearch, es_index_name: str):
    return ElasticsearchStore(
        index_name=es_index_name,
        embedding=get_embeddings(env),
        es_connection=es,
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
    )


def get_elasticsearch_store_without_embeddings(es: Elasticsearch, es_index_name: str):
    return ElasticsearchStore(
        index_name=es_index_name, es_connection=es, query_field="text", strategy=BM25RetrievalStrategy()
    )


def get_elasticsearch_storage_handler(es: Elasticsearch):
    return ElasticsearchStorageHandler(es, env.elastic_root_index)


@asynccontextmanager
async def lifespan(context: ContextRepo):
    es = env.elasticsearch_client()
    es_index_name = f"{env.elastic_root_index}-chunk"

    es.indices.create(index=es_index_name, ignore=[400])
    context.set_global("storage_handler", get_elasticsearch_storage_handler(es))

    context.set_global(
        "chunk_ingest_chain",
        ingest_from_loader(
            document_loader_type=UnstructuredTitleLoader,
            s3_client=env.s3_client(),
            vectorstore=get_elasticsearch_store(es, es_index_name),
            env=env,
        ),
    )

    context.set_global(
        "large_chunk_ingest_chain",
        ingest_from_loader(
            document_loader_type=UnstructuredLargeChunkLoader,
            s3_client=env.s3_client(),
            vectorstore=get_elasticsearch_store_without_embeddings(es, es_index_name),
            env=env,
        ),
    )
    yield


@broker.subscriber(list=env.ingest_queue_name)
async def ingest(
    file: File,
    chunk_ingest_chain: Runnable = Context(),
    large_chunk_ingest_chain: Runnable = Context(),
    storage_handler: ElasticsearchStorageHandler = Context(),
):
    logging.info("Ingesting file: %s", file)

    file.ingest_status = ProcessingStatusEnum.embedding
    storage_handler.update_item(file)

    try:
        new_ids = await RunnableParallel({"normal": chunk_ingest_chain, "largest": large_chunk_ingest_chain}).ainvoke(
            file
        )
        file.ingest_status = ProcessingStatusEnum.complete
        logging.info("File: %s %s chunks ingested", file, {k: len(v) for k, v in new_ids.items()})
    except Exception:
        logging.exception("Error while processing file [%s]", file)
        file.ingest_status = ProcessingStatusEnum.failed
    finally:
        storage_handler.update_item(file)


app = FastStream(broker=broker, lifespan=lifespan)
