#!/usr/bin/env python

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from faststream import Context, ContextRepo, FastStream
from faststream.redis import RedisBroker
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnableLambda, chain
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch.vectorstores import ElasticsearchStore

from redbox.models import File, Settings
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


@asynccontextmanager
async def lifespan(context: ContextRepo):
    es_index_name = f"{env.elastic_root_index}-chunk"
    es = env.elasticsearch_client()
    s3_client = env.s3_client()
    # embeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint=env.azure_openai_endpoint,
    #     openai_api_version="2023-05-15",
    #     model=env.azure_embedding_model,
    #     max_retries=env.embedding_max_retries,
    #     retry_min_seconds=4,
    #     retry_max_seconds=30
    # )
    embeddings = SentenceTransformerEmbeddings(model_name=env.embedding_model)
    elasticsearch_store = ElasticsearchStore(
        index_name=es_index_name,
        embedding=embeddings,
        es_connection=es,
        query_field="text",
        vector_query_field=env.embedding_document_field_name,
    )

    es.indices.create(index=es_index_name, ignore=[400])
    context.set_global("vectorstore", elasticsearch_store)
    context.set_global("s3_client", s3_client)
    yield


def document_loader(s3_client: S3Client, env: Settings):
    @chain
    def wrapped(file: File):
        return UnstructuredDocumentLoader(file, s3_client, env).lazy_load()

    return wrapped


@broker.subscriber(list=env.ingest_queue_name)
async def ingest(file: File, s3_client: S3Client = Context(), vectorstore: VectorStore = Context()):
    logging.info("Ingesting file: %s", file)

    new_ids = (
        document_loader(s3_client=s3_client, env=env) | RunnableLambda(list) | RunnableLambda(vectorstore.add_documents)
    ).invoke(file)

    logging.info("File: %s [%s] chunks ingested", file, len(new_ids))


app = FastStream(broker=broker, lifespan=lifespan)
