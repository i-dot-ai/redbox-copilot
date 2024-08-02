import logging
from functools import lru_cache
from typing import Annotated

from redbox import Redbox
import tiktoken
from fastapi import Depends
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from redbox.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever
from redbox.models import Settings
from redbox.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@lru_cache(1)
def get_env() -> Settings:
    return Settings()


def get_embedding_model(env: Annotated[Settings, Depends(get_env)]) -> Embeddings:
    return get_embeddings(env)


def get_parameterised_retriever(
    env: Annotated[Settings, Depends(get_env)], embeddings: Annotated[Embeddings, Depends(get_embedding_model)]
) -> BaseRetriever:
    """Creates an Elasticsearch retriever runnable.

    Runnable takes input of a dict keyed to question, file_uuids and user_uuid.

    Runnable returns a list of Chunks.
    """
    return ParameterisedElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=f"{env.elastic_root_index}-chunk",
        embedding_model=embeddings,
        embedding_field_name=env.embedding_document_field_name,
    )


@lru_cache(1)
def get_all_chunks_retriever(env: Annotated[Settings, Depends(get_env)]):
    return AllElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=f"{env.elastic_root_index}-chunk",
    )


@lru_cache(1)
def get_tokeniser() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def get_redbox(
    all_chunks_retriever: Annotated[AllElasticsearchRetriever, Depends(get_all_chunks_retriever)],
    parameterised_retriever: Annotated[ParameterisedElasticsearchRetriever, Depends(get_parameterised_retriever)],
    tokeniser: Annotated[tiktoken.Encoding, Depends(get_tokeniser)],
    env: Annotated[Settings, Depends(get_env)],
) -> Redbox:
    return Redbox(all_chunks_retriever, parameterised_retriever, tokeniser, env)
