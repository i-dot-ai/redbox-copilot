import logging
import os
from functools import lru_cache
from typing import Annotated

from redbox import Redbox
import tiktoken
from fastapi import Depends
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from redbox.api.callbacks import LoggerCallbackHandler
from redbox.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever
from redbox.models import Settings
from redbox.chains.components import get_embeddings

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
def get_llm(env: Annotated[Settings, Depends(get_env)]) -> ChatLiteLLM:
    logger_callback = LoggerCallbackHandler(logger=log)

    # Create the appropriate LLM, either openai, Azure, anthropic or bedrock
    if env.chat_backend == "openai":
        log.info("Creating OpenAI LLM Client")
        llm = ChatLiteLLM(
            streaming=True,
            openai_key=env.openai_api_key,
            callbacks=[logger_callback],
        )
    elif env.chat_backend == "azure":
        log.info("Creating Azure LLM Client")
        log.info("api_base: %s", env.azure_openai_endpoint)
        log.info("api_version: %s", env.openai_api_version)
        log.info("llm_max_tokens: %i", env.llm_max_tokens)

        # this nasty hack is required because, contrary to the docs:
        # using the api_version argument is not sufficient, and we need
        # to use the `OPENAI_API_VERSION` environment variable
        os.environ["AZURE_API_VERSION"] = env.openai_api_version
        os.environ["AZURE_OPENAI_API_KEY"] = env.azure_openai_api_key

        llm = ChatLiteLLM(
            model=env.azure_openai_model,
            azure_api_key=env.azure_openai_api_key,
            streaming=True,
            api_base=env.azure_openai_endpoint,
            max_tokens=env.llm_max_tokens,
            callbacks=[logger_callback],
        )
    else:
        msg = "Unknown LLM model specified or missing"
        log.exception(msg)
        raise ValueError(msg)
    return llm


@lru_cache(1)
def get_tokeniser() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def get_redbox(
    llm: Annotated[ChatLiteLLM, Depends(get_llm)],
    all_chunks_retriever: Annotated[AllElasticsearchRetriever, Depends(get_all_chunks_retriever)],
    parameterised_retriever: Annotated[ParameterisedElasticsearchRetriever, Depends(get_parameterised_retriever)],
    tokeniser: Annotated[tiktoken.Encoding, Depends(get_tokeniser)],
    env: Annotated[Settings, Depends(get_env)],
) -> Redbox:
    return Redbox(llm, all_chunks_retriever, parameterised_retriever, tokeniser, env, debug=True)
