import logging
import os
from functools import lru_cache
from typing import Annotated

import tiktoken
from elasticsearch import Elasticsearch
from fastapi import Depends
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField
from langchain_elasticsearch import ApproxRetrievalStrategy, ElasticsearchStore

from core_api.src.callbacks import LoggerCallbackHandler
from core_api.src.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever
from redbox.model_db import MODEL_PATH
from redbox.models import Settings
from redbox.storage import ElasticsearchStorageHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@lru_cache(1)
def get_env() -> Settings:
    return Settings()


@lru_cache(1)
def get_elasticsearch_client(env: Annotated[Settings, Depends(get_env)]) -> Elasticsearch:
    return env.elasticsearch_client()


@lru_cache(1)
def get_embedding_model(env: Annotated[Settings, Depends(get_env)]) -> Embeddings:
    embedding_model = SentenceTransformerEmbeddings(model_name=env.embedding_model, cache_folder=MODEL_PATH)
    log.info("Loaded embedding model from environment: %s", env.embedding_model)
    return embedding_model


@lru_cache(1)
def get_storage_handler(
    es: Annotated[Elasticsearch, Depends(get_elasticsearch_client)],
    env: Annotated[Settings, Depends(get_env)],
) -> ElasticsearchStorageHandler:
    return ElasticsearchStorageHandler(es_client=es, root_index=env.elastic_root_index)


@lru_cache(1)
def get_vector_store(
    env: Annotated[Settings, Depends(get_env)],
    es: Annotated[Elasticsearch, Depends(get_elasticsearch_client)],
) -> ElasticsearchStore:
    if env.elastic.subscription_level == "basic":
        strategy = ApproxRetrievalStrategy(hybrid=False)
    elif env.elastic.subscription_level in ["platinum", "enterprise"]:
        strategy = ApproxRetrievalStrategy(hybrid=True)
    else:
        message = f"Unknown Elastic subscription level {env.elastic.subscription_level}"
        raise ValueError(message)

    return ElasticsearchStore(
        es_connection=es,
        index_name=f"{env.elastic_root_index}-chunk",
        embedding=get_embedding_model(env),
        strategy=strategy,
        vector_query_field="embedding",
    )


@lru_cache(1)
def get_index_name(
    env: Annotated[Settings, Depends(get_env)],
) -> str:
    return f"{env.elastic_root_index}-chunk"


@lru_cache(1)
def get_parameterised_retriever(
    env: Annotated[Settings, Depends(get_env)],
    es: Annotated[Elasticsearch, Depends(get_elasticsearch_client)],
    index_name: Annotated[str, Depends(get_index_name)],
) -> BaseRetriever:
    """Creates an Elasticsearch retriever runnable.

    Runnable takes input of a dict keyed to question, file_uuids and user_uuid.

    Runnable returns a list of Chunks.
    """
    default_params = {
        "size": env.ai.rag_k,
        "num_candidates": env.ai.rag_num_candidates,
        "match_boost": 1,
        "knn_boost": 1,
        "similarity_threshold": 0,
    }
    return ParameterisedElasticsearchRetriever(
        es_client=es,
        index_name=index_name,
        params=default_params,
        embedding_model=get_embedding_model(env),
    ).configurable_fields(
        params=ConfigurableField(
            id="params", name="Retriever parameters", description="A dictionary of parameters to use for the retriever."
        )
    )


@lru_cache(1)
def get_all_chunks_retriever(
    env: Annotated[Settings, Depends(get_env)], es: Annotated[Elasticsearch, Depends(get_elasticsearch_client)]
):
    return AllElasticsearchRetriever(
        es_client=es,
        index_name=f"{env.elastic_root_index}-chunk",
    )


@lru_cache(1)
def get_llm(env: Annotated[Settings, Depends(get_env)]) -> ChatLiteLLM:
    logger_callback = LoggerCallbackHandler(logger=log)

    # Create the appropriate LLM, either openai, Azure, anthropic or bedrock
    if env.openai_api_key is not None:
        log.info("Creating OpenAI LLM Client")
        llm = ChatLiteLLM(
            streaming=True,
            openai_key=env.openai_api_key,
            callbacks=[logger_callback],
        )
    elif env.azure_openai_api_key is not None:
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
            streaming=True,
            api_base=env.azure_openai_endpoint,
            max_tokens=env.llm_max_tokens,
            callbacks=[logger_callback],
        )
    elif env.anthropic_api_key is not None:
        msg = "anthropic LLM not yet implemented"
        log.exception(msg)
        raise ValueError(msg)
    else:
        msg = "Unknown LLM model specified or missing"
        log.exception(msg)
        raise ValueError(msg)
    return llm


@lru_cache(1)
def get_tokeniser() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")
