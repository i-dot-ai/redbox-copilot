from functools import cache
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.utils import convert_to_secret_str
import tiktoken

from redbox.models.settings import Settings
from redbox.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever


def get_chat_llm(env: Settings):
    return AzureChatOpenAI(
        api_key=convert_to_secret_str(env.azure_openai_api_key),
        azure_endpoint=env.azure_openai_endpoint,
        model=env.azure_openai_model,
        api_version=env.azure_api_version_embeddings,
    )


@cache
def get_tokeniser() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def get_azure_embeddings(env: Settings):
    return AzureOpenAIEmbeddings(
        api_key=convert_to_secret_str(env.azure_openai_api_key),
        azure_endpoint=env.azure_openai_endpoint,
        api_version=env.azure_api_version_embeddings,
        model=env.azure_embedding_model,
        max_retries=env.embedding_max_retries,
        retry_min_seconds=env.embedding_retry_min_seconds,
        retry_max_seconds=env.embedding_retry_max_seconds,
    )


def get_openai_embeddings(env: Settings):
    return OpenAIEmbeddings(
        api_key=convert_to_secret_str(env.openai_api_key),
        base_url=env.embedding_openai_base_url,
        model=env.embedding_openai_model,
        chunk_size=env.embedding_max_batch_size,
    )


def get_embeddings(env: Settings) -> Embeddings:
    if env.embedding_backend == "azure":
        return get_azure_embeddings(env)
    elif env.embedding_backend == "openai":
        return get_openai_embeddings(env)
    elif env.embedding_backend == "fake":
        return FakeEmbeddings(size=3072)  # TODO
    else:
        raise Exception("No configured embedding model")


def get_all_chunks_retriever(env: Settings) -> ElasticsearchRetriever:
    return AllElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=f"{env.elastic_root_index}-chunk",
    )


def get_parameterised_retriever(env: Settings, embeddings: Embeddings | None = None):
    """Creates an Elasticsearch retriever runnable.

    Runnable takes input of a dict keyed to question, file_uuids and user_uuid.

    Runnable returns a list of Chunks.
    """
    return ParameterisedElasticsearchRetriever(
        es_client=env.elasticsearch_client(),
        index_name=f"{env.elastic_root_index}-chunk",
        embedding_model=embeddings or get_embeddings(env),
        embedding_field_name=env.embedding_document_field_name,
    )
