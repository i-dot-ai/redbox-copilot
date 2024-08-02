import logging
import os

from langchain_community.chat_models import ChatLiteLLM
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.utils import convert_to_secret_str
import tiktoken

from redbox.api.callbacks import LoggerCallbackHandler
from redbox.models.chain import AISettings
from redbox.models.settings import Settings
from redbox.retriever import AllElasticsearchRetriever, ParameterisedElasticsearchRetriever
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

log = logging.getLogger()


def get_llm(ai_settings: AISettings) -> ChatLiteLLM:
    logger_callback = LoggerCallbackHandler(logger=log)

    # Create the appropriate LLM, either openai, Azure, anthropic or bedrock
    if ai_settings.chat_backend == "openai":
        log.info("Creating OpenAI LLM Client")
        llm = ChatLiteLLM(
            streaming=True,
            openai_key=ai_settings.openai_api_key,
            callbacks=[logger_callback],
        )
    elif ai_settings.chat_backend == "azure":
        log.info("Creating Azure LLM Client")
        log.info("api_base: %s", ai_settings.azure_openai_endpoint)
        log.info("api_version: %s", ai_settings.openai_api_version)
        log.info("llm_max_tokens: %i", ai_settings.llm_max_tokens)

        # this nasty hack is required because, contrary to the docs:
        # using the api_version argument is not sufficient, and we need
        # to use the `OPENAI_API_VERSION` environment variable
        os.environ["AZURE_API_VERSION"] = ai_settings.openai_api_version
        os.environ["AZURE_OPENAI_API_KEY"] = ai_settings.azure_openai_api_key

        llm = ChatLiteLLM(
            model=ai_settings.azure_openai_model,
            azure_api_key=ai_settings.azure_openai_api_key,
            streaming=True,
            api_base=ai_settings.azure_openai_endpoint,
            max_tokens=ai_settings.llm_max_tokens,
            callbacks=[logger_callback],
        )
    elif ai_settings.chat_backend == "fake":
        return GenericFakeChatModel(messages=iter(ai_settings.fake_backend_responses))
    else:
        msg = "Unknown LLM model specified or missing"
        log.exception(msg)
        raise ValueError(msg)
    return llm


def get_tokeniser() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def get_azure_embeddings(env: Settings):
    return AzureOpenAIEmbeddings(
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
