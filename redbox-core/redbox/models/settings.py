import logging
from typing import Literal

import boto3
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


VANILLA_SYSTEM_PROMPT = (
    "You are an AI assistant called Redbox tasked with answering questions and providing information objectively."
)

RETRIEVAL_SYSTEM_PROMPT = (
    "Given the following conversation and extracted parts of a long document and a question, create a final answer. \n"
    "If you don't know the answer, just say that you don't know. Don't try to make up an answer. "
    "If a user asks for a particular format to be returned, such as bullet points, then please use that format. "
    "If a user asks for bullet points you MUST give bullet points. "
    "If the user asks for a specific number or range of bullet points you MUST give that number of bullet points. \n"
    "Use **bold** to highlight the most question relevant parts in your response. "
    "If dealing dealing with lots of data return it in markdown table format. "
)

SUMMARISATION_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to extract the most important information and present it in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

MAP_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to extract the most important information and present it in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

REDUCE_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to write a concise summary of list of summaries from a list of summaries in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

CONDENSE_SYSTEM_PROMPT = (
    "Given the following conversation and a follow up question, generate a follow "
    "up question to be a standalone question. "
    "You are only allowed to generate one question in response. "
    "Include sources from the chat history in the standalone question created, "
    "when they are available. "
    "If you don't know the answer, just say that you don't know, "
    "don't try to make up an answer. \n"
)

VANILLA_QUESTION_PROMPT = "{question}\n=========\n Response: "

RETRIEVAL_QUESTION_PROMPT = "{question} \n=========\n{formatted_documents}\n=========\nFINAL ANSWER: "

SUMMARISATION_QUESTION_PROMPT = "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "

MAP_QUESTION_PROMPT = "Question: {question}. "

MAP_DOCUMENT_PROMPT = "\n\n Documents: \n\n {documents} \n\n Answer: "

REDUCE_QUESTION_PROMPT = "Question: {question}. \n\n Documents: \n\n {summaries} \n\n Answer: "

CONDENSE_QUESTION_PROMPT = "{question}\n=========\n Standalone question: "


class AISettings(BaseModel):
    """prompts and other AI settings"""

    model_config = SettingsConfigDict(frozen=True)

    context_window_size: int = 8_000
    rag_k: int = 15
    rag_num_candidates: int = 10
    rag_desired_chunk_size: int = 300
    summarisation_chunk_max_tokens: int = 20_000
    summarisation_max_concurrency: int = 128
    vanilla_system_prompt: str = VANILLA_SYSTEM_PROMPT
    vanilla_question_prompt: str = VANILLA_QUESTION_PROMPT
    retrieval_system_prompt: str = RETRIEVAL_SYSTEM_PROMPT
    retrieval_question_prompt: str = RETRIEVAL_QUESTION_PROMPT
    condense_system_prompt: str = CONDENSE_SYSTEM_PROMPT
    condense_question_prompt: str = CONDENSE_QUESTION_PROMPT
    summarisation_system_prompt: str = SUMMARISATION_SYSTEM_PROMPT
    summarisation_question_prompt: str = SUMMARISATION_QUESTION_PROMPT
    map_system_prompt: str = MAP_SYSTEM_PROMPT
    map_question_prompt: str = MAP_QUESTION_PROMPT
    map_document_prompt: str = MAP_DOCUMENT_PROMPT
    reduce_system_prompt: str = REDUCE_SYSTEM_PROMPT
    reduce_question_prompt: str = REDUCE_QUESTION_PROMPT


class ElasticLocalSettings(BaseModel):
    """settings required for a local/ec2 instance of elastic"""

    model_config = SettingsConfigDict(frozen=True)

    host: str = "elasticsearch"
    port: int = 9200
    scheme: str = "http"
    user: str = "elastic"
    version: str = "8.11.0"
    password: str = "redboxpass"
    subscription_level: str = "basic"


class ElasticCloudSettings(BaseModel):
    """settings required for elastic-cloud"""

    model_config = SettingsConfigDict(frozen=True)

    api_key: str
    cloud_id: str
    subscription_level: str = "basic"


class Settings(BaseSettings):
    """Settings for the redbox application."""

    ai: AISettings = AISettings()

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None

    openai_api_version: str = "2023-12-01-preview"
    azure_api_version_embeddings: str = "2024-02-01"
    azure_openai_model: str = "azure/gpt-35-turbo-16k"
    azure_embedding_model: str = "text-embedding-3-large"
    llm_max_tokens: int = 1024

    embedding_max_retries: int = 10
    embedding_retry_min_seconds: int = 10
    embedding_retry_max_seconds: int = 120
    embedding_max_batch_size: int = 512
    embedding_document_field_name: str = "embedding"

    partition_strategy: Literal["auto", "fast", "ocr_only", "hi_res"] = "fast"
    clustering_strategy: Literal["full"] | None = None

    elastic: ElasticCloudSettings | ElasticLocalSettings = ElasticLocalSettings()
    elastic_root_index: str = "redbox-data"

    kibana_system_password: str = "redboxpass"
    metricbeat_internal_password: str = "redboxpass"
    filebeat_internal_password: str = "redboxpass"
    heartbeat_internal_password: str = "redboxpass"
    monitoring_internal_password: str = "redboxpass"
    beats_system_password: str = "redboxpass"

    minio_host: str = "minio"
    minio_port: int = 9000
    aws_access_key: str | None = None
    aws_secret_key: str | None = None

    aws_region: str = "eu-west-2"
    bucket_name: str = "redbox-storage-dev"
    embedding_model: str = "all-mpnet-base-v2"

    embed_queue_name: str = "redbox-embedder-queue"
    ingest_queue_name: str = "redbox-ingester-queue"

    worker_ingest_min_chunk_size: int = 120
    worker_ingest_max_chunk_size: int = 300

    redis_host: str = "redis"
    redis_port: int = 6379

    object_store: str = "minio"

    dev_mode: bool = False
    superuser_email: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="allow", frozen=True)

    def elasticsearch_client(self) -> Elasticsearch:
        if isinstance(self.elastic, ElasticLocalSettings):
            log.info("Connecting to self managed Elasticsearch")
            log.info("Elasticsearch host = %s", self.elastic.host)
            return Elasticsearch(
                hosts=[
                    {
                        "host": self.elastic.host,
                        "port": self.elastic.port,
                        "scheme": self.elastic.scheme,
                    }
                ],
                basic_auth=(self.elastic.user, self.elastic.password),
            )

        log.info("Connecting to Elastic Cloud Cluster")
        log.info("Cloud ID = %s", self.elastic.cloud_id)
        log.info("Elastic Cloud API Key = %s", self.elastic.api_key)

        return Elasticsearch(cloud_id=self.elastic.cloud_id, api_key=self.elastic.api_key)

    def s3_client(self):
        if self.object_store == "minio":
            client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key or "",
                aws_secret_access_key=self.aws_secret_key or "",
                endpoint_url=f"http://{self.minio_host}:{self.minio_port}",
            )

        elif self.object_store == "s3":
            client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region,
            )
        elif self.object_store == "moto":
            from moto import mock_aws

            mock = mock_aws()
            mock.start()

            client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region,
            )
        else:
            raise NotImplementedError

        return client

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/"
