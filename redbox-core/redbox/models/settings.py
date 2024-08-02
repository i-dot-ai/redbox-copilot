import logging
from functools import lru_cache
from typing import Literal

import boto3
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger()


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

    anthropic_api_key: str | None = None
    openai_api_key: str = "NotAKey"
    azure_openai_api_key: str = "NotAKey"
    azure_openai_endpoint: str | None = None

    azure_api_version_embeddings: str = "2024-02-01"
    azure_embedding_model: str = "text-embedding-3-large"

    embedding_backend: Literal["azure", "openai", "fake"] = "azure"
    embedding_max_retries: int = 10
    embedding_retry_min_seconds: int = 10
    embedding_retry_max_seconds: int = 120
    embedding_max_batch_size: int = 512
    embedding_document_field_name: str = "embedding"

    embedding_openai_base_url: str | None = None
    embedding_openai_model: str = "text-embedding-ada-002"

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

    ## Chunks
    ### Normal
    worker_ingest_min_chunk_size: int = 120
    worker_ingest_max_chunk_size: int = 300
    ### Largest
    worker_ingest_largest_chunk_size: int = 96000
    worker_ingest_largest_chunk_overlap: int = 0

    response_no_doc_available: str = "No available data for selected files. They may need to be removed and added again"
    response_max_content_exceeded: str = "Max content exceeded. Try smaller or fewer documents"
    response_no_such_keyword: str = "That keyword isn't recognised"

    redis_host: str = "redis"
    redis_port: int = 6379

    object_store: str = "minio"

    dev_mode: bool = False
    superuser_email: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="allow", frozen=True)

    @lru_cache(1)
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
