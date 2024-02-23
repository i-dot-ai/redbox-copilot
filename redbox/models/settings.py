import os
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")


class Settings(BaseSettings):
    elastic_host: str = "elasticsearch"
    elastic_port: int
    elastic_scheme: Literal["http"]
    elastic_user: str = "ELASTIC_PASSWORD"
    elastic_version: str
    elastic_password: str = "redboxpass"

    kibana_system_password: str = "redboxpass"
    metricbeat_internal_password: str = "redboxpass"
    filebeat_internal_password: str = "redboxpass"
    heartbeat_internal_password: str = "redboxpass"
    monitoring_internal_password: str = "redboxpass"
    beats_system_password: str = "redboxpass"

    minio_host: str = "minio"
    minio_port: int = 9000
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "eu-west-2"

    object_store: Literal["s3", "minio"] = "minio"

    bucket_name: str = "redbox-storage-dev"
    embedding_model: str = "all-mpnet-base-v2"

    embed_queue_name: str = "redbox-embed-queue"
    ingest_queue_name: str = "redbox-ingest-queue"

    queue: Literal["sqs", "rabbitmq"] = "rabbitmq"

    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"

    dev_mode: bool = False

    model_config = SettingsConfigDict(env_file=env_path)
