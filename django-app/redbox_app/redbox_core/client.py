import logging
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import boto3
import requests
from botocore.exceptions import ClientError
from dataclasses_json import Undefined, dataclass_json
from django.conf import settings
from yarl import URL

from redbox_app.redbox_core.models import User

logger = logging.getLogger(__name__)


def s3_client():
    if settings.OBJECT_STORE == "s3":
        client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME,
        )
    else:
        client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
            endpoint_url=f"http://{settings.MINIO_HOST}:{settings.MINIO_PORT}",
        )

    try:
        client.create_bucket(
            Bucket=settings.BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": settings.AWS_S3_REGION_NAME},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise
    return client


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class CoreChatResponseDoc:
    file_uuid: str
    page_content: str


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class CoreChatResponse:
    output_text: str
    source_documents: list[CoreChatResponseDoc]


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class FileStatus:
    processing_status: Literal[
        "uploaded", "parsing", "chunking", "embedding", "indexing", "complete", "unknown", "errored"
    ]


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class FileOperation:
    key: str
    bucket: str
    uuid: str


class CoreApiClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @property
    def url(self) -> URL:
        return URL(f"http://{self.host}:{self.port}")

    def upload_file(self, name: str, user: User) -> FileOperation:
        response = requests.post(
            self.url / "file", json={"key": name}, headers={"Authorization": user.get_bearer_token()}, timeout=30
        )
        response.raise_for_status()
        return FileOperation.schema().loads(response.content)

    def rag_chat(
        self, message_history: list[dict[str, str]], selected_files: list[dict[str, str]], user: User
    ) -> CoreChatResponse:
        response = requests.post(
            self.url / "chat/rag",
            json={"message_history": message_history, "selected_files": selected_files},
            headers={"Authorization": user.get_bearer_token()},
            timeout=60,
        )
        response.raise_for_status()
        response_data = CoreChatResponse.schema().loads(response.content)
        logger.debug("response_data: %s", response_data)

        return response_data

    def get_file_status(self, file_id: UUID, user: User) -> FileStatus:
        url = self.url / "file" / str(file_id) / "status"
        response = requests.get(url, headers={"Authorization": user.get_bearer_token()}, timeout=60)
        response.raise_for_status()
        return FileStatus.schema().loads(response.content)

    def delete_file(self, file_id: UUID, user: User) -> FileOperation:
        url = self.url / "file" / str(file_id)
        response = requests.delete(url, headers={"Authorization": user.get_bearer_token()}, timeout=60)
        response.raise_for_status()
        return FileOperation.schema().loads(response.content)

    def reingest_file(self, file_id: UUID, user: User) -> FileOperation:
        url = self.url / "file" / str(file_id)
        response = requests.put(url, headers={"Authorization": user.get_bearer_token()}, timeout=60)
        response.raise_for_status()
        return FileOperation.schema().loads(response.content)
