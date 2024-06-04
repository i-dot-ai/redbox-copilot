import logging
from types import SimpleNamespace
from typing import Annotated
from uuid import UUID

import boto3
import requests
from botocore.exceptions import ClientError
from django.conf import settings
from mypy_boto3_s3 import S3Client
from redbox_app.redbox_core.models import User
from wireup import Inject, service
from yarl import URL

logger = logging.getLogger(__name__)


@service
def s3_client_factory() -> S3Client:
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


@service
class CoreApiClient:
    host: Annotated[str, Inject(param="CORE_API_HOST")]
    port: Annotated[int, Inject(param="CORE_API_PORT")]

    @property
    def url(self) -> URL:
        return URL(f"http://{self.host}:{self.port}")

    def upload_file(self, name: str, user: User) -> SimpleNamespace:
        response = requests.post(
            self.url / "file", json={"key": name}, headers={"Authorization": user.get_bearer_token()}, timeout=30
        )
        response.raise_for_status()
        return response.json(object_hook=lambda d: SimpleNamespace(**d))

    def rag_chat(self, message_history: list[dict[str, str]], user: User) -> SimpleNamespace:
        response = requests.post(
            self.url / "chat/rag",
            json={"message_history": message_history},
            headers={"Authorization": user.get_bearer_token()},
            timeout=60,
        )
        response.raise_for_status()
        response_data = response.json(object_hook=lambda d: SimpleNamespace(**d))
        logger.debug("response_data: %s", response_data)

        return response_data

    def get_file_status(self, file_id: UUID, user: User) -> SimpleNamespace:
        url = self.url / "file" / str(file_id) / "status"
        response = requests.get(url, headers={"Authorization": user.get_bearer_token()}, timeout=60)
        response.raise_for_status()
        return response.json(object_hook=lambda d: SimpleNamespace(**d))

    def delete_file(self, file_id: UUID, user: User) -> SimpleNamespace:
        url = self.url / "file" / str(file_id)
        response = requests.delete(url, headers={"Authorization": user.get_bearer_token()}, timeout=60)
        response.raise_for_status()
        return response.json(object_hook=lambda d: SimpleNamespace(**d))
