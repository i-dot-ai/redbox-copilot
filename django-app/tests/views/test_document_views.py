import logging
import uuid
from http import HTTPStatus
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from django.conf import settings
from django.test import Client
from django.urls import reverse
from requests_mock import Mocker

from redbox_app.redbox_core.models import File, StatusEnum, User

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_upload_view(alice, client, file_pdf_path: Path, s3_client, requests_mock):
    """
    Given that the object store does not have a file with our test file in it
    When we POST our test file to /upload/
    We Expect to see this file in the object store
    """
    file_name = file_pdf_path.name

    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    assert not file_exists(s3_client, file_name)

    client.force_login(alice)

    # we mock the response from the core-api
    mocked_response = {
        "key": file_name,
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert file_exists(s3_client, file_name)
        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"


@pytest.mark.django_db()
def test_document_upload_status(client, alice, file_pdf_path: Path, s3_client, requests_mock):
    file_name = file_pdf_path.name

    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    assert not file_exists(s3_client, file_name)
    client.force_login(alice)
    previous_count = count_s3_objects(s3_client)

    mocked_response = {
        "key": file_name,
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"
        assert count_s3_objects(s3_client) == previous_count + 1
        uploaded_file = File.objects.filter(user=alice).order_by("-created_at")[0]
        assert uploaded_file.status == StatusEnum.uploaded


@pytest.mark.django_db()
def test_upload_view_duplicate_files(alice, bob, client, file_pdf_path: Path, s3_client, requests_mock):
    # we mock the response from the core-api
    mocked_response = {
        "key": "file_key",
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    previous_count = count_s3_objects(s3_client)
    client.force_login(alice)

    with file_pdf_path.open("rb") as f:
        client.post("/upload/", {"uploadDocs": f})
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"

        assert count_s3_objects(s3_client) == previous_count + 2

        client.force_login(bob)
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"

        assert count_s3_objects(s3_client) == previous_count + 3

        assert (
            File.objects.order_by("-created_at")[0].unique_name != File.objects.order_by("-created_at")[1].unique_name
        )


@pytest.mark.django_db()
def test_upload_view_bad_data(alice, client, file_py_path: Path, s3_client):
    previous_count = count_s3_objects(s3_client)
    client.force_login(alice)

    with file_py_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.OK
        assert "File type .py not supported" in str(response.content)
        assert count_s3_objects(s3_client) == previous_count


@pytest.mark.django_db()
def test_upload_view_no_file(alice, client):
    client.force_login(alice)

    response = client.post("/upload/")

    assert response.status_code == HTTPStatus.OK
    assert "No document selected" in str(response.content)


@pytest.mark.django_db()
def test_remove_doc_view(client: Client, alice: User, file_pdf_path: Path, s3_client: Client, requests_mock: Mocker):
    file_name = file_pdf_path.name

    client.force_login(alice)
    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    previous_count = count_s3_objects(s3_client)

    mocked_response = {
        "key": file_name,
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    with file_pdf_path.open("rb") as f:
        # create file before testing deletion
        client.post("/upload/", {"uploadDocs": f})
        assert file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count + 1

        new_file = File.objects.filter(user=alice).order_by("-created_at")[0]
        requests_mock.delete(
            f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file/{new_file.core_file_uuid}",
            status_code=201,
            json=mocked_response,
        )

        client.post(f"/remove-doc/{new_file.id}", {"doc_id": new_file.id})
        assert not file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count
        assert requests_mock.request_history[-1].method == "DELETE"
        assert File.objects.get(id=new_file.id).status == StatusEnum.deleted


@pytest.mark.django_db()
def test_remove_nonexistent_doc(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("remove-doc", kwargs={"doc_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


def count_s3_objects(s3_client) -> int:
    paginator = s3_client.get_paginator("list_objects")
    return sum(len(result.get("Contents", [])) for result in paginator.paginate(Bucket=settings.BUCKET_NAME) if result)


def file_exists(s3_client, file_name) -> bool:
    """
    if the file key exists return True otherwise False
    """
    try:
        s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))
    except ClientError as client_error:
        if client_error.response["Error"]["Code"] == "NoSuchKey":
            return False
        raise
    else:
        return True
