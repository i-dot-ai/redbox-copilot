import pytest
from django.conf import settings


@pytest.mark.django_db
def test_declaration_view_get(peter_rabbit, client):
    client.force_login(peter_rabbit)
    response = client.get("/")
    assert response.status_code == 200, response.status_code


def count_s3_objects(s3_client):
    return len(s3_client.list_objects_v2(Bucket=settings.BUCKET_NAME)["Contents"])


@pytest.mark.django_db
def test_upload_view(client, file_pdf_path, s3_client):
    previous_count = count_s3_objects(s3_client)

    with open(file_pdf_path, "rb") as f:
        response = client.post("/upload/", {"uploadDoc": f})

        assert response.status_code == 200
        assert "Your file has been uploaded" in str(response.content)

        assert count_s3_objects(s3_client) == previous_count + 1


@pytest.mark.django_db
def test_upload_view_bad_data(client, file_py_path, s3_client):
    previous_count = count_s3_objects(s3_client)

    with open(file_py_path, "rb") as f:
        response = client.post("/upload/", {"uploadDoc": f})

        assert response.status_code == 200
        assert "File type .py not supported" in str(response.content)
        assert count_s3_objects(s3_client) == previous_count
