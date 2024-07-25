from datetime import UTC, datetime, timedelta

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from redbox_app.redbox_core.models import File, StatusEnum


@pytest.mark.django_db()
def test_file_model_last_referenced(peter_rabbit, s3_client):  # noqa: ARG001
    mock_file = SimpleUploadedFile("test.txt", b"these are the file contents")

    new_file = File.objects.create(
        status=StatusEnum.uploaded,
        original_file=mock_file,
        user=peter_rabbit,
        original_file_name="test.txt",
    )

    # Tests the initial value of the last_referenced
    expected_last_referenced = new_file.created_at
    # The last_referenced should be FILE_EXPIRY_IN_SECONDS ahead of created_at
    # these fields are set during the same save() process
    # this test accounts for a delay between creating the fields
    assert abs(new_file.last_referenced - expected_last_referenced) < timedelta(seconds=1)

    # Tests that the last_referenced field can be updated
    new_date = datetime(2028, 1, 1, tzinfo=UTC)
    new_file.last_referenced = new_date
    new_file.save()
    assert new_file.last_referenced == new_date
