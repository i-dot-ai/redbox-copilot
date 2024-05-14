from datetime import date, timedelta

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from redbox_app.redbox_core.models import File, ProcessingStatusEnum


@pytest.mark.django_db
def test_file_model_expiry_date(peter_rabbit):
    mock_file = SimpleUploadedFile("test.txt", b"these are the file contents")

    new_file = File.objects.create(
        processing_status=ProcessingStatusEnum.uploaded,
        original_file=mock_file,
        user=peter_rabbit,
        original_file_name="test.txt",
    )

    # TODO: update by using freezegun
    assert new_file.expiry_date == date.today() + timedelta(days=30)
