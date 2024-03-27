import os
import uuid

import requests
from boto3.s3.transfer import TransferConfig
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from redbox_app.redbox_core.client import s3_client
from redbox_app.redbox_core.models import File, ProcessingStatusEnum

s3 = s3_client()
CHUNK_SIZE = 1024
# move this somewhere
APPROVED_FILE_EXTENSIONS = [
    ".eml",
    ".html",
    ".json",
    ".md",
    ".msg",
    ".rst",
    ".rtf",
    ".txt",
    ".xml",
    ".csv",
    ".doc",
    ".docx",
    ".epub",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".tsv",
    ".xlsx",
    ".htm",
]


@require_http_methods(["GET"])
def homepage_view(request):
    return render(
        request,
        template_name="homepage.html",
        context={"request": request},
    )


def documents_view(request):
    # Testing with dummy data for now
    if not File.objects.exists():
        File.objects.create(
            name="Document 1",
            path="#download1",
            processing_status=ProcessingStatusEnum.complete,
        )
        File.objects.create(
            name="Document 2",
            path="#download2",
            processing_status=ProcessingStatusEnum.parsing,
        )

    # Add processing_text
    files = File.objects.all()
    for file in files:
        file.processing_text = file.get_processing_text()

    return render(
        request,
        template_name="documents.html",
        context={"request": request, "files": files},
    )


def get_file_extension(file):
    # TODO: validate user file input using a less flaky approach than:
    # return mimetypes.guess_extension(str(magic.from_buffer(file.read(), mime=True)))

    _, extension = os.path.splitext(file.name)
    return extension


def upload_view(request):
    if request.method == "POST" and request.FILES["uploadDoc"]:
        # https://django-storages.readthedocs.io/en/1.13.2/backends/amazon-S3.html
        uploaded_file = request.FILES["uploadDoc"]

        file_extension = get_file_extension(uploaded_file)

        # TODO: Surface errors to the users
        if uploaded_file.name is None:
            raise ValueError("file name is null")
        if uploaded_file.content_type is None:
            raise ValueError("file type is null")
        if file_extension not in APPROVED_FILE_EXTENSIONS:
            raise ValueError(f"file type {file_extension} not approved")

        file_key = f"{uuid.uuid4()}{file_extension}"

        # TODO: can we upload chunks instead of having the file read?
        s3.upload_fileobj(
            Bucket=settings.BUCKET_NAME,
            Fileobj=uploaded_file,
            Key=file_key,
            ExtraArgs={"Tagging": f"file_type={uploaded_file.content_type}"},
            Config=TransferConfig(
                multipart_chunksize=CHUNK_SIZE,
                preferred_transfer_client="auto",
                multipart_threshold=CHUNK_SIZE,
                use_threads=True,
                max_concurrency=80,
            ),
        )

        # TODO: Handle S3 upload errors
        authenticated_s3_url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": settings.BUCKET_NAME,
                "Key": file_key,
            },
            ExpiresIn=3600,
        )
        # Strip off the query string (we don't need the keys)
        simple_s3_url = authenticated_s3_url.split("?")[0]

        # ingest file
        # TODO: set url using ENV vars
        url = "http://core-api:5002/file"
        api_response = requests.post(
            url,
            params={
                "name": uploaded_file.name,
                "type": uploaded_file.content_type,
                "location": simple_s3_url,
            },
        )

        # TODO: handle better
        return JsonResponse(api_response.json())

    return render(
        request,
        template_name="upload.html",
        context={"request": request},
    )


def remove_doc_view(request, doc_id: str):
    if request.method == "POST":
        print(f"Removing document: {request.POST['doc_id']}")
        # TO DO: handle document deletion here

    # Hard-coding document name for now, just to flag that this is needed in the template
    doc_name = "Document X"
    return render(
        request,
        template_name="remove-doc.html",
        context={"request": request, "doc_id": doc_id, "doc_name": doc_name},
    )
