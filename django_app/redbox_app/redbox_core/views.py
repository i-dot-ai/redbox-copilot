from boto3.s3.transfer import TransferConfig
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from redbox_app.redbox_core.client import s3_client

s3 = s3_client()
CHUNK_SIZE = 1024

from .models import File, ProcessingStatusEnum


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


def upload_view(request):
    if request.method == "POST" and request.FILES["uploadDoc"]:
        # https://django-storages.readthedocs.io/en/1.13.2/backends/amazon-S3.html
        file = request.FILES["uploadDoc"]

        # Do some error handling here
        # if file.filename is None:
        #     raise ValueError("file name is null")
        # if file.content_type is None:
        #     raise ValueError("file type is null")

        # s3.put_object(
        #     Bucket=settings.BUCKET_NAME,
        #     Body=file.file,
        #     Key=file.name,
        #     Tagging=f"file_type={file.content_type}",
        # )

        # for chunk in file.chunks():
        #     s3.upload_fileobj(
        #         Bucket=settings.BUCKET_NAME,
        #         Fileobj=BytesIO(chunk),
        #         Key=file.name,
        #         ExtraArgs={"Tagging": f"file_type={file.content_type}"},
        #     )

        s3.upload_fileobj(
            Bucket=settings.BUCKET_NAME,
            Fileobj=file.file,
            Key=file.name,
            ExtraArgs={"Tagging": f"file_type={file.content_type}"},
            Config=TransferConfig(
                multipart_chunksize=CHUNK_SIZE,
                preferred_transfer_client="auto",
                multipart_threshold=CHUNK_SIZE,
                use_threads=True,
                max_concurrency=80,
            ),
        )

        #     return JsonResponse(
        #         {
        #             "message": "OK",
        #             "fileUrl": file_url,
        #         }
        #     )
        # else:
        #     return JsonResponse(
        #         {
        #             "message": "Error: file {filename} already exists in bucket {bucket_name}".format(
        #                 filename=file_obj.name,
        #                 bucket_name=file_storage.bucket_name,
        #             ),
        #         },
        #         status=400,
        # )

        # url = "http://core-api:5002/file"
        # api_response = requests.post(url, files=files)

    #     if api_response.status_code == 422:
    #         print(api_response.json())
    #     else:
    #         print(api_response)

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
