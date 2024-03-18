import requests
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

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
        doc = request.FILES["uploadDoc"]
        files = {"file": (doc.name, doc.read(), doc.content_type)}
        url = "http://core-api:5002/file"
        api_response = requests.post(url, files=files)

        if api_response.status_code == 422:
            print(api_response.json())
        else:
            print(api_response)

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
