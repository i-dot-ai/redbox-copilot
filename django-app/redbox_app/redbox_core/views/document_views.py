import logging
import uuid
from collections.abc import MutableSequence, Sequence
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import FieldError, ValidationError
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from requests.exceptions import RequestException

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.models import File, StatusEnum, User

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)
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
    ".epub",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".tsv",
    ".xlsx",
    ".htm",
]
MAX_FILE_SIZE = 209715200  # 200 MB or 200 * 1024 * 1024


@login_required
def documents_view(request):
    completed_files = File.objects.filter(user=request.user, status=StatusEnum.complete).order_by("-created_at")
    hidden_statuses = [StatusEnum.deleted, StatusEnum.errored, StatusEnum.complete]
    processing_files = (
        File.objects.filter(user=request.user).exclude(status__in=hidden_statuses).order_by("-created_at")
    )

    ingest_errors = request.session.get("ingest_errors", [])
    request.session["ingest_errors"] = []

    return render(
        request,
        template_name="documents.html",
        context={
            "request": request,
            "completed_files": completed_files,
            "processing_files": processing_files,
            "ingest_errors": ingest_errors,
        },
    )


class UploadView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        return self.build_response(request)

    @method_decorator(login_required)
    def post(self, request: HttpRequest) -> HttpResponse:
        errors: MutableSequence[str] = []
        ingest_errors: MutableSequence[str] = []

        uploaded_files: MutableSequence[UploadedFile] = request.FILES.getlist("uploadDocs")

        if not uploaded_files:
            errors.append("No document selected")

        for uploaded_file in uploaded_files:
            errors += self.validate_uploaded_file(uploaded_file)

        if not errors:
            for uploaded_file in uploaded_files:
                # ingest errors are handled differently, as the other documents have started uploading by this point
                ingest_error = self.ingest_file(uploaded_file, request.user)
                if ingest_error:
                    ingest_errors.append(f"{uploaded_file.name}: {ingest_error[0]}")

            request.session["ingest_errors"] = ingest_errors
            return redirect(reverse(documents_view))

        return self.build_response(request, errors)

    @staticmethod
    def build_response(request: HttpRequest, errors: Sequence[str] | None = None) -> HttpResponse:
        return render(
            request,
            template_name="upload.html",
            context={
                "request": request,
                "errors": {"upload_doc": errors or []},
                "uploaded": not errors,
            },
        )

    @staticmethod
    def validate_uploaded_file(uploaded_file: UploadedFile) -> Sequence[str]:
        errors: MutableSequence[str] = []

        if not uploaded_file.name:
            errors.append("File has no name")
        else:
            file_extension = Path(uploaded_file.name).suffix
            if file_extension not in APPROVED_FILE_EXTENSIONS:
                errors.append(f"Error with {uploaded_file.name}: File type {file_extension} not supported")

        if not uploaded_file.content_type:
            errors.append(f"Error with {uploaded_file.name}: File has no content-type")

        if uploaded_file.size > MAX_FILE_SIZE:
            errors.append(f"Error with {uploaded_file.name}: File is larger than 200MB")

        return errors

    @staticmethod
    def ingest_file(uploaded_file: UploadedFile, user: User) -> Sequence[str]:
        errors: MutableSequence[str] = []
        try:
            file = File.objects.create(
                status=StatusEnum.uploaded.value,
                user=user,
                original_file=uploaded_file,
                original_file_name=uploaded_file.name,
            )
            file.save()
        except (ValueError, FieldError, ValidationError) as e:
            logger.exception("Error creating File model object for %s.", uploaded_file, exc_info=e)
            errors.append(e.args[0])
        else:
            try:
                upload_file_response = core_api.upload_file(file.unique_name, user)
            except RequestException as e:
                logger.exception("Error uploading file object %s.", file, exc_info=e)
                file.delete()
                errors.append("failed to connect to core-api")
            else:
                file.core_file_uuid = upload_file_response.uuid
                file.save()
        return errors


@login_required
def remove_doc_view(request, doc_id: uuid):
    file = get_object_or_404(File, id=doc_id)
    errors: list[str] = []

    if request.method == "POST":
        try:
            core_api.delete_file(file.core_file_uuid, request.user)
        except RequestException as e:
            logger.exception("Error deleting file object %s.", file, exc_info=e)
            errors.append("There was an error deleting this file")

        else:
            logger.info("Removing document: %s", request.POST["doc_id"])
            file.delete_from_s3()
            file.status = StatusEnum.deleted
            file.save()
            return redirect("documents")

    return render(
        request,
        template_name="remove-doc.html",
        context={"request": request, "doc_id": doc_id, "doc_name": file.name, "errors": errors},
    )


@require_http_methods(["GET"])
@login_required
def file_status_api_view(request: HttpRequest) -> JsonResponse:
    file_id = request.GET.get("id", None)
    if not file_id:
        logger.error("Error getting file object information - no file ID provided %s.")
        return JsonResponse({"status": StatusEnum.unknown.label})
    try:
        file: File = get_object_or_404(File, id=file_id)
    except File.DoesNotExist as ex:
        logger.exception("File object information not found in django - file does not exist %s.", file_id, exc_info=ex)
        return JsonResponse({"status": StatusEnum.unknown.label})
    try:
        core_file_status_response = core_api.get_file_status(file_id=file.core_file_uuid, user=request.user)
    except RequestException as ex:
        logger.exception("File object information from core not found - file does not exist %s.", file_id, exc_info=ex)
        if not file.status:
            file.status = StatusEnum.unknown.label
            file.save()
        return JsonResponse({"status": file.status})
    file.status = core_file_status_response.processing_status or StatusEnum.unknown.name
    file.save()
    return JsonResponse({"status": file.get_status_text()})
