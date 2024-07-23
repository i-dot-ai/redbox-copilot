import logging
import uuid
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass, field
from datetime import date
from http import HTTPStatus
from itertools import groupby
from operator import attrgetter
from pathlib import Path

from dataclasses_json import Undefined, dataclass_json
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import FieldError, ValidationError
from django.core.files.uploadedfile import UploadedFile
from django.db.models import Max, Min, Prefetch
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from django.views.generic import UpdateView
from requests.exceptions import RequestException
from yarl import URL

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.forms import DemographicsForm
from redbox_app.redbox_core.models import (
    ChatHistory,
    ChatMessage,
    ChatMessageRating,
    ChatMessageRatingChip,
    ChatRoleEnum,
    Citation,
    File,
    StatusEnum,
    User,
)

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


@require_http_methods(["GET"])
def homepage_view(request):
    return render(
        request,
        template_name="homepage.html",
        context={"request": request},
    )


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


class ChatsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, chat_id: uuid.UUID | None = None) -> HttpResponse:
        chat_history = (
            ChatHistory.objects.filter(users=request.user)
            .exclude(id=chat_id)
            .annotate(latest_message_date=Max("chatmessage__created_at"))
            .order_by("-latest_message_date")
        )

        messages: Sequence[ChatMessage] = []
        current_chat = None
        if chat_id:
            current_chat = get_object_or_404(ChatHistory, id=chat_id)
            if current_chat.users != request.user:
                return redirect(reverse("chats"))
            messages = (
                ChatMessage.objects.filter(chat_history__id=chat_id)
                .order_by("created_at")
                .prefetch_related(
                    Prefetch(
                        "source_files",
                        queryset=File.objects.all()
                        .annotate(min_created_at=Min("citation__created_at"))
                        .order_by("min_created_at"),
                    )
                )
            )
        endpoint = URL.build(scheme=settings.WEBSOCKET_SCHEME, host=request.get_host(), path=r"/ws/chat/")

        all_files = File.objects.filter(user=request.user, status=StatusEnum.complete).order_by("-created_at")
        self.decorate_selected_files(all_files, messages)
        ChatsView.decorate_history_with_date_group(chat_history)
        chat_history_grouped_by_date_group = groupby(chat_history, attrgetter("date_group"))

        context = {
            "chat_id": chat_id,
            "messages": messages,
            "chat_history_grouped_by_date_group": chat_history_grouped_by_date_group,
            "current_chat": current_chat,
            "streaming": {"endpoint": str(endpoint)},
            "contact_email": settings.CONTACT_EMAIL,
            "files": all_files,
            "chat_title_length": settings.CHAT_TITLE_LENGTH,
        }

        return render(
            request,
            template_name="chats.html",
            context=context,
        )

    @staticmethod
    def decorate_selected_files(all_files: Sequence[File], messages: Sequence[ChatMessage]) -> None:
        if messages:
            last_user_message = [m for m in messages if m.role == ChatRoleEnum.user][-1]
            selected_files: Sequence[File] = last_user_message.selected_files.all() or []
        else:
            selected_files = []

        for file in all_files:
            file.selected = file in selected_files

    @staticmethod
    def decorate_history_with_date_group(chats: Sequence[ChatHistory]) -> None:
        for chat in chats:
            newest_message_date = chat.chatmessage_set.aggregate(newest_date=Max("created_at"))["newest_date"]
            chat.date_group = ChatsView.get_date_group(newest_message_date.date())

    @staticmethod
    def get_date_group(on: date) -> str:
        today = timezone.now().date()
        age = (today - on).days
        if age > 30:  # noqa: PLR2004
            return "Older than 30 days"
        if age > 7:  # noqa: PLR2004
            return "Previous 30 days"
        if age > 1:
            return "Previous 7 days"
        if age > 0:
            return "Yesterday"
        return "Today"


class ChatsTitleView(View):
    @dataclass_json(undefined=Undefined.EXCLUDE)
    @dataclass(frozen=True)
    class Title:
        name: str

    @method_decorator(login_required)
    def post(self, request: HttpRequest, chat_id: uuid.UUID) -> HttpResponse:
        chat_history: ChatHistory = get_object_or_404(ChatHistory, id=chat_id)
        user_rating = ChatsTitleView.Title.schema().loads(request.body)

        chat_history.name = user_rating.name
        chat_history.save(update_fields=["name"])

        return HttpResponse(status=HTTPStatus.NO_CONTENT)


class CitationsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest, message_id: uuid.UUID | None = None) -> HttpResponse:
        message = get_object_or_404(ChatMessage, id=message_id)

        if message.chat_history.users != request.user:
            return redirect(reverse("chats"))

        source_files = (
            File.objects.filter(citation__chat_message_id=message_id)
            .annotate(min_created_at=Min("citation__created_at"))
            .order_by("min_created_at")
            .prefetch_related(Prefetch("citation_set", queryset=Citation.objects.filter(chat_message_id=message_id)))
        )

        context = {"message": message, "source_files": source_files}

        return render(
            request,
            template_name="citations.html",
            context=context,
        )


@require_http_methods(["POST"])
def post_message(request: HttpRequest) -> HttpResponse:
    message_text = request.POST.get("message", "New chat")
    selected_file_uuids: Sequence[uuid.UUID] = [uuid.UUID(v) for k, v in request.POST.items() if k.startswith("file-")]

    # get current session, or create a new one
    if session_id := request.POST.get("session-id", None):
        session = ChatHistory.objects.get(id=session_id)
    else:
        session_name = message_text[0 : settings.CHAT_TITLE_LENGTH]
        session = ChatHistory(name=session_name, users=request.user)
        session.save()

    selected_files = File.objects.filter(id__in=selected_file_uuids, user=request.user)

    # save user message
    user_message = ChatMessage(chat_history=session, text=message_text, role=ChatRoleEnum.user)
    user_message.save()
    user_message.selected_files.set(selected_files)

    # get LLM response
    message_history = [
        {"role": message.role, "text": message.text}
        for message in ChatMessage.objects.all().filter(chat_history=session)
    ]
    selected_files_message = [{"uuid": str(f.core_file_uuid)} for f in selected_files]
    response_data = core_api.rag_chat(message_history, selected_files_message, request.user)

    llm_message = ChatMessage(chat_history=session, text=response_data.output_text, role=ChatRoleEnum.ai)
    llm_message.save()

    doc_uuids: list[uuid.UUID] = [doc.file_uuid for doc in response_data.source_documents]
    files: list[File] = File.objects.filter(core_file_uuid__in=doc_uuids, user=request.user)

    for file in files:
        file.last_referenced = timezone.now()
        file.save()

    for doc in response_data.source_documents:
        new_citation = Citation(
            file=File.objects.get(core_file_uuid=doc.file_uuid), chat_message=llm_message, text=doc.page_content
        )
        new_citation.save()

    return redirect(reverse("chats", args=(session.id,)))


class RatingsView(View):
    @dataclass_json(undefined=Undefined.EXCLUDE)
    @dataclass(frozen=True)
    class Rating:
        rating: int
        text: str | None = None
        chips: set[str] = field(default_factory=set)

    @method_decorator(login_required)
    def post(self, request: HttpRequest, message_id: uuid.UUID) -> HttpResponse:
        message: ChatMessage = get_object_or_404(ChatMessage, id=message_id)
        user_rating = RatingsView.Rating.schema().loads(request.body)

        chat_message_rating: ChatMessageRating
        if chat_message_rating := ChatMessageRating.objects.filter(chat_message=message).first():
            existing_chips = {c.text for c in chat_message_rating.chatmessageratingchip_set.all()}
            chat_message_rating.rating = user_rating.rating
            chat_message_rating.text = user_rating.text
            for new_chip in user_rating.chips - existing_chips:
                ChatMessageRatingChip(rating_id=chat_message_rating.pk, text=new_chip).save()
            for removed_chip in existing_chips - user_rating.chips:
                ChatMessageRatingChip.objects.get(rating_id=chat_message_rating.pk, text=removed_chip).delete()
            chat_message_rating.save()
        else:
            chat_message_rating = ChatMessageRating(
                chat_message=message, rating=user_rating.rating, text=user_rating.text
            )
            for chip in user_rating.chips:
                chat_message_rating.chatmessageratingchip_set.create(rating=chat_message_rating, text=chip)
            chat_message_rating.save()
        return HttpResponse(status=HTTPStatus.NO_CONTENT)


@require_http_methods(["GET"])
@login_required
def file_status_api_view(request: HttpRequest) -> JsonResponse:
    file_id = request.GET.get("id", None)
    if not file_id:
        logger.error("Error getting file object information - no file ID provided %s.")
        return JsonResponse({"status": StatusEnum.unknown.label})
    try:
        file = get_object_or_404(File, id=file_id)
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
    file.status = core_file_status_response.processing_status
    file.save()
    return JsonResponse({"status": file.get_status_text()})


@require_http_methods(["GET"])
def health(_request: HttpRequest) -> HttpResponse:
    """this required by ECS Fargate"""
    return HttpResponse(status=200)


class CheckDemographicsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        user: User = request.user
        if all([user.grade, user.business_unit, user.profession]):
            return redirect(documents_view)
        else:
            return redirect("demographics")


class DemographicsView(UpdateView):
    model = User
    template_name = "demographics.html"
    form_class = DemographicsForm
    success_url = "/documents/"

    def get_object(self, **kwargs):  # noqa: ARG002
        return self.request.user
