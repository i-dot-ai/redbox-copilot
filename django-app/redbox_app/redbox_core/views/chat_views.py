import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from http import HTTPStatus
from itertools import groupby
from operator import attrgetter

from dataclasses_json import Undefined, dataclass_json
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Max, Min, Prefetch
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods
from yarl import URL

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.models import (
    ChatHistory,
    ChatMessage,
    ChatRoleEnum,
    Citation,
    File,
    StatusEnum,
)

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)


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
