import logging
import uuid

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Min, Prefetch
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.models import ChatMessage, Citation, File

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)


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
