import logging
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus

from dataclasses_json import Undefined, dataclass_json
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views import View

from redbox_app.redbox_core.models import (
    ChatMessage,
    ChatMessageRating,
    ChatMessageRatingChip,
)

logger = logging.getLogger(__name__)


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
