import json
import logging
from http import HTTPStatus

import pytest
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import ChatMessage, ChatMessageRating, User

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_post_new_rating_only(alice: User, chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("ratings", kwargs={"message_id": chat_message.id})
    response = client.post(url, json.dumps({"rating": 5}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    rating = ChatMessageRating.objects.get(pk=chat_message.pk)
    assert rating.rating == 5
    assert rating.text is None
    assert {c.text for c in rating.chatmessageratingchip_set.all()} == set()


@pytest.mark.django_db()
def test_post_new_rating(alice: User, chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("ratings", kwargs={"message_id": chat_message.id})
    response = client.post(
        url,
        json.dumps({"rating": 5, "text": "Lorem Ipsum.", "chips": ["speed", "accuracy", "swearing"]}),
        content_type="application/json",
    )

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    rating = ChatMessageRating.objects.get(pk=chat_message.pk)
    assert rating.rating == 5
    assert rating.text == "Lorem Ipsum."
    assert {c.text for c in rating.chatmessageratingchip_set.all()} == {"speed", "accuracy", "swearing"}


@pytest.mark.django_db()
def test_post_new_rating_with_naughty_string(alice: User, chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("ratings", kwargs={"message_id": chat_message.id})
    response = client.post(
        url,
        json.dumps({"rating": 5, "text": "Lorem Ipsum. \x00", "chips": ["speed", "accuracy", "swearing"]}),
        content_type="application/json",
    )

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    rating = ChatMessageRating.objects.get(pk=chat_message.pk)
    assert rating.rating == 5
    assert rating.text == "Lorem Ipsum. \ufffd"
    assert {c.text for c in rating.chatmessageratingchip_set.all()} == {"speed", "accuracy", "swearing"}


@pytest.mark.django_db()
def test_post_updated_rating(alice: User, chat_message_with_rating: ChatMessage, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("ratings", kwargs={"message_id": chat_message_with_rating.id})
    response = client.post(
        url,
        json.dumps({"rating": 5, "text": "Lorem Ipsum.", "chips": ["speed", "accuracy", "swearing"]}),
        content_type="application/json",
    )

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    rating = ChatMessageRating.objects.get(pk=chat_message_with_rating.pk)
    assert rating.rating == 5
    assert rating.text == "Lorem Ipsum."
    assert {c.text for c in rating.chatmessageratingchip_set.all()} == {"speed", "accuracy", "swearing"}


@pytest.mark.django_db()
def test_post_updated_rating_with_naughty_string(alice: User, chat_message_with_rating: ChatMessage, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("ratings", kwargs={"message_id": chat_message_with_rating.id})
    response = client.post(
        url,
        json.dumps({"rating": 5, "text": "Lorem Ipsum. \x00", "chips": ["speed", "accuracy", "swearing"]}),
        content_type="application/json",
    )

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    rating = ChatMessageRating.objects.get(pk=chat_message_with_rating.pk)
    assert rating.rating == 5
    assert rating.text == "Lorem Ipsum. \ufffd"
    assert {c.text for c in rating.chatmessageratingchip_set.all()} == {"speed", "accuracy", "swearing"}
