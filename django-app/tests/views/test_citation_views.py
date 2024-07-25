import logging
import uuid
from collections.abc import Sequence
from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from django.test import Client
from django.urls import reverse

from redbox_app.redbox_core.models import (
    ChatHistory,
    ChatMessage,
    ChatRoleEnum,
    Citation,
    File,
    User,
)

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_citations_shown_in_correct_order(
    client: Client, alice: User, chat_history: ChatHistory, several_files: Sequence[File]
):
    # Given
    client.force_login(alice)
    chat_message = ChatMessage.objects.create(chat_history=chat_history, text="Some answer.", role=ChatRoleEnum.ai)

    Citation.objects.create(file=several_files[1], chat_message=chat_message, text="Citation 1")
    Citation.objects.create(file=several_files[0], chat_message=chat_message, text="Citation 2")
    Citation.objects.create(file=several_files[1], chat_message=chat_message, text="Citation 3")
    Citation.objects.create(file=several_files[2], chat_message=chat_message, text="Citation 4")
    Citation.objects.create(file=several_files[0], chat_message=chat_message, text="Citation 5")

    # When
    response = client.get(f"/citations/{chat_message.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content)
    sources_panel = soup.select("div.iai-panel")[1]
    files = sources_panel.find_all("h3")
    filenames = [h3.get_text().strip() for h3 in files]
    citations = [
        [li.get_text().strip() for li in citations.find_all(class_="rb-citations__item")]
        for citations in [h3.next_sibling.next_sibling for h3 in files]
    ]

    assert filenames == ["original_file_1.txt", "original_file_0.txt", "original_file_2.txt"]
    assert citations == [["Citation 1", "Citation 3"], ["Citation 2", "Citation 5"], ["Citation 4"]]


@pytest.mark.django_db()
def test_user_can_see_their_own_citations(chat_message_with_citation: ChatMessage, alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get(f"/citations/{chat_message_with_citation.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db()
def test_user_cannot_see_other_users_citations(chat_message_with_citation: ChatHistory, bob: User, client: Client):
    # Given
    client.force_login(bob)

    # When
    response = client.get(f"/citations/{chat_message_with_citation.id}/")

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers.get("Location") == "/chats/"


@pytest.mark.django_db()
def test_nonexistent_citations(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("citations", kwargs={"message_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND
