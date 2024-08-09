import json
import logging
import uuid
from collections.abc import Sequence
from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from django.conf import settings
from django.test import Client
from django.urls import reverse
from requests_mock import Mocker
from yarl import URL

from redbox_app.redbox_core.models import (
    Chat,
    ChatMessage,
    ChatRoleEnum,
    Citation,
    File,
    User,
)

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_post_message_to_new_session(alice: User, client: Client, requests_mock: Mocker):
    # Given
    client.force_login(alice)
    rag_url = f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/chat/rag"
    requests_mock.register_uri(
        "POST",
        rag_url,
        json={"output_text": "Good afternoon, Mr. Amor.", "source_documents": []},
    )

    # When
    response = client.post("/post-message/", {"message": "Are you there?"})

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert "Location" in response.headers
    session_id = URL(response.url).parts[-2]
    assert ChatMessage.objects.get(chat__id=session_id, role=ChatRoleEnum.user).text == "Are you there?"
    assert ChatMessage.objects.get(chat__id=session_id, role=ChatRoleEnum.ai).text == "Good afternoon, Mr. Amor."


@pytest.mark.django_db()
def test_post_message_to_existing_session(chat: Chat, client: Client, requests_mock: Mocker, uploaded_file: File):
    # Given
    client.force_login(chat.user)
    session_id = chat.id
    rag_url = f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/chat/rag"
    requests_mock.register_uri(
        "POST",
        rag_url,
        json={
            "output_text": "Good afternoon, Mr. Amor.",
            "source_documents": [
                {"file_uuid": str(uploaded_file.core_file_uuid), "page_content": "Here is a source chunk"}
            ],
        },
    )
    initial_file_expiry_date = File.objects.get(core_file_uuid=uploaded_file.core_file_uuid).expires_at

    # When
    response = client.post("/post-message/", {"message": "Are you there?", "session-id": session_id})

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert URL(response.url).parts[-2] == str(session_id)
    assert ChatMessage.objects.get(chat__id=session_id, role=ChatRoleEnum.ai).text == "Good afternoon, Mr. Amor."
    assert ChatMessage.objects.get(chat__id=session_id, role=ChatRoleEnum.ai).source_files.first() == uploaded_file
    assert initial_file_expiry_date != File.objects.get(core_file_uuid=uploaded_file.core_file_uuid).expires_at
    assert (
        Citation.objects.get(chat_message=ChatMessage.objects.get(chat__id=session_id, role=ChatRoleEnum.ai)).text
        == "Here is a source chunk"
    )


@pytest.mark.django_db()
def test_post_message_with_files_selected(
    chat: Chat, client: Client, requests_mock: Mocker, several_files: Sequence[File]
):
    # Given
    client.force_login(chat.user)
    session_id = chat.id
    selected_files = several_files[::2]

    rag_url = f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/chat/rag"
    requests_mock.register_uri(
        "POST",
        rag_url,
        json={
            "output_text": "Only those, then.",
            "source_documents": [
                {"file_uuid": str(f.core_file_uuid), "page_content": "Here is a source chunk"} for f in selected_files
            ],
        },
    )

    # When
    response = client.post(
        "/post-message/",
        {
            "message": "Only tell me about these, please.",
            "session-id": session_id,
            **{f"file-{f.id}": f.id for f in selected_files},
        },
    )

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert (
        list(ChatMessage.objects.get(chat__id=session_id, role=ChatRoleEnum.user).selected_files.all())
        == selected_files
    )
    assert json.loads(requests_mock.last_request.text).get("selected_files") == [
        {"uuid": str(f.core_file_uuid)} for f in selected_files
    ]


@pytest.mark.django_db()
def test_user_can_see_their_own_chats(chat: Chat, alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get(f"/chats/{chat.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db()
def test_user_cannot_see_other_users_chats(chat: Chat, bob: User, client: Client):
    # Given
    client.force_login(bob)

    # When
    response = client.get(f"/chats/{chat.id}/")

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers.get("Location") == "/chats/"


@pytest.mark.django_db()
def test_view_session_with_documents(chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(chat_message.chat.user)
    chat_id = chat_message.chat.id

    # When
    response = client.get(f"/chats/{chat_id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert b"original_file.txt" in response.content


@pytest.mark.django_db()
def test_chat_grouped_by_age(user_with_chats_with_messages_over_time: User, client: Client):
    # Given
    client.force_login(user_with_chats_with_messages_over_time)

    # When
    response = client.get(reverse("chats"))

    # Then
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content)
    date_groups = soup.find_all("h3", {"class": "rb-chat-history__date_group"})
    assert len(date_groups) == 5
    for date_group, (header, chat_name) in zip(
        date_groups,
        [
            ("Today", "today"),
            ("Yesterday", "yesterday"),
            ("Previous 7 days", "5 days old"),
            ("Previous 30 days", "20 days old"),
            ("Older than 30 days", "40 days old"),
        ],
        strict=False,
    ):
        assert date_group.text == header
        assert date_group.find_next_sibling("ul").find("a").text == chat_name


@pytest.mark.django_db()
def test_nonexistent_chats(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("chats", kwargs={"chat_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.django_db()
def test_post_chat_title(alice: User, chat: Chat, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("chat-titles", kwargs={"chat_id": chat.id})
    response = client.post(url, json.dumps({"name": "New chat name"}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    chat.refresh_from_db()
    assert chat.name == "New chat name"


@pytest.mark.django_db()
def test_post_chat_title_with_naughty_string(alice: User, chat: Chat, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("chat-titles", kwargs={"chat_id": chat.id})
    response = client.post(url, json.dumps({"name": "New chat name \x00"}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    chat.refresh_from_db()
    assert chat.name == "New chat name \ufffd"


@pytest.mark.django_db()
def test_staff_user_can_see_route(chat_with_files: Chat, client: Client):
    # Given
    chat_with_files.user.is_staff = True
    chat_with_files.user.save()
    client.force_login(chat_with_files.user)

    # When
    response = client.get(f"/chats/{chat_with_files.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert b"iai-chat-bubble__route" in response.content
    assert b"iai-chat-bubble__route govuk-!-display-none" not in response.content
