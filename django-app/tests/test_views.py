import json
import logging
import uuid
from collections.abc import Sequence
from datetime import date, timedelta
from http import HTTPStatus
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from django.conf import settings
from django.test import Client
from django.urls import reverse
from django.utils import timezone
from pytest_django.asserts import assertRedirects
from requests_mock import Mocker
from yarl import URL

from redbox_app.redbox_core.models import (
    BusinessUnit,
    ChatHistory,
    ChatMessage,
    ChatMessageRating,
    ChatRoleEnum,
    Citation,
    File,
    StatusEnum,
    User,
)
from redbox_app.redbox_core.views import ChatsView

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_declaration_view_get(peter_rabbit, client):
    client.force_login(peter_rabbit)
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK, response.status_code


def count_s3_objects(s3_client) -> int:
    paginator = s3_client.get_paginator("list_objects")
    return sum(len(result.get("Contents", [])) for result in paginator.paginate(Bucket=settings.BUCKET_NAME) if result)


def file_exists(s3_client, file_name) -> bool:
    """
    if the file key exists return True otherwise False
    """
    try:
        s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))
    except ClientError as client_error:
        if client_error.response["Error"]["Code"] == "NoSuchKey":
            return False
        raise
    else:
        return True


@pytest.mark.django_db()
def test_upload_view(alice, client, file_pdf_path: Path, s3_client, requests_mock):
    """
    Given that the object store does not have a file with our test file in it
    When we POST our test file to /upload/
    We Expect to see this file in the object store
    """
    file_name = file_pdf_path.name

    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    assert not file_exists(s3_client, file_name)

    client.force_login(alice)

    # we mock the response from the core-api
    mocked_response = {
        "key": file_name,
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert file_exists(s3_client, file_name)
        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"


@pytest.mark.django_db()
def test_document_upload_status(client, alice, file_pdf_path: Path, s3_client, requests_mock):
    file_name = file_pdf_path.name

    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    assert not file_exists(s3_client, file_name)
    client.force_login(alice)
    previous_count = count_s3_objects(s3_client)

    mocked_response = {
        "key": file_name,
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    with file_pdf_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"
        assert count_s3_objects(s3_client) == previous_count + 1
        uploaded_file = File.objects.filter(user=alice).order_by("-created_at")[0]
        assert uploaded_file.status == StatusEnum.uploaded


@pytest.mark.django_db()
def test_upload_view_duplicate_files(alice, bob, client, file_pdf_path: Path, s3_client, requests_mock):
    # we mock the response from the core-api
    mocked_response = {
        "key": "file_key",
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    previous_count = count_s3_objects(s3_client)
    client.force_login(alice)

    with file_pdf_path.open("rb") as f:
        client.post("/upload/", {"uploadDocs": f})
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"

        assert count_s3_objects(s3_client) == previous_count + 2

        client.force_login(bob)
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.FOUND
        assert response.url == "/documents/"

        assert count_s3_objects(s3_client) == previous_count + 3

        assert (
            File.objects.order_by("-created_at")[0].unique_name != File.objects.order_by("-created_at")[1].unique_name
        )


@pytest.mark.django_db()
def test_upload_view_bad_data(alice, client, file_py_path: Path, s3_client):
    previous_count = count_s3_objects(s3_client)
    client.force_login(alice)

    with file_py_path.open("rb") as f:
        response = client.post("/upload/", {"uploadDocs": f})

        assert response.status_code == HTTPStatus.OK
        assert "File type .py not supported" in str(response.content)
        assert count_s3_objects(s3_client) == previous_count


@pytest.mark.django_db()
def test_upload_view_no_file(alice, client):
    client.force_login(alice)

    response = client.post("/upload/")

    assert response.status_code == HTTPStatus.OK
    assert "No document selected" in str(response.content)


@pytest.mark.django_db()
def test_remove_doc_view(client: Client, alice: User, file_pdf_path: Path, s3_client: Client, requests_mock: Mocker):
    file_name = file_pdf_path.name

    client.force_login(alice)
    # we begin by removing any file in minio that has this key
    s3_client.delete_object(Bucket=settings.BUCKET_NAME, Key=file_name.replace(" ", "_"))

    previous_count = count_s3_objects(s3_client)

    mocked_response = {
        "key": file_name,
        "bucket": settings.BUCKET_NAME,
        "uuid": str(uuid.uuid4()),
    }
    requests_mock.post(
        f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file",
        status_code=201,
        json=mocked_response,
    )

    with file_pdf_path.open("rb") as f:
        # create file before testing deletion
        client.post("/upload/", {"uploadDocs": f})
        assert file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count + 1

        new_file = File.objects.filter(user=alice).order_by("-created_at")[0]
        requests_mock.delete(
            f"http://{settings.CORE_API_HOST}:{settings.CORE_API_PORT}/file/{new_file.core_file_uuid}",
            status_code=201,
            json=mocked_response,
        )

        client.post(f"/remove-doc/{new_file.id}", {"doc_id": new_file.id})
        assert not file_exists(s3_client, file_name)
        assert count_s3_objects(s3_client) == previous_count
        assert requests_mock.request_history[-1].method == "DELETE"
        assert File.objects.get(id=new_file.id).status == StatusEnum.deleted


@pytest.mark.django_db()
def test_remove_nonexistent_doc(alice: User, client: Client):
    # Given
    client.force_login(alice)
    nonexistent_uuid = uuid.uuid4()

    # When
    url = reverse("remove-doc", kwargs={"doc_id": nonexistent_uuid})
    response = client.get(url)

    # Then
    assert response.status_code == HTTPStatus.NOT_FOUND


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
    assert ChatMessage.objects.get(chat_history__id=session_id, role=ChatRoleEnum.user).text == "Are you there?"
    assert (
        ChatMessage.objects.get(chat_history__id=session_id, role=ChatRoleEnum.ai).text == "Good afternoon, Mr. Amor."
    )


@pytest.mark.django_db()
def test_post_message_to_existing_session(
    chat_history: ChatHistory, client: Client, requests_mock: Mocker, uploaded_file: File
):
    # Given
    client.force_login(chat_history.users)
    session_id = chat_history.id
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
    assert (
        ChatMessage.objects.get(chat_history__id=session_id, role=ChatRoleEnum.ai).text == "Good afternoon, Mr. Amor."
    )
    assert (
        ChatMessage.objects.get(chat_history__id=session_id, role=ChatRoleEnum.ai).source_files.first() == uploaded_file
    )
    assert initial_file_expiry_date != File.objects.get(core_file_uuid=uploaded_file.core_file_uuid).expires_at
    assert (
        Citation.objects.get(
            chat_message=ChatMessage.objects.get(chat_history__id=session_id, role=ChatRoleEnum.ai)
        ).text
        == "Here is a source chunk"
    )


@pytest.mark.django_db()
def test_post_message_with_files_selected(
    chat_history: ChatHistory, client: Client, requests_mock: Mocker, several_files: Sequence[File]
):
    # Given
    client.force_login(chat_history.users)
    session_id = chat_history.id
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
        list(ChatMessage.objects.get(chat_history__id=session_id, role=ChatRoleEnum.user).selected_files.all())
        == selected_files
    )
    assert json.loads(requests_mock.last_request.text).get("selected_files") == [
        {"uuid": str(f.core_file_uuid)} for f in selected_files
    ]


@pytest.mark.django_db()
def test_user_can_see_their_own_chats(chat_history: ChatHistory, alice: User, client: Client):
    # Given
    client.force_login(alice)

    # When
    response = client.get(f"/chats/{chat_history.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK


@pytest.mark.django_db()
def test_user_cannot_see_other_users_chats(chat_history: ChatHistory, bob: User, client: Client):
    # Given
    client.force_login(bob)

    # When
    response = client.get(f"/chats/{chat_history.id}/")

    # Then
    assert response.status_code == HTTPStatus.FOUND
    assert response.headers.get("Location") == "/chats/"


@pytest.mark.django_db()
def test_view_session_with_documents(chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(chat_message.chat_history.users)
    chat_id = chat_message.chat_history.id

    # When
    response = client.get(f"/chats/{chat_id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert b"original_file.txt" in response.content


@pytest.mark.django_db()
def test_chat_history_grouped_by_age(user_with_chats_with_messages_over_time: User, client: Client):
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


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (timezone.now().date(), "Today"),
        ((timezone.now() - timedelta(days=1)).date(), "Yesterday"),
        ((timezone.now() - timedelta(days=2)).date(), "Previous 7 days"),
        ((timezone.now() - timedelta(days=7)).date(), "Previous 7 days"),
        ((timezone.now() - timedelta(days=8)).date(), "Previous 30 days"),
        ((timezone.now() - timedelta(days=30)).date(), "Previous 30 days"),
        ((timezone.now() - timedelta(days=31)).date(), "Older than 30 days"),
    ],
)
def test_date_group_calculation(given: date, expected: str):
    # Given

    # When
    actual = ChatsView.get_date_group(given)

    # Then
    assert actual == expected


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
def test_post_chat_title(alice: User, chat_history: ChatHistory, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("chat-titles", kwargs={"chat_id": chat_history.id})
    response = client.post(url, json.dumps({"name": "New chat name"}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    chat_history.refresh_from_db()
    assert chat_history.name == "New chat name"


@pytest.mark.django_db()
def test_post_chat_title_with_naughty_string(alice: User, chat_history: ChatHistory, client: Client):
    # Given
    client.force_login(alice)

    # When
    url = reverse("chat-titles", kwargs={"chat_id": chat_history.id})
    response = client.post(url, json.dumps({"name": "New chat name \x00"}), content_type="application/json")

    # Then
    status = HTTPStatus(response.status_code)
    assert status.is_success
    chat_history.refresh_from_db()
    assert chat_history.name == "New chat name \ufffd"


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


@pytest.mark.django_db()
def test_staff_user_can_see_route(chat_history_with_files: ChatHistory, client: Client):
    # Given
    chat_history_with_files.users.is_staff = True
    chat_history_with_files.users.save()
    client.force_login(chat_history_with_files.users)

    # When
    response = client.get(f"/chats/{chat_history_with_files.id}/")

    # Then
    assert response.status_code == HTTPStatus.OK
    assert b"iai-chat-bubble__route" in response.content
    assert b"iai-chat-bubble__route govuk-!-display-none" not in response.content


@pytest.mark.django_db()
def test_check_demographics_redirect_if_unpopulated(client: Client, alice: User):
    # Given
    client.force_login(alice)

    # When
    response = client.get("/check-demographics/", follow=True)

    # Then
    assertRedirects(response, "/demographics/")


@pytest.mark.django_db()
def test_check_demographics_redirect_if_populated(client: Client, user_with_demographic_data: User):
    # Given
    client.force_login(user_with_demographic_data)

    # When
    response = client.get("/check-demographics/", follow=True)

    # Then
    assertRedirects(response, "/documents/")


@pytest.mark.django_db()
def test_view_demographic_details_form(client: Client, user_with_demographic_data: User):
    # Given
    client.force_login(user_with_demographic_data)

    # When
    response = client.get("/demographics/")

    # Then
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content)
    assert soup.find(id="id_grade").find_all("option", selected=True)[0].text == "Director General"
    assert soup.find(id="id_profession").find_all("option", selected=True)[0].text == "Analysis"
    assert soup.find(id="id_business_unit").find_all("option", selected=True)[0].text == "Paperclip Reconciliation"


@pytest.mark.django_db()
def test_post_to_demographic_details_form(client: Client, alice: User, business_unit: BusinessUnit):
    # Given
    client.force_login(alice)

    # When
    response = client.post(
        "/demographics/",
        {"grade": "AO", "profession": "AN", "business_unit": business_unit.id},
        follow=True,
    )

    # Then
    assertRedirects(response, "/documents/")
    alice.refresh_from_db()
    assert alice.grade == "AO"
