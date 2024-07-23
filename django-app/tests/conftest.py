import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile, UploadedFile
from django.core.management import call_command
from django.utils import timezone
from freezegun import freeze_time

from redbox_app.redbox_core import client
from redbox_app.redbox_core.models import (
    BusinessUnit,
    ChatHistory,
    ChatMessage,
    ChatMessageRating,
    ChatMessageRatingChip,
    ChatRoleEnum,
    Citation,
    File,
    User,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="session")
def _collect_static():
    call_command("collectstatic", "--no-input")


@pytest.fixture()
def create_user():
    def _create_user(email, date_joined_iso, is_staff=False):
        date_joined = datetime.fromisoformat(date_joined_iso).astimezone(UTC)
        return User.objects.create_user(email=email, date_joined=date_joined, is_staff=is_staff)

    return _create_user


@pytest.fixture()
def alice(create_user):
    return create_user("alice@cabinetoffice.gov.uk", "2000-01-01")


@pytest.fixture()
def bob(create_user):
    return create_user("bob@example.com", "2000-01-01")


@pytest.fixture()
def chris(create_user):
    return create_user("chris@example.com", "2000-01-02")


@pytest.fixture()
def peter_rabbit():
    return User.objects.create_user(email="peter.rabbit@example.com", password="P455W0rd")


@pytest.fixture()
def jemima_puddleduck():
    return User.objects.create_user(email="jemima.puddleduck@example.com", password="P455W0rd")


@pytest.fixture()
def user_with_demographic_data(business_unit: BusinessUnit) -> User:
    return User.objects.create_user(
        email="mrs.tiggywinkle@example.com", grade="DG", business_unit=business_unit, profession="AN"
    )


@pytest.fixture()
def staff_user(create_user):
    return create_user("staff@example.com", "2000-01-01", True)


@pytest.fixture()
def superuser() -> User:
    return User.objects.create_superuser("super@example.com", "2000-01-01")


@pytest.fixture()
def business_unit() -> BusinessUnit:
    return BusinessUnit.objects.create(name="Paperclip Reconciliation")


@pytest.fixture()
def file_pdf_path() -> Path:
    return Path(__file__).parent / "data" / "pdf" / "Cabinet Office - Wikipedia.pdf"


@pytest.fixture()
def file_py_path() -> Path:
    return Path(__file__).parent / "data" / "py" / "test_data.py"


@pytest.fixture()
def s3_client():
    return client.s3_client()


@pytest.fixture()
def chat_history(alice: User) -> ChatHistory:
    session_id = uuid.uuid4()
    return ChatHistory.objects.create(id=session_id, users=alice, name="A chat")


@pytest.fixture()
def chat_message(chat_history: ChatHistory, uploaded_file: File) -> ChatMessage:
    chat_message = ChatMessage.objects.create(
        chat_history=chat_history, text="A question?", role=ChatRoleEnum.user, route="A route"
    )
    chat_message.source_files.set([uploaded_file])
    return chat_message


@pytest.fixture()
def chat_message_with_citation(chat_history: ChatHistory, uploaded_file: File) -> ChatMessage:
    chat_message = ChatMessage.objects.create(chat_history=chat_history, text="An answer.", role=ChatRoleEnum.ai)
    Citation.objects.create(file=uploaded_file, chat_message=chat_message, text="Lorem ipsum.")
    return chat_message


@pytest.fixture()
def uploaded_file(alice: User, original_file: UploadedFile, s3_client) -> File:  # noqa: ARG001
    file = File.objects.create(
        user=alice,
        original_file=original_file,
        original_file_name=original_file.name,
        core_file_uuid=uuid.uuid4(),
        last_referenced=datetime.now(tz=UTC) - timedelta(days=14),
    )
    file.save()
    yield file
    file.delete()


@pytest.fixture()
def original_file() -> UploadedFile:
    return SimpleUploadedFile("original_file.txt", b"Lorem Ipsum.")


@pytest.fixture()
def chat_history_with_files(chat_history: ChatHistory, several_files: Sequence[File]) -> ChatHistory:
    ChatMessage.objects.create(chat_history=chat_history, text="A question?", role=ChatRoleEnum.user)
    chat_message = ChatMessage.objects.create(
        chat_history=chat_history, text="An answer.", role=ChatRoleEnum.ai, route="search"
    )
    chat_message.source_files.set(several_files[0::2])
    chat_message = ChatMessage.objects.create(
        chat_history=chat_history, text="A second question?", role=ChatRoleEnum.user
    )
    chat_message.selected_files.set(several_files[0:2])
    chat_message = ChatMessage.objects.create(
        chat_history=chat_history, text="A second answer.", role=ChatRoleEnum.ai, route="search"
    )
    chat_message.source_files.set([several_files[2]])
    return chat_history


@pytest.fixture()
def chat_history_with_messages_over_time(chat_history: ChatHistory) -> ChatHistory:
    now = timezone.now()
    with freeze_time(now - timedelta(days=40)):
        ChatMessage.objects.create(chat_history=chat_history, text="40 days old", role=ChatRoleEnum.user)
    with freeze_time(now - timedelta(days=20)):
        ChatMessage.objects.create(chat_history=chat_history, text="20 days old", role=ChatRoleEnum.user)
    with freeze_time(now - timedelta(days=5)):
        ChatMessage.objects.create(chat_history=chat_history, text="5 days old", role=ChatRoleEnum.user)
    with freeze_time(now - timedelta(days=1)):
        ChatMessage.objects.create(chat_history=chat_history, text="yesterday", role=ChatRoleEnum.user)
    ChatMessage.objects.create(chat_history=chat_history, text="today", role=ChatRoleEnum.user)
    return chat_history


@pytest.fixture()
def user_with_chats_with_messages_over_time(alice: User) -> User:
    now = timezone.now()
    with freeze_time(now - timedelta(days=40)):
        chats = [
            ChatHistory.objects.create(id=uuid.uuid4(), users=alice, name="40 days old"),
            ChatHistory.objects.create(id=uuid.uuid4(), users=alice, name="20 days old"),
            ChatHistory.objects.create(id=uuid.uuid4(), users=alice, name="5 days old"),
            ChatHistory.objects.create(id=uuid.uuid4(), users=alice, name="yesterday"),
            ChatHistory.objects.create(id=uuid.uuid4(), users=alice, name="today"),
        ]
        ChatMessage.objects.create(chat_history=chats[0], text="40 days old", role=ChatRoleEnum.user)
    with freeze_time(now - timedelta(days=20)):
        ChatMessage.objects.create(chat_history=chats[1], text="20 days old", role=ChatRoleEnum.user)
    with freeze_time(now - timedelta(days=5)):
        ChatMessage.objects.create(chat_history=chats[2], text="5 days old", role=ChatRoleEnum.user)
    with freeze_time(now - timedelta(days=1)):
        ChatMessage.objects.create(chat_history=chats[3], text="yesterday", role=ChatRoleEnum.user)
    ChatMessage.objects.create(chat_history=chats[4], text="today", role=ChatRoleEnum.user)

    return alice


@pytest.fixture()
def several_files(alice: User, number_to_create: int = 4) -> Sequence[File]:
    files = []
    for i in range(number_to_create):
        filename = f"original_file_{i}.txt"
        files.append(
            File.objects.create(
                user=alice,
                original_file=SimpleUploadedFile(filename, b"Lorem Ipsum."),
                original_file_name=filename,
                core_file_uuid=uuid.uuid4(),
            )
        )
    return files


@pytest.fixture()
def chat_message_with_rating(chat_message: ChatMessage) -> ChatMessage:
    chat_message_rating = ChatMessageRating(chat_message=chat_message, rating=3, text="Ipsum Lorem.")
    chat_message_rating.save()
    ChatMessageRatingChip(rating_id=chat_message_rating.pk, text="speed").save()
    ChatMessageRatingChip(rating_id=chat_message_rating.pk, text="accuracy").save()
    ChatMessageRatingChip(rating_id=chat_message_rating.pk, text="blasphemy").save()
    return chat_message
