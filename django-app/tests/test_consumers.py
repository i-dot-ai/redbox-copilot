import json
import logging
from asyncio import CancelledError
from collections.abc import Sequence
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.db.models import Model
from websockets import WebSocketClientProtocol
from websockets.legacy.client import Connect

from redbox_app.redbox_core import error_messages
from redbox_app.redbox_core.consumers import ChatConsumer
from redbox_app.redbox_core.models import ChatHistory, ChatMessage, ChatRoleEnum, File, User

logger = logging.getLogger(__name__)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_new_session(alice: User, uploaded_file: File, mocked_connect: Connect):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal."})
        response1 = await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)
        response3 = await communicator.receive_json_from(timeout=5)
        response4 = await communicator.receive_json_from(timeout=5)
        response5 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response2["type"] == "text"
        assert response2["data"] == "Good afternoon, "
        assert response3["type"] == "text"
        assert response3["data"] == "Mr. Amor."
        assert response4["type"] == "hidden-route"
        assert response4["data"] == "gratitude"
        assert response5["type"] == "source"
        assert response5["data"]["original_file_name"] == uploaded_file.original_file_name
        # Close
        await communicator.disconnect()

    assert await get_chat_message_text(alice, ChatRoleEnum.user) == ["Hello Hal."]
    assert await get_chat_message_text(alice, ChatRoleEnum.ai) == ["Good afternoon, Mr. Amor."]
    assert await get_chat_message_route(alice, ChatRoleEnum.ai) == ["gratitude"]

    expected_citations = {(None, ()), ("Good afternoon Mr Amor", ()), ("Good afternoon Mr Amor", (34, 35))}
    assert await get_chat_message_citation_set(alice, ChatRoleEnum.ai) == expected_citations
    await refresh_from_db(uploaded_file)
    assert uploaded_file.last_referenced.date() == datetime.now(tz=UTC).date()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_staff_user(staff_user: User, mocked_connect: Connect):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = staff_user
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal."})
        response1 = await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)
        response3 = await communicator.receive_json_from(timeout=5)
        response4 = await communicator.receive_json_from(timeout=5)
        _response5 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response2["type"] == "text"
        assert response2["data"] == "Good afternoon, "
        assert response3["type"] == "text"
        assert response3["data"] == "Mr. Amor."
        assert response4["type"] == "route"
        assert response4["data"] == "gratitude"
        # Close
        await communicator.disconnect()

    assert await get_chat_message_route(staff_user, ChatRoleEnum.ai) == ["gratitude"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_existing_session(alice: User, chat_history: ChatHistory, mocked_connect: Connect):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal.", "sessionId": str(chat_history.id)})
        response1 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response1["data"] == str(chat_history.id)

        # Close
        await communicator.disconnect()

    assert await get_chat_message_text(alice, ChatRoleEnum.user) == ["Hello Hal."]
    assert await get_chat_message_text(alice, ChatRoleEnum.ai) == ["Good afternoon, Mr. Amor."]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_naughty_question(alice: User, uploaded_file: File, mocked_connect: Connect):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal. \x00"})
        response1 = await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)
        response3 = await communicator.receive_json_from(timeout=5)
        response4 = await communicator.receive_json_from(timeout=5)
        response5 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response2["type"] == "text"
        assert response2["data"] == "Good afternoon, "
        assert response3["type"] == "text"
        assert response3["data"] == "Mr. Amor."
        assert response4["type"] == "hidden-route"
        assert response4["data"] == "gratitude"
        assert response5["type"] == "source"
        assert response5["data"]["original_file_name"] == uploaded_file.original_file_name
        # Close
        await communicator.disconnect()

    assert await get_chat_message_text(alice, ChatRoleEnum.user) == ["Hello Hal. \ufffd"]
    assert await get_chat_message_text(alice, ChatRoleEnum.ai) == ["Good afternoon, Mr. Amor."]
    assert await get_chat_message_route(alice, ChatRoleEnum.ai) == ["gratitude"]
    await refresh_from_db(uploaded_file)
    assert uploaded_file.last_referenced.date() == datetime.now(tz=UTC).date()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_naughty_citation(
    alice: User, uploaded_file: File, mocked_connect_with_naughty_citation: Connect
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect_with_naughty_citation):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal."})
        response1 = await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)
        response3 = await communicator.receive_json_from(timeout=5)
        response4 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response2["type"] == "text"
        assert response2["data"] == "Good afternoon, Mr. Amor."
        assert response3["type"] == "hidden-route"
        assert response3["data"] == "gratitude"
        assert response4["type"] == "source"
        assert response4["data"]["original_file_name"] == uploaded_file.original_file_name
        # Close
        await communicator.disconnect()

    assert await get_chat_message_text(alice, ChatRoleEnum.user) == ["Hello Hal."]
    assert await get_chat_message_text(alice, ChatRoleEnum.ai) == ["Good afternoon, Mr. Amor."]
    assert await get_chat_message_route(alice, ChatRoleEnum.ai) == ["gratitude"]
    await refresh_from_db(uploaded_file)
    assert uploaded_file.last_referenced.date() == datetime.now(tz=UTC).date()


@database_sync_to_async
def get_chat_message_text(user: User, role: ChatRoleEnum) -> Sequence[str]:
    return [m.text for m in ChatMessage.objects.filter(chat_history__users=user, role=role)]


@database_sync_to_async
def get_chat_message_citation_set(user: User, role: ChatRoleEnum) -> Sequence[tuple[str, tuple[int]]]:
    return {
        (citation.text, tuple(citation.page_numbers or []))
        for message in ChatMessage.objects.filter(chat_history__users=user, role=role)
        for source_file in message.source_files.all()
        for citation in source_file.citation_set.all()
    }


@database_sync_to_async
def get_chat_message_route(user: User, role: ChatRoleEnum) -> Sequence[str]:
    return [m.route for m in ChatMessage.objects.filter(chat_history__users=user, role=role)]


@pytest.mark.xfail()
@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_selected_files(
    alice: User,
    several_files: Sequence[File],
    chat_history_with_files: ChatHistory,
    mocked_connect_with_several_files: Connect,
):
    # Given
    selected_files: Sequence[File] = several_files[2:]

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect_with_several_files):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        selected_file_core_uuids: Sequence[str] = [str(f.core_file_uuid) for f in selected_files]
        await communicator.send_json_to(
            {
                "message": "Third question, with selected files?",
                "sessionId": str(chat_history_with_files.id),
                "selectedFiles": selected_file_core_uuids,
            }
        )
        response1 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response1["data"] == str(chat_history_with_files.id)

        # Close
        await communicator.disconnect()

    # Then

    # TODO (@brunns): Assert selected files sent to core.
    # Requires fix for https://github.com/django/channels/issues/1091
    mocked_websocket = mocked_connect_with_several_files.return_value.__aenter__.return_value
    expected = json.dumps(
        {
            "message_history": [
                {"role": "user", "text": "A question?"},
                {"role": "ai", "text": "An answer."},
                {"role": "user", "text": "A second question?"},
                {"role": "ai", "text": "A second answer."},
                {"role": "user", "text": "Third question, with selected files?"},
            ],
            "selected_files": selected_file_core_uuids,
        }
    )
    mocked_websocket.send.assert_called_with(expected)

    # TODO (@brunns): Assert selected files saved to model.
    # Requires fix for https://github.com/django/channels/issues/1091
    all_messages = get_chat_messages(alice)
    last_user_message = [m for m in all_messages if m.rule == ChatRoleEnum.user][-1]
    assert last_user_message.selected_files == selected_files


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_connection_error(alice: User, mocked_breaking_connect: Connect):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_breaking_connect):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal."})
        await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response2["type"] == "error"


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_explicit_unhandled_error(
    alice: User, mocked_connect_with_explicit_unhandled_error: Connect
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect_with_explicit_unhandled_error):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal."})
        response1 = await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)
        response3 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response2["type"] == "text"
        assert response2["data"] == "Good afternoon, "
        assert response3["type"] == "error"
        assert response3["data"] == error_messages.CORE_ERROR_MESSAGE
        # Close
        await communicator.disconnect()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio()
async def test_chat_consumer_with_explicit_no_document_selected_error(
    alice: User, mocked_connect_with_explicit_no_document_selected_error: Connect
):
    # Given

    # When
    with patch("redbox_app.redbox_core.consumers.connect", new=mocked_connect_with_explicit_no_document_selected_error):
        communicator = WebsocketCommunicator(ChatConsumer.as_asgi(), "/ws/chat/")
        communicator.scope["user"] = alice
        connected, _ = await communicator.connect()
        assert connected

        await communicator.send_json_to({"message": "Hello Hal."})
        response1 = await communicator.receive_json_from(timeout=5)
        response2 = await communicator.receive_json_from(timeout=5)

        # Then
        assert response1["type"] == "session-id"
        assert response2["type"] == "text"
        assert response2["data"] == error_messages.SELECT_DOCUMENT
        # Close
        await communicator.disconnect()


@database_sync_to_async
def get_chat_messages(user: User) -> Sequence[ChatMessage]:
    return list(
        ChatMessage.objects.filter(chat_history__users=user)
        .order_by("created_at")
        .prefetch_related("chat_history")
        .prefetch_related("source_files")
        .prefetch_related("selected_files")
    )


@pytest.fixture()
def mocked_connect(uploaded_file: File) -> Connect:
    mocked_websocket = AsyncMock(spec=WebSocketClientProtocol, name="mocked_websocket")
    mocked_connect = MagicMock(spec=Connect, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.return_value = [
        json.dumps({"resource_type": "text", "data": "Good afternoon, "}),
        json.dumps({"resource_type": "text", "data": "Mr. Amor."}),
        json.dumps({"resource_type": "route_name", "data": "gratitude"}),
        json.dumps({"resource_type": "documents", "data": [{"file_uuid": str(uploaded_file.core_file_uuid)}]}),
        json.dumps(
            {
                "resource_type": "documents",
                "data": [
                    {"file_uuid": str(uploaded_file.core_file_uuid), "page_content": "Good afternoon Mr Amor"},
                    {
                        "file_uuid": str(uploaded_file.core_file_uuid),
                        "page_content": "Good afternoon Mr Amor",
                        "page_numbers": [34, 35],
                    },
                ],
            }
        ),
        json.dumps({"resource_type": "end"}),
    ]
    return mocked_connect


@pytest.fixture()
def mocked_connect_with_naughty_citation(uploaded_file: File) -> Connect:
    mocked_websocket = AsyncMock(spec=WebSocketClientProtocol, name="mocked_websocket")
    mocked_connect = MagicMock(spec=Connect, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.return_value = [
        json.dumps({"resource_type": "text", "data": "Good afternoon, Mr. Amor."}),
        json.dumps({"resource_type": "route_name", "data": "gratitude"}),
        json.dumps(
            {
                "resource_type": "documents",
                "data": [
                    {"file_uuid": str(uploaded_file.core_file_uuid), "page_content": "Good afternoon Mr Amor"},
                    {"file_uuid": str(uploaded_file.core_file_uuid), "page_content": "I shouldn't send a \x00"},
                ],
            }
        ),
        json.dumps({"resource_type": "end"}),
    ]
    return mocked_connect


@pytest.fixture()
def mocked_breaking_connect() -> Connect:
    mocked_websocket = AsyncMock(spec=WebSocketClientProtocol, name="mocked_websocket")
    mocked_connect = MagicMock(spec=Connect, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.side_effect = CancelledError()
    return mocked_connect


@pytest.fixture()
def mocked_connect_with_explicit_unhandled_error() -> Connect:
    mocked_websocket = AsyncMock(spec=WebSocketClientProtocol, name="mocked_websocket")
    mocked_connect = MagicMock(spec=Connect, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.return_value = [
        json.dumps({"resource_type": "text", "data": "Good afternoon, "}),
        json.dumps({"resource_type": "error", "data": {"code": "unknown", "message": "Oh dear."}}),
    ]
    return mocked_connect


@pytest.fixture()
def mocked_connect_with_explicit_no_document_selected_error() -> Connect:
    mocked_websocket = AsyncMock(spec=WebSocketClientProtocol, name="mocked_websocket")
    mocked_connect = MagicMock(spec=Connect, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.return_value = [
        json.dumps({"resource_type": "error", "data": {"code": "no-document-selected", "message": "whatever"}}),
    ]
    return mocked_connect


@pytest.fixture()
def mocked_connect_with_several_files(several_files: Sequence[File]) -> Connect:
    mocked_websocket = AsyncMock(spec=WebSocketClientProtocol, name="mocked_websocket")
    mocked_connect = MagicMock(spec=Connect, name="mocked_connect")
    mocked_connect.return_value.__aenter__.return_value = mocked_websocket
    mocked_websocket.__aiter__.return_value = [
        json.dumps({"resource_type": "text", "data": "Third "}),
        json.dumps({"resource_type": "text", "data": "answer."}),
        json.dumps(
            {
                "resource_type": "documents",
                "data": [
                    {"file_uuid": str(f.core_file_uuid), "page_content": "a secret forth answer"}
                    for f in several_files[2:]
                ],
            }
        ),
        json.dumps({"resource_type": "end"}),
    ]
    return mocked_connect


@database_sync_to_async
def refresh_from_db(obj: Model) -> None:
    obj.refresh_from_db()
