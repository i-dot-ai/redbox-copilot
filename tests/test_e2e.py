import json
import time
from http import HTTPStatus
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

import pytest
import requests
import websockets
from jose import jwt
from websockets import ConnectionClosed

USER_UUIDS: list[UUID] = [uuid4(), uuid4()]
TEST_ORIGIN = "localhost:5002"


def make_headers(user_uuid: UUID):
    token = jwt.encode({"user_uuid": str(user_uuid)}, key="super-secure-private-key")
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.incremental()
class TestEndToEnd:
    file_uuids: ClassVar[dict[UUID, str]] = {}
    source_document_file_uuids: ClassVar[dict[UUID, set[str]]] = {}
    route_name: str = ""

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_upload_to_search(self, file_path: Path, s3_client, user_uuid):
        """
        Given that I have uploaded a file to s3
        When I POST the file key to core-api/file
        I Expect a file uuid to be returned
        """
        with file_path.open("rb") as f:
            file_key = file_path.name
            file_type = file_path.suffix
            bucket_name = "redbox-storage-dev"
            s3_client.upload_fileobj(
                Bucket=bucket_name,
                Fileobj=f,
                Key=file_key,
                ExtraArgs={"Tagging": f"file_type={file_type}"},
            )

            response = requests.post(
                url=f"http://{TEST_ORIGIN}/file",
                json={
                    "key": file_key,
                    "bucket": bucket_name,
                },
                headers=make_headers(user_uuid),
                timeout=30,
            )
            TestEndToEnd.file_uuids[user_uuid] = response.json()["uuid"]
            assert response.status_code == HTTPStatus.CREATED

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_get_file_status(self, user_uuid):
        """
        Given that I have POSTed a file key to core-api/file
        When I GET the file status
        I Expect the file status to be "completed" within 120 seconds
        """
        timeout = 120
        start_time = time.time()
        error = None
        embedding_complete = False

        while time.time() - start_time < timeout:
            time.sleep(1)
            chunk_response = requests.get(
                f"http://{TEST_ORIGIN}/file/{TestEndToEnd.file_uuids[user_uuid]}/status",
                headers=make_headers(user_uuid),
                timeout=30,
            )
            if chunk_response.status_code == HTTPStatus.OK and chunk_response.json()["processing_status"] == "complete":
                embedding_complete = True
                break  # test passed
            else:
                error = chunk_response.text

        if not embedding_complete:
            pytest.fail(reason=f"failed to get embedded chunks within {timeout} seconds, potential error: {error}")

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_get_file_chunks(self, user_uuid):
        """
        Given that I have POSTed a file key to core-api/file
        And the file status is complete
        When I GET the file chunks
        I Expect a 200 response code
        """
        chunks_response = requests.get(
            f"http://{TEST_ORIGIN}/file/{TestEndToEnd.file_uuids[user_uuid]}/chunks",
            headers=make_headers(user_uuid),
            timeout=30,
        )
        assert chunks_response.status_code == HTTPStatus.OK

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_post_rag(self, user_uuid):
        """
        Given that I have POSTed a file key to core-api/file
        And the file status is complete
        When I POST a question to the rag endpoint
        I Expect an answer and for the cited documents to be the one I uploaded
        """
        rag_response = requests.post(
            f"http://{TEST_ORIGIN}/chat/rag",
            json={
                "message_history": [
                    {"role": "user", "text": "what is routing?"},
                ]
            },
            headers=make_headers(user_uuid),
            timeout=30,
        )
        assert rag_response.status_code == HTTPStatus.OK
        source_document_file_uuids = {
            source_document["file_uuid"] for source_document in rag_response.json()["source_documents"]
        }

        assert TestEndToEnd.file_uuids[user_uuid] in source_document_file_uuids
        TestEndToEnd.source_document_file_uuids[user_uuid] = source_document_file_uuids

        TestEndToEnd.route_name = rag_response.json()["route_name"]
        assert TestEndToEnd.route_name == "search"

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_post_rag_fail(self, user_uuid):
        """
        Given that I have POSTed a file key to core-api/file
        And the file status is complete
        When I POST a question to the rag endpoint with the wrong file selected
        I Expect an answer and for no cited documents to be returned
        """
        rag_response = requests.post(
            f"http://{TEST_ORIGIN}/chat/rag",
            json={
                "message_history": [
                    {"role": "user", "text": "what is routing?"},
                ],
                "selected_files": [{"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}],
            },
            headers=make_headers(user_uuid),
            timeout=30,
        )
        assert rag_response.status_code == HTTPStatus.OK

        source_document_file_uuids = {
            source_document["file_uuid"] for source_document in rag_response.json()["source_documents"]
        }
        assert TestEndToEnd.file_uuids[user_uuid] not in source_document_file_uuids

        TestEndToEnd.route_name = rag_response.json()["route_name"]
        assert TestEndToEnd.route_name == "search"

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_post_rag_summarisation(self, user_uuid):
        rag_response = requests.post(
            f"http://{TEST_ORIGIN}/chat/rag",
            json={
                "message_history": [
                    {
                        "role": "user",
                        "text": "Please summarise the contents of the uploaded files.",
                    }
                ],
                "selected_files": [{"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}],
            },
            headers=make_headers(user_uuid),
            timeout=30,
        )
        assert rag_response.status_code == HTTPStatus.OK

        TestEndToEnd.route_name = rag_response.json()["route_name"]
        assert TestEndToEnd.route_name == "summarise"

    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    def test_permissions(self, user_uuid):
        """
        Given that I have POSTed a file key to core-api/file
        And the file status is complete, And asked some RAG questions
        Even though the uploaded documents and RAG questions are identical
        I Expect that the sourced documents are only available to the user
        that uploaded them.
        """

        for other_user_uuid, source_document_file_uuids in TestEndToEnd.source_document_file_uuids.items():
            if other_user_uuid != user_uuid:
                assert TestEndToEnd.file_uuids[user_uuid] not in source_document_file_uuids

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    async def test_streaming_rag(self, user_uuid):
        """
        Given a legitimate message_history
        When I send to ws://<host>/chat/rag
        I expect a text response
        """
        message_history = {
            "message_history": [
                {"role": "user", "text": "what is routing"},
            ]
        }
        all_text, source_documents = [], []

        async for websocket in websockets.connect(
            f"ws://{TEST_ORIGIN}/chat/rag", extra_headers=make_headers(user_uuid)
        ):
            await websocket.send(json.dumps(message_history))

            try:
                while True:
                    actual_str = await websocket.recv()
                    actual = json.loads(actual_str)
                    if actual["resource_type"] == "text":
                        all_text.append(actual["data"])
                    elif actual["resource_type"] == "documents":
                        source_documents.extend(actual["data"])
                    elif actual["resource_type"] == "route_name":
                        TestEndToEnd.route_name = actual["data"]
            except ConnectionClosed:
                break

        assert all_text
        source_document_file_uuids = {source_document["file_uuid"] for source_document in source_documents}

        assert TestEndToEnd.file_uuids[user_uuid] in source_document_file_uuids
        TestEndToEnd.source_document_file_uuids[user_uuid] = source_document_file_uuids

        assert TestEndToEnd.route_name == "search"

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("user_uuid", USER_UUIDS)
    async def test_streaming_gratitude(self, user_uuid):
        """
        Given a pleasant message
        When I send to ws://<host>/chat/rag
        I expect a pleasant response!
        """
        message_history = {
            "message_history": [
                {"role": "user", "text": "thank you"},
            ]
        }
        all_text, source_documents = [], []

        async for websocket in websockets.connect(
            f"ws://{TEST_ORIGIN}/chat/rag", extra_headers=make_headers(user_uuid)
        ):
            await websocket.send(json.dumps(message_history))

            try:
                while True:
                    actual_str = await websocket.recv()
                    actual = json.loads(actual_str)
                    if actual["resource_type"] == "text":
                        all_text.append(actual["data"])
                    elif actual["resource_type"] == "documents":
                        source_documents.extend(actual["data"])
                    elif actual["resource_type"] == "route_name":
                        TestEndToEnd.route_name = actual["data"]
            except ConnectionClosed:
                break

        assert all_text == ["You're welcome!"]

        assert TestEndToEnd.route_name == "gratitude"
