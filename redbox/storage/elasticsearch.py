import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import scan
from pydantic import ValidationError

from redbox.models import Chunk, ChunkStatus, FileStatus, ProcessingStatusEnum
from redbox.models.base import PersistableModel
from redbox.storage.storage_handler import BaseStorageHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class ElasticsearchStorageHandler(BaseStorageHandler):
    """Storage Handler for Elasticsearch"""

    def __init__(
        self,
        es_client: Elasticsearch,
        root_index: str,
    ):
        """Initialise the storage handler

        Args:
            es_client (Elasticsearch): Elasticsearch client
            root_index (str, optional): Root index to use. Defaults to "redbox".
        """
        self.es_client = es_client
        self.root_index = root_index

    def refresh(self, index: str = "*") -> ObjectApiResponse:
        return self.es_client.indices.refresh(index=f"{self.root_index}-{index}")

    def write_item(self, item: PersistableModel) -> ObjectApiResponse:
        target_index = f"{self.root_index}-{item.model_type.lower()}"

        return self.es_client.index(
            index=target_index,
            id=str(item.uuid),
            body=item.model_dump(mode="json"),
        )

    def write_items(self, items: Sequence[PersistableModel]) -> Sequence[ObjectApiResponse]:
        return list(map(self.write_item, items))

    def read_item(self, item_uuid: UUID, model_type: str):
        target_index = f"{self.root_index}-{model_type.lower()}"
        result = self.es_client.get(index=target_index, id=str(item_uuid))
        model = self.get_model_by_model_type(model_type)
        return model(**result.body["_source"])

    def read_items(self, item_uuids: list[UUID], model_type: str):
        target_index = f"{self.root_index}-{model_type.lower()}"
        result = self.es_client.mget(index=target_index, body={"ids": list(map(str, item_uuids))})

        model = self.get_model_by_model_type(model_type)
        return [model(**item["_source"]) for item in result.body["docs"]]

    def update_item(self, item: PersistableModel) -> ObjectApiResponse:
        target_index = f"{self.root_index}-{item.model_type.lower()}"

        return self.es_client.index(
            index=target_index,
            id=str(item.uuid),
            body=item.model_dump(mode="json"),
        )

    def update_items(self, items: list[PersistableModel]) -> list[ObjectApiResponse]:
        return list(map(self.update_item, items))

    def delete_item(self, item: PersistableModel) -> ObjectApiResponse:
        target_index = f"{self.root_index}-{item.model_type.lower()}"
        return self.es_client.delete(index=target_index, id=str(item.uuid))

    def delete_items(self, items: list[PersistableModel]) -> ObjectApiResponse | None:
        if not items:
            return None

        if len({item.model_type for item in items}) > 1:
            message = "Items with differing model types: {item.model_type for item in items}"
            raise ValueError(message)
        model_type = items[0].model_type
        target_index = f"{self.root_index}-{model_type.lower()}"
        return self.es_client.delete_by_query(
            index=target_index,
            body={"query": {"terms": {"_id": [str(item.uuid) for item in items]}}},
        )

    def read_all_items(self, model_type: str, user_uuid: UUID) -> list[PersistableModel]:
        target_index = f"{self.root_index}-{model_type.lower()}"
        try:
            result = scan(
                client=self.es_client,
                index=target_index,
                query={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"creator_user_uuid.keyword": str(user_uuid)}},
                                {"term": {"metadata.creator_user_uuid.keyword": str(user_uuid)}},
                            ]
                        }
                    }
                },
                _source=True,
            )

        except NotFoundError:
            log.info("Index %s not found. Returning empty list.", target_index)
            return []

        # Grab the model we'll use to deserialize the items
        model = self.get_model_by_model_type(model_type)
        try:
            results = list(result)
        except NotFoundError:
            return []

        items = []
        for item in results:
            try:
                items.append(model(**item["_source"]))
            except ValidationError as e:
                log.exception("Validation exception for %s", item, exc_info=e)
        return items

    def list_all_items(self, model_type: str, user_uuid: UUID) -> list[UUID]:
        target_index = f"{self.root_index}-{model_type.lower()}"
        try:
            # Only return _id
            results = scan(
                client=self.es_client,
                index=target_index,
                query={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"creator_user_uuid.keyword": str(user_uuid)}},
                                {"term": {"metadata.creator_user_uuid.keyword": str(user_uuid)}},
                            ]
                        }
                    }
                },
                _source=False,
            )

        except NotFoundError:
            log.info("Index %s not found. Returning empty list.", target_index)
            return []
        return [UUID(item["_id"]) for item in results]

    def get_file_chunks(self, parent_file_uuid: UUID, user_uuid: UUID) -> list[Chunk]:
        """get chunks for a given file"""
        target_index = f"{self.root_index}-chunk"

        return [
            hit_to_chunk(item)
            for item in scan(
                client=self.es_client,
                index=target_index,
                query={
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "bool": {
                                        "should": [
                                            {"term": {"parent_file_uuid.keyword": str(parent_file_uuid)}},
                                            {"term": {"metadata.parent_file_uuid.keyword": str(parent_file_uuid)}},
                                        ]
                                    }
                                },
                                {
                                    "bool": {
                                        "should": [
                                            {"term": {"creator_user_uuid.keyword": str(user_uuid)}},
                                            {"term": {"metadata.creator_user_uuid.keyword": str(user_uuid)}},
                                        ]
                                    }
                                },
                            ]
                        }
                    }
                },
            )
        ]

    def get_file_status(self, file_uuid: UUID, user_uuid: UUID) -> FileStatus:
        """Get the status of a file and associated Chunks

        Args:
            file_uuid (UUID): The UUID of the file to get the status of
            user_uuid (UUID): the UUID of the user

        Returns:
            FileStatus: The status of the file
        """

        # Test 1: Get the file
        try:
            file = self.read_item(file_uuid, "File")
        except NotFoundError as e:
            log.exception("file/%s not found", file_uuid)
            message = f"File {file_uuid} not found"
            raise ValueError(message) from e
        if file.creator_user_uuid != user_uuid:
            log.error("file/%s.%s not owned by %s", file_uuid, file.creator_user_uuid, user_uuid)
            message = f"File {file_uuid} not found"
            raise ValueError(message)

        # Test 2: Get the number of chunks for the file
        chunks = self.get_file_chunks(file_uuid, file.creator_user_uuid)

        if not chunks:
            # File has not been chunked yet
            return FileStatus(
                file_uuid=file_uuid,
                chunk_statuses=[],
                processing_status=ProcessingStatusEnum.chunking,
            )

        # Test 3: Determine the number of embedded chunks for the file
        chunk_statuses = [ChunkStatus(chunk_uuid=chunk.uuid, embedded=bool(chunk.embedding)) for chunk in chunks]

        # Test 4: Determine the latest status
        is_complete = all(chunk_status.embedded for chunk_status in chunk_statuses)

        return FileStatus(
            file_uuid=file_uuid,
            chunk_statuses=chunk_statuses,
            processing_status=ProcessingStatusEnum.complete if is_complete else ProcessingStatusEnum.embedding,
        )


def hit_to_chunk(hit: dict[str, Any]) -> Chunk:
    if hit["_source"].get("uuid"):
        # Legacy direct chunk storage
        return Chunk(**hit["_source"])
    else:
        # Document storage
        return Chunk(
            uuid=hit["_id"],
            text=hit["_source"]["text"],
            index=hit["_source"]["metadata"]["index"],
            embedding=hit["_source"]["embedding"],
            created_datetime=hit["_source"]["metadata"]["created_datetime"],
            creator_user_uuid=hit["_source"]["metadata"]["creator_user_uuid"],
            parent_file_uuid=hit["_source"]["metadata"]["parent_file_uuid"],
            metadata=hit["_source"]["metadata"],
        )
