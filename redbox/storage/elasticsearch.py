import logging
from collections.abc import Sequence
from datetime import UTC, datetime
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


class ReindexError(Exception):
    """Migrating from one index mapping to another has failed."""


def create_or_update_index_mapping(es: Elasticsearch, alias: str, mapping: dict[str, Any]) -> None:
    try:
        alias_info = es.indices.get_alias(name=alias)
        old_index = next(iter(alias_info), None)
    except NotFoundError:
        alias_info = None
        old_index = None

    new_index = f"{alias}.{datetime.now(UTC).strftime('%Y-%m-%d-%H-%M-%S')}"

    if old_index is None:
        log.info("Creating new index %s, aliased to %s", new_index, alias)
        es.indices.create(
            index=new_index,
            aliases={alias: {}},
            mappings=mapping,
        )
    else:
        mapping_remote = es.indices.get_mapping(index=old_index)[old_index]["mappings"]

        if mapping_remote != mapping:
            log.info("Updating mapping for %s, aliased to %s", old_index, alias)
            es.indices.create(
                index=new_index,
                mappings=mapping,
            )

            response = es.reindex(source={"index": old_index}, dest={"index": new_index}, wait_for_completion=True)

            if len(response["failures"]) > 0:
                es.indices.delete(index=new_index)
                error_count = f"Reindexing failed with {len(response['failures'])} errors"
                raise ReindexError(error_count)

            es.indices.update_aliases(
                actions=[
                    {"remove": {"index": old_index, "alias": alias}},
                    {"add": {"index": new_index, "alias": alias}},
                ]
            )

            es.indices.delete(index=old_index)
            log.info(
                "New index of %s, aliased to %s, successfully created with data from %s", new_index, alias, old_index
            )
        else:
            log.info("No updates made to index %s, aliased to %s", new_index, alias)


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

        create_or_update_index_mapping(
            es=self.es_client,
            alias=f"{self.root_index}-file",
            mapping={
                "properties": {
                    "bucket": {"type": "keyword", "ignore_above": 50},
                    "created_datetime": {"type": "date"},
                    "creator_user_uuid": {"type": "keyword", "ignore_above": 36},
                    "key": {"type": "keyword"},
                    "model_type": {"type": "keyword", "ignore_above": 50},
                    "uuid": {"type": "keyword", "ignore_above": 36},
                }
            },
        )

        create_or_update_index_mapping(
            es=self.es_client,
            alias=f"{self.root_index}-chunk",
            mapping={
                "properties": {
                    "created_datetime": {"type": "date"},
                    "creator_user_uuid": {"type": "keyword", "ignore_above": 36},
                    "index": {"type": "long"},
                    "metadata": {
                        "properties": {
                            "languages": {"type": "text"},
                            "page_number": {"type": "long"},
                            "parent_doc_uuid": {"type": "keyword", "ignore_above": 36},
                        }
                    },
                    "model_type": {"type": "keyword", "ignore_above": 50},
                    "parent_file_uuid": {"type": "keyword", "ignore_above": 36},
                    "text": {"type": "text"},
                    "text_hash": {"type": "text"},
                    "token_count": {"type": "long"},
                    "uuid": {"type": "keyword", "ignore_above": 36},
                }
            },
        )

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
                query={"query": {"term": {"creator_user_uuid": str(user_uuid)}}},
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
                query={"query": {"term": {"creator_user_uuid": str(user_uuid)}}},
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
            Chunk(**item["_source"])
            for item in scan(
                client=self.es_client,
                index=target_index,
                query={
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "parent_file_uuid": str(parent_file_uuid),
                                    }
                                },
                                {
                                    "term": {
                                        "creator_user_uuid": str(user_uuid),
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
