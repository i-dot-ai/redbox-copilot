import time
from typing import Any
from uuid import UUID, uuid4

import pytest
from elasticsearch import Elasticsearch, NotFoundError

from redbox.models import Chunk
from redbox.storage.elasticsearch import ElasticsearchStorageHandler, create_or_update_index_mapping


def get_alias_first_index_mapping(client: Elasticsearch, alias: str) -> dict[str, Any]:
    alias_info = client.indices.get_alias(name=alias)
    index_name = next(iter(alias_info))

    return client.indices.get_mapping(index=index_name)[index_name]["mappings"]


def delete_alias_and_indices(client: Elasticsearch, alias: str) -> None:
    try:
        alias_info = client.indices.get_alias(name=alias)
        for index in alias_info:
            client.indices.delete(index=index)
    except NotFoundError:
        pass


def test_elasticsearch_client_connection(elasticsearch_client, elasticsearch_storage_handler):
    """
    Given that I have a valid Elasticsearch client
    When I call the info method
    Then I expect to see a valid response

    This test is to check all our following elasticsearch tests can proceed.
    """
    conn_test_resp = elasticsearch_client.info()
    assert conn_test_resp["tagline"] == "You Know, for Search"

    assert isinstance(elasticsearch_storage_handler.model_type_map, dict)


def test_create_or_update_index_mapping(elasticsearch_client):
    """
    Given that I have an empty index
    When I create a new one with mapping A and add a document
    Then I expect to see a that index with mapping A and document in remote

    Given that I have a created index with mapping A and a document
    When I create the same index with mapping B
    Then I expect it to be created successfully and all data moved over
    """
    index = "foo_bar"
    document = {"foo": "bar"}
    mapping_a = {"properties": {"foo": {"type": "text"}}}
    mapping_b = {"properties": {"foo": {"type": "keyword"}}}

    delete_alias_and_indices(elasticsearch_client, index)

    time.sleep(1)

    create_or_update_index_mapping(elasticsearch_client, index, mapping_a)
    elasticsearch_client.index(index=index, document=document)

    time.sleep(1)

    mapping_remote = get_alias_first_index_mapping(client=elasticsearch_client, alias=index)
    document_remote = elasticsearch_client.search(index=index, query={"match": {"foo": "bar"}})["hits"]["hits"][0]

    assert mapping_remote == mapping_a
    assert document_remote["_source"] == document

    create_or_update_index_mapping(elasticsearch_client, index, mapping_b)

    time.sleep(1)

    mapping_remote = get_alias_first_index_mapping(client=elasticsearch_client, alias=index)
    document_remote = elasticsearch_client.search(index=index, query={"term": {"foo": "bar"}})["hits"]["hits"][0]

    assert mapping_remote == mapping_b
    assert document_remote["_source"] == document

    delete_alias_and_indices(elasticsearch_client, index)

    time.sleep(1)


def test_elasticsearch_created_index(elasticsearch_client, elasticsearch_storage_handler, env):
    """
    Given that I have a valid Elasticsearch storage handler
    When I check the indices
    Then I expect their mappings to match my definitions
    """
    assert elasticsearch_storage_handler

    file_mapping = get_alias_first_index_mapping(client=elasticsearch_client, alias=f"{env.elastic_root_index}-file")
    file_mapping_ref = {
        "properties": {
            "bucket": {"type": "keyword", "ignore_above": 50},
            "created_datetime": {"type": "date"},
            "creator_user_uuid": {"type": "keyword", "ignore_above": 36},
            "key": {"type": "keyword"},
            "model_type": {"type": "keyword", "ignore_above": 50},
            "uuid": {"type": "keyword", "ignore_above": 36},
        }
    }

    assert file_mapping.items() <= file_mapping_ref.items()

    chunk_mapping = get_alias_first_index_mapping(client=elasticsearch_client, alias=f"{env.elastic_root_index}-chunk")
    chunk_mapping_ref = {
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
    }

    assert chunk_mapping.items() <= chunk_mapping_ref.items()


def test_elasticsearch_write_read_item(elasticsearch_storage_handler, chunk_belonging_to_alice):
    """
    Given that `Chunk` is a valid model
    When I
    Then I expect a valid Chunk to be returned on read"
    """
    # Write the chunk
    elasticsearch_storage_handler.write_item(item=chunk_belonging_to_alice)

    # Read the chunk
    chunk_read = elasticsearch_storage_handler.read_item(chunk_belonging_to_alice.uuid, "Chunk")

    assert chunk_read.uuid == chunk_belonging_to_alice.uuid


def test_elastic_read_item(elasticsearch_storage_handler, stored_chunk_belonging_to_alice):
    read_chunk = elasticsearch_storage_handler.read_item(stored_chunk_belonging_to_alice.uuid, "Chunk")
    assert read_chunk.uuid == stored_chunk_belonging_to_alice.uuid
    assert read_chunk.parent_file_uuid == stored_chunk_belonging_to_alice.parent_file_uuid
    assert read_chunk.index == stored_chunk_belonging_to_alice.index
    assert read_chunk.text == stored_chunk_belonging_to_alice.text
    assert read_chunk.metadata == stored_chunk_belonging_to_alice.metadata
    assert read_chunk.creator_user_uuid == stored_chunk_belonging_to_alice.creator_user_uuid
    assert read_chunk.token_count == stored_chunk_belonging_to_alice.token_count


def test_elastic_delete_item_fail(
    elasticsearch_storage_handler: ElasticsearchStorageHandler,
    chunk_belonging_to_bob,
):
    """
    Given that I have an non-existent item uuid
    When I call delete_item on it
    Then I expect to see a NotFoundError error raised
    """
    with pytest.raises(NotFoundError):
        elasticsearch_storage_handler.delete_item(chunk_belonging_to_bob)


def test_elastic_read_item_fail(
    elasticsearch_storage_handler: ElasticsearchStorageHandler,
):
    """
    Given that I have an non-existent item uuid
    When I call read_item on its uuid
    Then I expect to see a NotFoundError error raised
    """
    with pytest.raises(NotFoundError):
        elasticsearch_storage_handler.read_item(uuid4(), "Chunk")


def test_elastic_write_read_delete_items(elasticsearch_storage_handler):
    """
    Given that I have a list of items
    When I call write_items on them
    Then I expect to see them written to the database
    """
    creator_user_uuid = uuid4()
    chunks = [
        Chunk(
            creator_user_uuid=creator_user_uuid,
            parent_file_uuid=uuid4(),
            index=i,
            text="test_text",
        )
        for i in range(10)
    ]

    elasticsearch_storage_handler.write_items(chunks)

    read_chunks = elasticsearch_storage_handler.read_items([chunk.uuid for chunk in chunks], "Chunk")

    assert read_chunks == chunks

    # Delete the chunks
    elasticsearch_storage_handler.delete_items(chunks)

    # Check that the chunks are deleted
    items_left = elasticsearch_storage_handler.list_all_items("Chunk", creator_user_uuid)

    assert all(chunk.uuid not in items_left for chunk in chunks)


def test_list_all_items(
    elasticsearch_storage_handler: ElasticsearchStorageHandler,
    stored_chunk_belonging_to_alice: Chunk,
    stored_chunk_belonging_to_bob: Chunk,
    chunk_belonging_to_claire: Chunk,
    alice: UUID,
):
    """
    Given that I have
    * a saved chunk belonging to Alice
    * a saved chunk belonging to Bob
    * an unsaved chunk belonging to Claire
    When I call list_all_items as alice
    Then I expect to see the uuids of the saved objects that belong to alice returned
    """
    uuids = elasticsearch_storage_handler.list_all_items("Chunk", alice)
    assert len(uuids) == 1


def test_read_all_items(
    elasticsearch_storage_handler: ElasticsearchStorageHandler,
    stored_chunk_belonging_to_alice: Chunk,
    stored_chunk_belonging_to_bob: Chunk,
    chunk_belonging_to_claire: Chunk,
    alice: UUID,
):
    """
    Given that I have
    * a saved chunk belonging to Alice
    * a saved chunk belonging to Bob
    * an unsaved chunk belonging to Claire
    When I call read_all_items as alice
    Then I expect to see the one Chunk belonging to alice
    """
    chunks = elasticsearch_storage_handler.read_all_items("Chunk", alice)
    assert len(chunks) == 1
    assert chunks[0].creator_user_uuid == alice


def test_elastic_delete_item(elasticsearch_storage_handler, stored_chunk_belonging_to_alice):
    """
    Given that I have a saved object
    When I call delete_item on it
    Then I expect to not be able to read the item
    """
    elasticsearch_storage_handler.delete_item(stored_chunk_belonging_to_alice)

    with pytest.raises(NotFoundError):
        elasticsearch_storage_handler.read_item(stored_chunk_belonging_to_alice.uuid, "Chunk")


def test_get_file_chunks(
    elasticsearch_storage_handler: ElasticsearchStorageHandler,
    stored_chunk_belonging_to_alice: Chunk,
):
    """
    Given that a chunk belonging to a file belonging alice have been saved
    When I call get_file_chunks with the right file id and alice's id
    I Expect the single chunk to be retrieved
    """
    assert stored_chunk_belonging_to_alice.creator_user_uuid

    chunks = elasticsearch_storage_handler.get_file_chunks(
        stored_chunk_belonging_to_alice.parent_file_uuid,
        stored_chunk_belonging_to_alice.creator_user_uuid,
    )

    assert len(chunks) == 1


def test_get_file_chunks_fail(
    elasticsearch_storage_handler: ElasticsearchStorageHandler,
    stored_chunk_belonging_to_alice: Chunk,
):
    """
    Given that a chunk belonging to a file belonging alice have been saved
    When I call get_file_chunks with the right file id and another id
    I Expect the no chunks to be retrieved
    """
    other_chunks = elasticsearch_storage_handler.get_file_chunks(
        stored_chunk_belonging_to_alice.parent_file_uuid,
        uuid4(),
    )
    assert not other_chunks
