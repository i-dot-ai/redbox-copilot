import pytest
from moto import mock_aws

from ingester.src.worker import broker, ingest_channel, app
from redbox.models import ProcessingStatusEnum
from redbox.storage import ElasticsearchStorageHandler

from faststream.rabbit import TestRabbitBroker, TestApp


@mock_aws
@pytest.mark.asyncio
async def test_ingest_file(s3_client, es_client, embedding_model, file):
    """
    Given that I have written a text File to s3
    When I call ingest_file
    I Expect to see this file to be:
    1. chunked
    2. written to Elasticsearch
    """

    storage_handler = ElasticsearchStorageHandler(es_client=es_client, root_index="redbox-data")

    async with TestRabbitBroker(broker) as br, TestApp(app):
        await br.publish(file, queue=ingest_channel)

        assert (
            storage_handler.read_item(
                item_uuid=file.uuid,
                model_type="file",
            ).processing_status
            is ProcessingStatusEnum.chunking
        )
