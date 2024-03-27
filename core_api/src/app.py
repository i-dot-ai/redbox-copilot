import logging
from datetime import datetime
from uuid import UUID

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from faststream.rabbit import RabbitExchange, RabbitQueue
from faststream.rabbit.fastapi import RabbitRouter

from redbox.model_db import SentenceTransformerDB
from redbox.models import (
    Chunk,
    EmbeddingResponse,
    File,
    ModelInfo,
    ProcessingStatusEnum,
    Settings,
    StatusResponse,
)
from redbox.storage import ElasticsearchStorageHandler

# === Logging ===

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

env = Settings()
model_db = SentenceTransformerDB(env.embedding_model)


# === Object Store ===

s3 = env.s3_client()


# === Queues ===

if env.queue != "rabbitmq":
    raise NotImplementedError("SQS is not yet implemented")


router = RabbitRouter(env.rabbit_url)

ingest_channel = RabbitQueue(name=env.ingest_queue_name, durable=True)

publisher = router.publisher(
    queue=ingest_channel,
    exchange=RabbitExchange("redbox-core-exchange", durable=True),
    routing_key=env.ingest_queue_name,
)


# === Storage ===

es = env.elasticsearch_client()


storage_handler = ElasticsearchStorageHandler(es_client=es, root_index="redbox-data")


# === API Setup ===

start_time = datetime.now()


# Create API

app = FastAPI(
    title="Core API",
    description="Redbox Core API",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Health check"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=router.lifespan_context,
)

app.include_router(router)

# === API Routes ===

# Basic Setup


@app.get("/", include_in_schema=False, response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=StatusResponse, tags=["health"])
def health():
    """Returns the health of the API

    Returns:
        StatusResponse: The health of the API
    """

    uptime = datetime.now() - start_time
    uptime_seconds = uptime.total_seconds()

    output = {
        "status": "ready",
        "uptime_seconds": uptime_seconds,
        "version": app.version,
    }

    return output


@app.post("/file", response_model=File, tags=["file"])
async def create_upload_file(name: str, type: str, location: str, ingest=True) -> File:
    """Upload a file to the object store and create a record in the database

    Args:
        name (str): The file name to be recorded
        type (str): The file type to be recorded
        location (str): The presigned file resource location

    Returns:
        File: The file record
    """

    file_record = File(
        name=name,
        path=location,
        type=type,
        creator_user_uuid="dev",
        storage_kind=env.object_store,
        processing_status=ProcessingStatusEnum.uploaded,
    )

    storage_handler.write_item(file_record)

    if ingest:
        await ingest_file(file_record.uuid)

    # TODO: return something more sensible to the user (including errors)
    return file_record


@app.get("/file/{file_uuid}", response_model=File, tags=["file"])
def get_file(file_uuid: UUID) -> File:
    """Get a file from the object store

    Args:
        file_uuid (str): The UUID of the file to get

    Returns:
        File: The file
    """
    return storage_handler.read_item(str(file_uuid), model_type="File")


@app.delete("/file/{file_uuid}", response_model=File, tags=["file"])
def delete_file(file_uuid: str) -> File:
    """Delete a file from the object store and the database

    Args:
        file_uuid (str): The UUID of the file to delete

    Returns:
        File: The file that was deleted
    """
    file = storage_handler.read_item(file_uuid, model_type="File")
    s3.delete_object(Bucket=env.bucket_name, Key=file.name)
    storage_handler.delete_item(file_uuid, model_type="File")
    return file


@app.post("/file/{file_uuid}/ingest", response_model=File, tags=["file"])
async def ingest_file(file_uuid: str) -> File:
    """Trigger the ingest process for a file to a queue.

    Args:
        file_uuid (str): The UUID of the file to ingest

    Returns:
        File: The file that was ingested
    """
    file = storage_handler.read_item(file_uuid, model_type="File")

    file.processing_status = ProcessingStatusEnum.parsing
    storage_handler.update_item(item_uuid=file.uuid, item=file)

    log.info(f"publishing {file.uuid}")
    await publisher.publish(file)

    return file


@app.get("/file/{file_uuid}/chunks", tags=["file"])
def get_file_chunks(file_uuid: UUID) -> list[Chunk]:
    log.info(f"getting chunks for file {file_uuid}")
    return storage_handler.get_file_chunks(file_uuid)


@app.get("/model", tags=["models"])
def get_model() -> ModelInfo:
    """Returns information about the model

    Returns:
        ModelInfo: Information about the model
    """

    return model_db.get_model_info()


@app.post("/embedding", tags=["models"])
def embed_sentences(sentences: list[str]) -> EmbeddingResponse:
    """Embeds a list of sentences using a given model

    Args:
        sentences (list[str]): A list of sentences

    Returns:
        EmbeddingResponse: The embeddings of the sentences
    """

    return model_db.embed_sentences(sentences)
