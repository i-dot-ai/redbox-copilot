from typing import TYPE_CHECKING
import logging
from io import BytesIO
from functools import partial

from langchain.vectorstores import VectorStore
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnableLambda, chain, Runnable

from redbox.models.settings import Settings
from redbox.models.file import File
from redbox.loader.base import BaseRedboxFileLoader


if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


@chain
def log_chunks(chunks: list[Document]):
    log.info("Processing %s chunks", len(chunks))
    return chunks


def document_loader(document_loader_type: type[BaseRedboxFileLoader], s3_client: S3Client, env: Settings) -> Runnable:
    @chain
    def wrapped(file: File):
        file_bytes = s3_client.get_object(Bucket=file.bucket, Key=file.key)["Body"].read()
        return document_loader_type(file=file, file_bytes=BytesIO(file_bytes), env=env).lazy_load()

    return wrapped


def ingest_from_loader(
    document_loader_type: type[BaseRedboxFileLoader], s3_client: S3Client, vectorstore: VectorStore, env: Settings
) -> Runnable:
    return (
        document_loader(document_loader_type=document_loader_type, s3_client=s3_client, env=env)
        | RunnableLambda(list)
        | log_chunks
        | RunnableLambda(partial(vectorstore.aadd_documents, create_index_if_not_exists=False))  # type: ignore[arg-type]
    )
