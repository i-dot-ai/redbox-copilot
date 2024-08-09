from __future__ import annotations

from enum import Enum, StrEnum
from uuid import UUID, uuid4
import datetime

import tiktoken
from pydantic import BaseModel, Field

from redbox.models.base import PersistableModel

encoding = tiktoken.get_encoding("cl100k_base")


class ProcessingStatusEnum(str, Enum):
    """Current status of the file processing.

    Note: The Django app interprets these as:
    "processing" -> "processing"
    "complete" -> "complete"
    "failed" -> "errored"
    anything else -> "processing"

    If you add any other fail state options, the Django app will need to be updated.
    django_app/redbox_app/redbox_core/models.py File.update_status_from_core
    """

    processing = "processing"
    embedding = "embedding"  # Used for processing while we transition to new statuses
    failed = "failed"
    complete = "complete"


class File(PersistableModel):
    """Reference to file stored on s3"""

    key: str = Field(description="file key")
    bucket: str = Field(description="s3 bucket")
    ingest_status: ProcessingStatusEnum | None = Field(
        description="Status of file ingest for files loaded by new worker", default=None
    )


class Link(BaseModel):
    text: str | None
    url: str
    start_index: int

    def __le__(self, other: Link):
        """required for sorted"""
        return self.start_index <= other.start_index

    def __hash__(self):
        return hash(self.text) ^ hash(self.url) ^ hash(self.start_index)


class FileStatus(BaseModel):
    """Status of a file."""

    file_uuid: UUID
    processing_status: ProcessingStatusEnum | None
    chunk_statuses: None = Field(default=None, description="deprecated, see processing_status")


class ChunkResolution(StrEnum):
    smallest = "smallest"
    small = "small"
    normal = "normal"
    large = "large"
    largest = "largest"


class ChunkMetadata(BaseModel):
    """
    Worker model for document metadata for new style chunks.
    This is the minimal metadata that all ingest chains provide and should not be used to map retrieved documents (as fields will be lost)
    """

    uuid: UUID = Field(default_factory=uuid4)
    parent_file_uuid: UUID
    creator_user_uuid: UUID
    index: int
    file_name: str
    page_number: int | None = None
    created_datetime: datetime.datetime = datetime.datetime.now(datetime.UTC)
    token_count: int
    chunk_resolution: ChunkResolution = ChunkResolution.normal
