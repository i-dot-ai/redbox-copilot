from collections.abc import Iterator
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING

import requests

import tiktoken
from langchain_core.documents import Document

from redbox.models.file import File, ChunkResolution, ChunkMetadata
from redbox.models.settings import Settings
from worker.loader.base import BaseRedboxFileLoader

encoding = tiktoken.get_encoding("cl100k_base")

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


class UnstructuredTitleLoader(BaseRedboxFileLoader):
    """Load, partition and chunk a document using local unstructured library"""

    def __init__(self, file: File, file_bytes: IO[bytes], env: Settings) -> None:
        super().__init__(file, file_bytes, env)

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        url = f"http://{self.host}:8000/general/v0/general"
        files = {
            "files": (self.file.key, self.file_bytes.read().decode()),
        }
        response = requests.post(url, files=files, data={"strategy": "fast"})

        if response.status_code != 200:
            raise ValueError(response.text)

        elements = response.json()

        if not elements:
            raise ValueError("Unstructured failed to extract text for this file")

        for i, raw_chunk in enumerate(elements):
            yield Document(
                page_content=raw_chunk["text"],
                metadata=ChunkMetadata(
                    parent_file_uuid=self.file.uuid,
                    creator_user_uuid=self.file.creator_user_uuid,
                    index=i,
                    file_name=raw_chunk["metadata"].get("filename"),
                    page_number=raw_chunk["metadata"].get("page_number"),
                    created_datetime=datetime.now(UTC),
                    token_count=len(encoding.encode(raw_chunk["text"])),
                    chunk_resolution=ChunkResolution.normal,
                ).model_dump(),
            )
