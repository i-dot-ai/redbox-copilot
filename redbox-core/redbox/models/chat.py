from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    text: str = Field(description="The text of the message")
    role: Literal["user", "ai", "system"] = Field(description="The role of the message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "You are helpful AI Assistant", "role": "system"},
                {"text": "Hello", "role": "user"},
                {"text": "Hi there!", "role": "ai"},
            ]
        }
    }


class SelectedDocument(BaseModel):
    uuid: UUID | None = Field(description="uuid of selected file", default=None)


class ChatRequest(BaseModel):
    message_history: list[ChatMessage] = Field(description="The history of messages in the chat")
    selected_files: list[SelectedDocument] = Field(
        description="Documents selected to use for the current chat request",
        default_factory=list,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_history": [
                        {"text": "You are a helpful AI Assistant", "role": "system"},
                        {"text": "What is AI?", "role": "user"},
                    ],
                    "selected_files": [
                        {"uuid": "9aa1aa15-dde0-471f-ab27-fd410612025b"},
                        {"uuid": "219c2e94-9877-4f83-ad6a-a59426f90171"},
                    ],
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    page_content: str = Field(description="chunk text")
    file_uuid: UUID | None = Field(description="uuid of original file", default=None)
    page_numbers: list[int] | None = Field(
        description="page number of the file that this chunk came from", default=None
    )


class SourceDocuments(BaseModel):
    source_documents: list[SourceDocument] | None = Field(
        description="documents retrieved to form this response", default=None
    )


class ChatRoute(StrEnum):
    info = "info"
    ability = "ability"
    coach = "coach"
    gratitude = "gratitude"
    search = "search"
    summarise = "summarise"
    map_reduce_summarise = "summarise/documents/large"
    chat = "chat"
    chat_with_docs = "chat/documents"
    chat_with_docs_map_reduce = "chat/documents/large"


class ChatResponse(BaseModel):
    source_documents: list[SourceDocument] | None = Field(
        description="documents retrieved to form this response", default=None
    )
    output_text: str = Field(
        description="response text",
        examples=["The current Prime Minister of the UK is The Rt Hon. Rishi Sunak MP."],
    )
    route_name: ChatRoute = Field(description="the conversation route taken")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ClientResponse(BaseModel):
    # Needs to match CoreChatResponse in django-app/redbox_app/redbox_core/consumers.py
    resource_type: Literal["text", "documents", "route_name", "end", "error"]
    data: list[SourceDocument] | str | ErrorDetail | None = None
