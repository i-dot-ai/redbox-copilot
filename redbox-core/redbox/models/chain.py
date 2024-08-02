"""
There is some repeated definition and non-pydantic style code in here.
These classes are pydantic v1 which is compatible with langchain tools classes, we need
to provide a pydantic v1 definition to work with these. As these models are mostly
used in conjunction with langchain this is the tidiest boxing of pydantic v1 we can do
"""

from typing import TypedDict, Literal, Annotated, Required, NotRequired
from uuid import UUID
from operator import add

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document


class ChainChatMessage(TypedDict):
    role: Literal["user", "ai", "system"]
    text: str


CHAT_SYSTEM_PROMPT = (
    "You are an AI assistant called Redbox tasked with answering questions and providing information objectively."
)

CHAT_WITH_DOCS_SYSTEM_PROMPT = "You are an AI assistant called Redbox tasked with answering questions on user provided documents and providing information objectively."

CHAT_WITH_DOCS_REDUCE_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with answering questions on user provided documents. "
    "Your goal is to answer the user question based on list of summaries in a coherent manner."
    "Please follow these guidelines while answering the question: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the answer is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

RETRIEVAL_SYSTEM_PROMPT = (
    "Given the following conversation and extracted parts of a long document and a question, create a final answer. \n"
    "If you don't know the answer, just say that you don't know. Don't try to make up an answer. "
    "If a user asks for a particular format to be returned, such as bullet points, then please use that format. "
    "If a user asks for bullet points you MUST give bullet points. "
    "If the user asks for a specific number or range of bullet points you MUST give that number of bullet points. \n"
    "Use **bold** to highlight the most question relevant parts in your response. "
    "If dealing dealing with lots of data return it in markdown table format. "
)

SUMMARISATION_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to extract the most important information and present it in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

MAP_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to extract the most important information and present it in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

REDUCE_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to write a concise summary of list of summaries from a list of summaries in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

CONDENSE_SYSTEM_PROMPT = (
    "Given the following conversation and a follow up question, generate a follow "
    "up question to be a standalone question. "
    "You are only allowed to generate one question in response. "
    "Include sources from the chat history in the standalone question created, "
    "when they are available. "
    "If you don't know the answer, just say that you don't know, "
    "don't try to make up an answer. \n"
)

CHAT_QUESTION_PROMPT = "{question}\n=========\n Response: "

CHAT_WITH_DOCS_QUESTION_PROMPT = "Question: {question}. \n\n Documents: \n\n {formatted_documents} \n\n Answer: "

CHAT_WITH_DOCS_REDUCE_QUESTION_PROMPT = "Question: {question}. \n\n Documents: \n\n {summaries} \n\n Answer: "

RETRIEVAL_QUESTION_PROMPT = "{question} \n=========\n{formatted_documents}\n=========\nFINAL ANSWER: "

SUMMARISATION_QUESTION_PROMPT = "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "

CHAT_MAP_QUESTION_PROMPT = "Question: {question}. \n Documents: \n {formatted_documents} \n\n Answer: "


REDUCE_QUESTION_PROMPT = "Question: {question}. \n\n Documents: \n\n {formatted_documents} \n\n Answer: "

CONDENSE_QUESTION_PROMPT = "{question}\n=========\n Standalone question: "


class AISettings(BaseModel):
    """prompts and other AI settings"""

    context_window_size: int = 8_000
    rag_k: int = 30
    rag_num_candidates: int = 10
    rag_desired_chunk_size: int = 300
    elbow_filter_enabled: bool = False
    chat_system_prompt: str = CHAT_SYSTEM_PROMPT
    chat_question_prompt: str = CHAT_QUESTION_PROMPT
    stuff_chunk_context_ratio: float = 0.75
    chat_with_docs_system_prompt: str = CHAT_WITH_DOCS_SYSTEM_PROMPT
    chat_with_docs_question_prompt: str = CHAT_WITH_DOCS_QUESTION_PROMPT
    chat_with_docs_reduce_system_prompt: str = CHAT_WITH_DOCS_REDUCE_SYSTEM_PROMPT
    chat_with_docs_reduce_question_prompt: str = CHAT_WITH_DOCS_REDUCE_QUESTION_PROMPT
    retrieval_system_prompt: str = RETRIEVAL_SYSTEM_PROMPT
    retrieval_question_prompt: str = RETRIEVAL_QUESTION_PROMPT
    condense_system_prompt: str = CONDENSE_SYSTEM_PROMPT
    condense_question_prompt: str = CONDENSE_QUESTION_PROMPT
    summarisation_system_prompt: str = SUMMARISATION_SYSTEM_PROMPT
    summarisation_question_prompt: str = SUMMARISATION_QUESTION_PROMPT
    map_max_concurrency: int = 128
    map_system_prompt: str = MAP_SYSTEM_PROMPT
    chat_map_question_prompt: str = CHAT_MAP_QUESTION_PROMPT
    reduce_system_prompt: str = REDUCE_SYSTEM_PROMPT
    reduce_question_prompt: str = REDUCE_QUESTION_PROMPT

    match_boost: int = 1
    knn_boost: int = 1
    similarity_threshold: int = 0

    chat_backend: Literal["azure", "openai", "fake"] = "azure"
    fake_backend_responses: list = ["hello"]
    llm_max_tokens: int = 1024

    openai_api_key: str = "NotAKey"  # also in settings for embeddings
    openai_api_version: str = "2023-12-01-preview"

    azure_openai_api_key: str = "NotAKey"  # also in settings for embeddings
    azure_openai_model: str = "azure/gpt-35-turbo-16k"
    azure_openai_endpoint: str | None = None  # also in settings for embeddings

    @property
    def stuff_chunk_max_tokens(self) -> int:
        return int(self.context_window_size * self.stuff_chunk_context_ratio)


class ChainInput(BaseModel):
    question: str = Field(description="The last user chat message")
    file_uuids: list[UUID] = Field(description="List of files to process")
    user_uuid: UUID = Field(description="User the chain in executing for")
    chat_history: list[ChainChatMessage] = Field(description="All previous messages in chat (excluding question)")
    ai_settings: AISettings = Field(description="User request AI settings", default_factory=AISettings)


class ChainState(TypedDict):
    query: Required[ChainInput]
    documents: NotRequired[list[Document]]
    response: NotRequired[str | None]
    route_name: NotRequired[str | None]
    prompt_args: NotRequired[dict[str, str]]


class ChatMapReduceState(ChainState):
    intermediate_docs: Annotated[NotRequired[list[Document]], add]
