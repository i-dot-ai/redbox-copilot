import logging
from http import HTTPStatus
from http.client import HTTPException
from typing import Any
from uuid import UUID

import numpy as np
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_elasticsearch import ElasticsearchStore

from core_api.src.format import get_file_chunked_to_tokens
from core_api.src.runnables import make_es_retriever, make_rag_runnable, make_stuff_document_runnable
from redbox.llm.prompts.chat import CONDENSE_QUESTION_PROMPT, STUFF_DOCUMENT_PROMPT, WITH_SOURCES_PROMPT
from redbox.models import ChatRequest, Chunk
from redbox.storage import ElasticsearchStorageHandler

# === Logging ===

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# Define the system prompt for summarisation
summarisation_prompt = (
    "You are an AI assistant tasked with summarizing documents. "
    "Your goal is to extract the most important information and present it in "
    "a concise and coherent manner. Please follow these guidelines while summarizing: \n"
    "1) Identify and highlight key points,\n"
    "2) Avoid repetition,\n"
    "3) Ensure the summary is easy to understand,\n"
    "4) Maintain the original context and meaning.\n"
)

# Define the system prompt for RAG
rag_prompt = (
    "You are Redbox. "
    "An AI focused on helping UK Civil Servants, Political Advisors "
    "and Ministers triage and summarise information from a wide variety of sources. "
    "You are impartial and non-partisan. "
    "You are not a replacement for human judgement, but you can help humans "
    "make more informed decisions. "
    "If you are asked a question you cannot answer based on your following "
    "instructions, you should say so. "
    "Be concise and professional in your responses. "
    "Respond in markdown format. \n\n"
    "=== RULES === \n\n"
    "All responses to tasks **MUST** be in British English. "
    "This is so that the user can understand your responses. \n\n"
    "Given the following extracted parts of a long document and a question, "
    "create a final answer. "
    "If you don't know the answer, just say that you don't know. "
    "Don't try to make up an answer. "
    "If a user asks for a particular format to be returned, such as bullet points, "
    "then please use that format. "
    "If a user asks for bullet points you MUST give bullet points. "
    "If the user asks for a specific number or range of bullet points "
    "you MUST give that number of bullet points. "
    "For example: \n"
    "QUESTION: Please give me 6-8 bullet points on tigers \n"
    "FINAL ANSWER: "
    "- Tigers are orange. \n"
    "- Tigers are big. \n"
    "- Tigers are scary. \n"
    "- Tigers are cool. \n"
    "- Tigers are cats. \n"
    "- Tigers are animals. \n\n"
    "If the number of bullet points a user asks for is not supported by the "
    "amount of information that you have, then say so, else give what the user "
    "asks for. \n\n"
    "At the end of your response do not add a 'Sources:' section with the documents "
    "you used. DO NOT NAME CITED DOCUMENTS IN YOUR RESPONSE. \n"
    "Use **bold** to highlight the most question relevant parts in your response. "
    "If dealing dealing with lots of data return it in markdown table format. "
)


async def build_vanilla_chain(
    chat_request: ChatRequest,
    **kwargs,  # noqa: ARG001
) -> ChatPromptTemplate:
    """Get a LLM response to a question history"""

    if len(chat_request.message_history) < 2:  # noqa: PLR2004
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="Chat history should include both system and user prompts",
        )

    if chat_request.message_history[0].role != "system":
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="The first entry in the chat history should be a system prompt",
        )

    if chat_request.message_history[-1].role != "user":
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="The final entry in the chat history should be a user question",
        )

    return ChatPromptTemplate.from_messages((msg.role, msg.text) for msg in chat_request.message_history)


async def build_retrieval_chain(
    chat_request: ChatRequest,
    user_uuid: UUID,
    llm: ChatLiteLLM,
    vector_store: ElasticsearchStore,
) -> tuple[Runnable, dict[str, Any]]:
    question = chat_request.message_history[-1].text
    previous_history = list(chat_request.message_history[:-1])
    previous_history = ChatPromptTemplate.from_messages(
        (msg.role, msg.text) for msg in previous_history
    ).format_messages()

    docs_with_sources_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=WITH_SOURCES_PROMPT,
        document_prompt=STUFF_DOCUMENT_PROMPT,
        verbose=True,
    )

    condense_question_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    standalone_question = condense_question_chain({"question": question, "chat_history": previous_history})["text"]

    search_kwargs = {"filter": {"bool": {"must": [{"term": {"creator_user_uuid.keyword": str(user_uuid)}}]}}}

    if chat_request.selected_files:
        logging.info("chat_request.selected_files: %s", str(chat_request.selected_files))
        search_kwargs["filter"]["bool"]["must"] = [
            {"term": {"parent_file_uuid.keyword": str(file.uuid)}} for file in chat_request.selected_files
        ]

    docs = vector_store.as_retriever(search_kwargs=search_kwargs).get_relevant_documents(standalone_question)

    params = {
        "question": standalone_question,
        "input_documents": docs,
    }

    return docs_with_sources_chain, params


async def build_summary_chain(
    chat_request: ChatRequest,
    user_uuid: UUID,
    llm: ChatLiteLLM,
    storage_handler: ElasticsearchStorageHandler,
    **kwargs,  # noqa: ARG001
) -> tuple[Runnable, dict[str, Any]]:
    question = chat_request.message_history[-1].text
    previous_history = list(chat_request.message_history[:-1])

    chain = make_stuff_document_runnable(system_prompt=summarisation_prompt, llm=llm)

    documents: list[Chunk] = []
    for selected_file in chat_request.selected_files:
        chunks = get_file_chunked_to_tokens(
            file_uuid=selected_file.uuid,
            user_uuid=user_uuid,
            storage_handler=storage_handler,
        )
        documents += chunks

    # right now, can only handle a single document so we manually truncate
    max_tokens = 20_000  # parameterise later
    doc_token_sum = np.cumsum([doc.token_count for doc in documents])
    doc_token_sum_limit_index = len([i for i in doc_token_sum if i < max_tokens])

    documents_trunc = documents[:doc_token_sum_limit_index]
    if len(documents) < doc_token_sum_limit_index:
        log.info("Documents were longer than 20k tokens. Truncating to the first 20k.")

    params = {
        "question": question,
        "documents": documents_trunc,
        "messages": [(msg.role, msg.text) for msg in previous_history],
    }

    return chain, params


async def build_k_retrieval_chain(
    chat_request: ChatRequest,
    user_uuid: UUID,
    llm: ChatLiteLLM,
    embedding_model: Embeddings,
    storage_handler: ElasticsearchStorageHandler,
    k: int,
    **kwargs,  # noqa: ARG001
) -> tuple[Runnable, dict[str, Any]]:
    question = chat_request.message_history[-1].text
    previous_history = list(chat_request.message_history[:-1])

    retriever = make_es_retriever(
        es=storage_handler.es_client,
        embedding_model=embedding_model,
        chunk_index_name=f"{storage_handler.root_index}-chunk",
        k=k,
    )

    chain = make_rag_runnable(system_prompt=rag_prompt, llm=llm, retriever=retriever)

    params = {
        "question": question,
        "file_uuids": [file.uuid for file in chat_request.selected_files],
        "user_uuid": user_uuid,
        "messages": [(msg.role, msg.text) for msg in previous_history],
    }

    return chain, params
