from operator import itemgetter
from typing import Any, TypedDict
from uuid import UUID

from elasticsearch import Elasticsearch
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_elasticsearch import ElasticsearchRetriever

from core_api.src.format import format_chunks
from redbox.models import Chunk


def make_chat_runnable(
    system_prompt: str,
    llm: ChatLiteLLM,
) -> Runnable:
    """Takes a system prompt and LLM returns a chat runnable.

    Runnable takes input of a dict keyed to question and messages.

    Runnable returns a string.
    """
    chat_history = [
        ("system", system_prompt),
        ("placeholder", "{messages}"),
        ("user", "{question}"),
    ]

    return (
        {
            "question": itemgetter("question"),
            "messages": itemgetter("messages"),
        }
        | ChatPromptTemplate.from_messages(chat_history)
        | llm
        | StrOutputParser()
    )


def make_stuff_document_runnable(
    system_prompt: str,
    llm: ChatLiteLLM,
) -> Runnable:
    """Takes a system prompt and LLM returns a stuff document runnable.

    Runnable takes input of a dict keyed to question, messages and documents.

    Runnable returns a string.
    """
    chat_history = [
        ("system", system_prompt),
        ("placeholder", "{messages}"),
        ("user", "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "),
    ]

    return (
        {
            "question": itemgetter("question"),
            "messages": itemgetter("messages"),
            "documents": itemgetter("documents") | RunnableLambda(format_chunks),
        }
        | ChatPromptTemplate.from_messages(chat_history)
        | llm
        | StrOutputParser()
    )


class ESQuery(TypedDict):
    question: str
    file_uuids: list[UUID]
    user_uuid: UUID


def make_es_retriever(
    es: Elasticsearch, embedding_model: SentenceTransformerEmbeddings, chunk_index_name: str, k: int = 5
) -> ElasticsearchRetriever:
    """Creates an Elasticsearch retriever runnable.

    Runnable takes input of a dict keyed to question, file_uuids and user_uuid.

    Runnable returns a list of Chunks.
    """

    def es_query(query: ESQuery) -> dict[str, Any]:
        vector = embedding_model.embed_query(query["question"])

        knn_filter = [{"term": {"creator_user_uuid.keyword": str(query["user_uuid"])}}]

        if len(query["file_uuids"]) != 0:
            knn_filter.append({"terms": {"parent_file_uuid.keyword": [str(uuid) for uuid in query["file_uuids"]]}})

        return {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": vector,
                                "num_candidates": k,
                                "filter": knn_filter,
                            }
                        }
                    ]
                }
            },
        }

    def chunk_mapper(hit: dict[str, Any]) -> Chunk:
        return Chunk(**hit["_source"])

    return ElasticsearchRetriever(
        es_client=es, index_name=chunk_index_name, body_func=es_query, document_mapper=chunk_mapper
    )


def make_rag_runnable(
    system_prompt: str,
    llm: ChatLiteLLM,
    retriever: VectorStoreRetriever,
) -> Runnable:
    """Takes a system prompt, LLM and retriever and returns a basic RAG runnable.

    Runnable takes input of a dict keyed to question, messages and file_uuids and user_uuid.

    Runnable returns a dict keyed to response and sources.
    """
    chat_history = [
        ("system", system_prompt),
        ("placeholder", "{messages}"),
        ("user", "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "),
    ]

    prompt = ChatPromptTemplate.from_messages(chat_history)

    return (
        RunnablePassthrough()
        | {
            "question": itemgetter("question"),
            "messages": itemgetter("messages"),
            "documents": retriever | format_chunks,
            "sources": retriever,
        }
        | {
            "response": prompt | llm | StrOutputParser(),
            "sources": itemgetter("sources"),
        }
    )


def make_condense_question_runnable(llm: ChatLiteLLM) -> Runnable:
    """Takes a system prompt and LLM returns a condense question runnable.

    Runnable takes input of a dict keyed to question and messages.

    Runnable returns a string.
    """
    condense_prompt = (
        "Given the following conversation and a follow up question, "
        "rephrase the follow up question to be a standalone question. \n"
        "Chat history:"
    )

    chat_history = [
        ("system", condense_prompt),
        ("placeholder", "{messages}"),
        ("user", "Follow up question: {question}. \nStandalone question: "),
    ]

    return (
        {
            "question": itemgetter("question"),
            "messages": itemgetter("messages"),
        }
        | ChatPromptTemplate.from_messages(chat_history)
        | llm
        | StrOutputParser()
    )


def make_condense_rag_runnable(
    system_prompt: str,
    llm: ChatLiteLLM,
    retriever: VectorStoreRetriever,
) -> Runnable:
    """Takes a system prompt, LLM and retriever and returns a condense RAG runnable.

    This attempts to condense the chat history into a more salient question for the
    LLM to answer, and doesn't pass the entire history on to RAG -- just the condensed
    question.

    Runnable takes input of a dict keyed to question, messages and file_uuids and user_uuid.

    Runnable returns a dict keyed to response and sources.
    """
    chat_history = [
        ("system", system_prompt),
        ("user", "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "),
    ]

    prompt = ChatPromptTemplate.from_messages(chat_history)

    condense_question_runnable = make_condense_question_runnable(llm=llm)

    condense_question_chain = {
        "question": itemgetter("question"),
        "messages": itemgetter("messages"),
    } | condense_question_runnable

    return (
        RunnablePassthrough()
        | {
            "question": condense_question_chain,
            "documents": retriever | format_chunks,
            "sources": retriever,
        }
        | {
            "response": prompt | llm | StrOutputParser(),
            "sources": itemgetter("sources"),
        }
    )
