import logging
from operator import itemgetter
from typing import Annotated

from fastapi import Depends
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    chain,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from core_api.src import dependencies
from core_api.src.format import format_chunks, get_file_chunked_to_tokens
from core_api.src.runnables import make_chat_prompt_from_messages_runnable
from redbox.models import ChatRoute, Chunk, Settings
from redbox.storage import ElasticsearchStorageHandler

# === Logging ===

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def build_vanilla_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    return (
        make_chat_prompt_from_messages_runnable(env.ai.vanilla_system_prompt, env.ai.vanilla_question_prompt)
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.vanilla.value),
        }
    )


def build_retrieval_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    retriever: Annotated[VectorStoreRetriever, Depends(dependencies.get_es_retriever)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    return (
        RunnablePassthrough.assign(documents=retriever)
        | RunnablePassthrough.assign(
            formatted_documents=(RunnablePassthrough() | itemgetter("documents") | format_chunks)
        )
        | {
            "response": make_chat_prompt_from_messages_runnable(
                env.ai.retrieval_system_prompt, env.ai.retrieval_question_prompt
            )
            | llm
            | StrOutputParser(),
            "source_documents": itemgetter("documents"),
            "route_name": RunnableLambda(lambda _: ChatRoute.search.value),
        }
    )


def build_summary_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    storage_handler: Annotated[ElasticsearchStorageHandler, Depends(dependencies.get_storage_handler)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def make_document_context(input_dict: dict):
        documents: list[Chunk] = []
        for selected_file in input_dict["file_uuids"]:
            chunks = get_file_chunked_to_tokens(
                file_uuid=selected_file,
                user_uuid=input_dict["user_uuid"],
                storage_handler=storage_handler,
                max_tokens=env.ai.summarisation_chunk_max_tokens,
            )
            documents += chunks

        input_dict["documents"] = documents
        return documents

    # Stuff chain now missing the RunnabeLambda to format the chunks
    stuff_chain = (
        make_chat_prompt_from_messages_runnable(
            env.ai.summarisation_system_prompt, env.ai.summarisation_question_prompt
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.stuff_summary.value),
        }
    )

    @chain
    def map_operation(input_dict):
        system_map_prompt = env.ai.map_system_prompt
        prompt_template = PromptTemplate.from_template(env.ai.map_question_prompt)

        formatted_map_question_prompt = prompt_template.format(question=input_dict["question"])

        map_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_map_prompt),
                ("human", formatted_map_question_prompt + env.ai.map_document_prompt),
            ]
        )

        documents = [chunk.text for chunk in input_dict["documents"]]

        map_summaries = (map_prompt | llm | StrOutputParser()).batch(
            documents,
            config=RunnableConfig(max_concurrency=env.ai.summarisation_max_concurrency),
        )

        summaries = " ; ".join(map_summaries)
        input_dict["summaries"] = summaries
        return input_dict

    map_reduce_chain = (
        map_operation
        | make_chat_prompt_from_messages_runnable(env.ai.reduce_system_prompt, env.ai.reduce_question_prompt)
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.map_reduce_summary.value),
        }
    )

    @chain
    def summarisation_route(input_dict):
        documents = [chunk.text for chunk in input_dict["documents"]]

        if len(documents) == 1:
            return stuff_chain

        elif len(documents) > 1:
            return map_reduce_chain

        else:
            msg = "No documents to summarise"
            raise ValueError(msg)

    return RunnablePassthrough.assign(documents=make_document_context) | summarisation_route


def build_static_response_chain(prompt_template, route_name) -> Runnable:
    return RunnablePassthrough.assign(
        response=(ChatPromptTemplate.from_template(prompt_template) | RunnableLambda(lambda p: p.messages[0].content)),
        source_documents=RunnableLambda(lambda _: []),
        route_name=RunnableLambda(lambda _: route_name.value),
    )
