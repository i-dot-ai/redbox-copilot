import logging
import sys
from operator import itemgetter
from typing import Annotated

from core_api import dependencies
from fastapi import Depends
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, chain
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from tiktoken import Encoding

from redbox.api.format import format_documents
from redbox.api.runnables import filter_by_elbow, make_chat_prompt_from_messages_runnable, resize_documents
from redbox.models import ChatRoute, Settings
from redbox.models.errors import NoDocumentSelected

# === Logging ===

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()


def build_chat_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    return (
        make_chat_prompt_from_messages_runnable(
            system_prompt=env.ai.chat_system_prompt,
            question_prompt=env.ai.chat_question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.chat.value),
        }
    )


def build_chat_with_docs_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    all_chunks_retriever: Annotated[BaseRetriever, Depends(dependencies.get_all_chunks_retriever)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def make_document_context():
        return all_chunks_retriever | resize_documents(env.ai.stuff_chunk_max_tokens)

    @chain
    def map_operation(input_dict):
        system_map_prompt = env.ai.map_system_prompt
        prompt_template = PromptTemplate.from_template(env.ai.chat_map_question_prompt)

        formatted_map_question_prompt = prompt_template.format(question=input_dict["question"])

        map_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_map_prompt),
                ("human", formatted_map_question_prompt + env.ai.map_document_prompt),
            ]
        )

        documents = input_dict["documents"]

        map_summaries = (map_prompt | llm | StrOutputParser()).batch(
            documents,
            config=RunnableConfig(max_concurrency=env.ai.map_max_concurrency),
        )

        summaries = " ; ".join(map_summaries)
        input_dict["summaries"] = summaries
        return input_dict

    @chain
    def chat_with_docs_route(input_dict: dict):
        log.info("Length documents: %s", len(input_dict["documents"]))
        if len(input_dict["documents"]) == 1:
            return RunnablePassthrough.assign(
                formatted_documents=(itemgetter("documents") | format_documents)
            ) | {
                "response": make_chat_prompt_from_messages_runnable(
                    system_prompt=env.ai.chat_with_docs_system_prompt,
                    question_prompt=env.ai.chat_with_docs_question_prompt,
                    input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                    tokeniser=tokeniser,
                )
                | llm
                | StrOutputParser(),
                "route_name": RunnableLambda(lambda _: ChatRoute.chat_with_docs.value),
            }

        elif len(input_dict["documents"]) > 1:
            return (
                map_operation
                | RunnablePassthrough.assign(
                    formatted_documents=(itemgetter("documents") | format_documents)
                )
                | {
                    "response": make_chat_prompt_from_messages_runnable(
                        system_prompt=env.ai.chat_with_docs_reduce_system_prompt,
                        question_prompt=env.ai.chat_with_docs_reduce_question_prompt,
                        input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                        tokeniser=tokeniser,
                    )
                    | llm
                    | StrOutputParser(),
                    "route_name": RunnableLambda(lambda _: ChatRoute.chat_with_docs.value),
                }
            )

        else:
            raise NoDocumentSelected

    return RunnablePassthrough.assign(documents=make_document_context()) | chat_with_docs_route


def build_retrieval_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    retriever: Annotated[VectorStoreRetriever, Depends(dependencies.get_parameterised_retriever)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    return (
        RunnablePassthrough.assign(documents=retriever)
        | RunnablePassthrough.assign(
            formatted_documents=(itemgetter("documents") | format_documents)
        )
        | {
            "response": make_chat_prompt_from_messages_runnable(
                system_prompt=env.ai.retrieval_system_prompt,
                question_prompt=env.ai.retrieval_question_prompt,
                input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                tokeniser=tokeniser,
            )
            | llm
            | StrOutputParser(),
            "source_documents": itemgetter("documents"),
            "route_name": RunnableLambda(lambda _: ChatRoute.search.value),
        }
    )


def build_condense_retrieval_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    retriever: Annotated[VectorStoreRetriever, Depends(dependencies.get_parameterised_retriever)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def route(input_dict: dict):
        if len(input_dict["chat_history"]) > 0:
            return RunnablePassthrough.assign(
                question=make_chat_prompt_from_messages_runnable(
                    system_prompt=env.ai.condense_system_prompt,
                    question_prompt=env.ai.condense_question_prompt,
                    input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                    tokeniser=tokeniser,
                )
                | llm
                | StrOutputParser()
            )
        else:
            return RunnablePassthrough()

    return (
        RunnableLambda(route)
        | RunnablePassthrough.assign(documents=retriever | filter_by_elbow(enabled=env.ai.elbow_filter_enabled))
        | RunnablePassthrough.assign(
            formatted_documents=(itemgetter("documents") | format_documents)
        )
        | {
            "response": make_chat_prompt_from_messages_runnable(
                system_prompt=env.ai.retrieval_system_prompt,
                question_prompt=env.ai.retrieval_question_prompt,
                input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
                tokeniser=tokeniser,
            )
            | llm
            | StrOutputParser(),
            "source_documents": itemgetter("documents"),
            "route_name": RunnableLambda(lambda _: ChatRoute.search.value),
        }
    )


def build_summary_chain(
    llm: Annotated[ChatLiteLLM, Depends(dependencies.get_llm)],
    all_chunks_retriever: Annotated[BaseRetriever, Depends(dependencies.get_all_chunks_retriever)],
    tokeniser: Annotated[Encoding, Depends(dependencies.get_tokeniser)],
    env: Annotated[Settings, Depends(dependencies.get_env)],
) -> Runnable:
    def make_document_context():
        return (
            all_chunks_retriever
            | resize_documents(env.ai.stuff_chunk_max_tokens)
            | RunnableLambda(lambda docs: [d.page_content for d in docs])
        )

    # Stuff chain now missing the RunnabeLambda to format the chunks
    stuff_chain = (
        make_chat_prompt_from_messages_runnable(
            system_prompt=env.ai.summarisation_system_prompt,
            question_prompt=env.ai.summarisation_question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.summarise.value),
        }
    )

    @chain
    def map_operation(input_dict):
        system_map_prompt = env.ai.map_system_prompt
        prompt_template = PromptTemplate.from_template(env.ai.chat_map_question_prompt)

        formatted_map_question_prompt = prompt_template.format(question=input_dict["question"])

        map_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_map_prompt),
                ("human", formatted_map_question_prompt + env.ai.chat_map_question_prompt),
            ]
        )

        documents = input_dict["documents"]

        map_summaries = (map_prompt | llm | StrOutputParser()).batch(
            documents,
            config=RunnableConfig(max_concurrency=env.ai.map_max_concurrency),
        )

        summaries = " ; ".join(map_summaries)
        input_dict["summaries"] = summaries
        return input_dict

    map_reduce_chain = (
        map_operation
        | make_chat_prompt_from_messages_runnable(
            system_prompt=env.ai.reduce_system_prompt,
            question_prompt=env.ai.reduce_question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | {
            "response": StrOutputParser(),
            "route_name": RunnableLambda(lambda _: ChatRoute.map_reduce_summarise.value),
        }
    )

    @chain
    def summarisation_route(input_dict):
        if len(input_dict["documents"]) == 1:
            return stuff_chain

        elif len(input_dict["documents"]) > 1:
            return map_reduce_chain

        else:
            raise NoDocumentSelected

    return RunnablePassthrough.assign(documents=make_document_context()) | summarisation_route


def build_static_response_chain(prompt_template, route_name) -> Runnable:
    return RunnablePassthrough.assign(
        response=(ChatPromptTemplate.from_template(prompt_template) | RunnableLambda(lambda p: p.messages[0].content)),
        source_documents=RunnableLambda(lambda _: []),
        route_name=RunnableLambda(lambda _: route_name.value),
    )
