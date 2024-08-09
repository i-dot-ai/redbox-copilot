import logging
import re
from typing import Any, Callable
from uuid import uuid4
from functools import reduce

from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from redbox.chains.components import get_tokeniser
from redbox.chains.runnables import build_chat_prompt_from_messages_runnable
from redbox.models import ChatRoute
from redbox.models.chain import RedboxState
from redbox.transform import combine_documents, structure_documents
from redbox.models.chain import PromptSet
from redbox.transform import flatten_document_state


log = logging.getLogger()
re_keyword_pattern = re.compile(r"@(\w+)")


# Patterns: functions that build processes

## Core patterns


def build_retrieve_pattern(retriever: VectorStoreRetriever) -> Callable[[RedboxState], dict[str, Any]]:
    """Returns a function that uses state["request"] and state["text"] to set state["documents"]."""

    def _retrieve(state: RedboxState) -> dict[str, Any]:
        return RunnableParallel({"documents": retriever | structure_documents}).invoke(state)

    return _retrieve


def build_chat_pattern(
    llm: BaseChatModel,
    prompt_set: PromptSet,
    final_response_chain: bool = False,
) -> Callable[[RedboxState], dict[str, Any]]:
    """Returns a function that uses state["request"] to set state["text"]."""
    _llm = llm.with_config(tags=["response_flag"]) if final_response_chain else llm

    def _chat(state: RedboxState) -> dict[str, Any]:
        chat_chain = build_chat_prompt_from_messages_runnable(prompt_set) | _llm | {"text": StrOutputParser()}

        return chat_chain.invoke(state)

    return _chat


def build_merge_pattern(
    llm: BaseChatModel,
    prompt_set: PromptSet,
    final_response_chain: bool = False,
) -> Callable[[RedboxState], dict[str, Any]]:
    """Returns a function that uses state["request"] and state["documents"] to return one item in state["documents"].

    When combined with chunk send, will replace each Document with what's returned from the LLM.

    When combined with group send, with combine all Documents and use the metadata of the first.

    When used without a send, the first Document receieved defines the metadata.
    """
    _llm = llm.with_config(tags=["response_flag"]) if final_response_chain else llm
    tokeniser = get_tokeniser()

    def _merge(state: RedboxState) -> dict[str, Any]:
        flattened_documents = flatten_document_state(state["documents"])

        merged_document = reduce(lambda left, right: combine_documents(left, right), flattened_documents)

        merge_chain = build_chat_prompt_from_messages_runnable(prompt_set) | _llm | StrOutputParser()

        merge_state = RedboxState(
            request=state["request"],
            documents={
                merged_document.metadata["parent_file_uuid"]: {merged_document.metadata["uuid"]: merged_document}
            },
        )

        merged_document.page_content = merge_chain.invoke(merge_state)
        merged_document.metadata["token_count"] = len(tokeniser.encode(merged_document.page_content))

        group_uuid = merged_document.metadata.get("parent_file_uuid", uuid4())
        document_uuid = merged_document.metadata.get("uuid", uuid4())

        # Clear old documents, add new one
        document_state = state["documents"].copy()

        for group in document_state:
            for document in document_state[group]:
                document_state[group][document] = None

        document_state[group_uuid][document_uuid] = merged_document

        return {"documents": document_state}

    return _merge


def build_stuff_pattern(
    llm: BaseChatModel,
    prompt_set: PromptSet,
    final_response_chain: bool = False,
) -> Callable[[RedboxState], dict[str, Any]]:
    """Returns a function that uses state["request"] and state["documents"] to set state["text"]."""
    _llm = llm.with_config(tags=["response_flag"]) if final_response_chain else llm

    def _stuff(state: RedboxState) -> dict[str, Any]:
        stuff_chain = build_chat_prompt_from_messages_runnable(prompt_set) | _llm | StrOutputParser()
        return {"text": stuff_chain.invoke(state)}

    return _stuff


## Utility patterns


def build_set_route_pattern(route: ChatRoute) -> Callable[[RedboxState], dict[str, Any]]:
    """Returns a function that sets state["route_name"]."""

    def _set_route(state: RedboxState) -> dict[str, Any]:
        set_route_chain = (RunnablePassthrough() | StrOutputParser()).with_config(tags=["route_flag"])

        return {"route_name": set_route_chain.invoke(route.value)}

    return _set_route


def build_passthrough_pattern() -> Callable[[RedboxState], dict[str, Any]]:
    """Returns a function that uses state["request"] to set state["text"]."""

    def _passthrough(state: RedboxState) -> dict[str, Any]:
        return {
            "text": state["request"].question,
        }

    return _passthrough


def build_set_text_pattern(text: str, final_response_chain: bool = False):
    """Returns a function that can arbitrarily set state["text"] to a value."""

    def _set_text(state: RedboxState) -> dict[str, Any]:
        set_text_chain = RunnablePassthrough() | StrOutputParser()

        if final_response_chain:
            set_text_chain = set_text_chain.with_config(tags=["response_flag"])

        return {"text": set_text_chain.invoke(text)}

    return _set_text


# Raw processes


def clear_documents_process(state: RedboxState) -> dict[str, Any]:
    return {"documents": {group_id: None for group_id in state["documents"].keys()}}


def empty_process(state: RedboxState) -> None:
    return None
