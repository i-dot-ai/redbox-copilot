import logging
from langgraph.graph import StateGraph, START
from langgraph.constants import Send
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import chain, RunnableLambda, Runnable
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import TextSplitter, TokenTextSplitter
from tiktoken import Encoding

from redbox.api.format import format_documents
from redbox.chains.components import get_llm
from redbox.models.chain import ChainState, ChatMapReduceState
from redbox.models.chat import ChatRoute
from redbox.chains.graph import (
    set_prompt_args,
    build_llm_chain,
    make_chat_prompt_from_messages_runnable,
    build_get_docs,
    set_state_field,
)
from redbox.models.settings import Settings

log = logging.getLogger()


@chain
def to_map_step(state: ChatMapReduceState):
    """
    Map each doc in the state to an execution of the llm map step which will create an answer
    per current document
    """
    return [
        Send(
            "llm_map",
            ChatMapReduceState(
                query=state["query"], documents=[doc], route_name=state["route_name"], prompt_args=state["prompt_args"]
            ),
        )
        for doc in state["documents"]
    ]


def build_reduce_docs_step(splitter: TextSplitter):
    return RunnableLambda(
        lambda state: [
            Document(page_content=s) for s in splitter.split_text(format_documents(state["intermediate_docs"]))
        ]
    ) | RunnableLambda(lambda docs: {"documents": docs})


def get_chat_graph(tokeniser: Encoding, debug: bool = False) -> CompiledGraph:
    app = StateGraph(ChainState)
    app.set_entry_point("set_chat_prompt_args")

    app.add_node("set_chat_prompt_args", set_prompt_args)
    app.add_edge("set_chat_prompt_args", "llm")

    app.add_node(
        "llm",
        build_llm_chain(tokeniser, final_response_chain=True),
    )

    return app.compile(debug=debug)


@chain
def set_chat_method(state: ChainState):
    """
    Choose an approach to chatting based on the current state
    """
    log.debug("Selecting chat method")
    number_of_docs = len(state["documents"])
    if number_of_docs == 0:
        selected_tool = ChatRoute.chat
    elif number_of_docs == 1:
        selected_tool = ChatRoute.chat_with_docs
    else:
        selected_tool = ChatRoute.chat_with_docs_map_reduce
    log.info(f"Selected: {selected_tool} for execution")
    return {"route_name": selected_tool}


def build_llm_map_chain(tokeniser: Encoding) -> Runnable:
    @chain
    def _build_llm_map_chain(chain_state: ChainState):
        llm = get_llm(chain_state["query"].ai_settings)
        return (
            make_chat_prompt_from_messages_runnable(
                tokeniser=tokeniser, llm_max_tokens=chain_state["query"].ai_settings.llm_max_tokens
            )
            | llm
            | StrOutputParser()
            | RunnableLambda(lambda s: {"intermediate_docs": [Document(page_content=s)]})
        )

    return _build_llm_map_chain


def get_chat_with_docs_graph(
    all_chunks_retriever: VectorStoreRetriever,
    tokeniser: Encoding,
    env: Settings,
    debug: bool = False,
) -> CompiledGraph:
    app = StateGraph(ChainState)

    app.add_node("get_chat_docs", build_get_docs(all_chunks_retriever))
    app.add_node("set_chat_prompt_args", set_prompt_args)
    app.add_node("set_chat_method", set_chat_method)

    app.add_node("no_docs_available", set_state_field("response", env.response_no_doc_available))
    app.add_node(
        "llm",
        build_llm_chain(
            tokeniser,
            final_response_chain=True,
        ),
    )
    app.add_node(
        ChatRoute.chat_with_docs_map_reduce,
        get_chat_with_docs_map_reduce_graph(tokeniser, env, debug),
    )
    app.add_node("clear_documents", set_state_field("documents", []))

    app.add_edge(START, "get_chat_docs")
    app.add_edge("get_chat_docs", "set_chat_prompt_args")
    app.add_edge("set_chat_prompt_args", "set_chat_method")
    app.add_conditional_edges(
        "set_chat_method",
        lambda state: state["route_name"],
        {
            ChatRoute.chat: "no_docs_available",
            ChatRoute.chat_with_docs: "llm",
            ChatRoute.chat_with_docs_map_reduce: ChatRoute.chat_with_docs_map_reduce,
        },
    )
    app.add_edge(ChatRoute.chat_with_docs_map_reduce, "set_chat_prompt_args")

    # Remove docs so we don't provide citations for chat (using whole doc so irrelevant anyway)
    app.add_edge("llm", "clear_documents")

    return app.compile(debug=debug)


def get_chat_with_docs_map_reduce_graph(tokeniser: Encoding, env: Settings, debug: bool = False) -> CompiledGraph:
    app = StateGraph(ChatMapReduceState)

    app.add_node("llm_map", build_llm_map_chain(tokeniser))
    app.add_node(
        "reduce",
        build_reduce_docs_step(
            TokenTextSplitter(
                model_name="gpt-4",
                chunk_size=env.worker_ingest_largest_chunk_size,
                chunk_overlap=env.worker_ingest_largest_chunk_overlap,
            )
        ),
    )

    app.add_conditional_edges(START, to_map_step, then="reduce")

    return app.compile(debug=debug)
