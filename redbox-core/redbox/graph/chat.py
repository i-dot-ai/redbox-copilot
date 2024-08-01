import logging
from langgraph.graph import StateGraph, START
from langgraph.constants import Send
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import chain, RunnableLambda, Runnable
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import TextSplitter, TokenTextSplitter
from tiktoken import Encoding

from redbox.api.format import format_documents
from redbox.models.chain import ChainState, ChatMapReduceState
from redbox.models.chat import ChatRoute
from redbox.chains.graph import (
    set_prompt_args,
    build_llm_chain,
    make_chat_prompt_from_messages_runnable,
    build_get_docs,
    get_no_docs_available,
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


def get_chat_graph(llm: BaseChatModel, tokeniser: Encoding, env: Settings, debug: bool = False) -> CompiledGraph:
    app = StateGraph(ChainState)
    app.set_entry_point("set_chat_prompt_args")

    app.add_node("set_chat_prompt_args", set_prompt_args)
    app.add_edge("set_chat_prompt_args", "llm")

    app.add_node("llm", build_llm_chain(llm, tokeniser, env, env.ai.chat_system_prompt, env.ai.chat_question_prompt))

    return app


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


def build_llm_map_chain(
    llm: BaseChatModel, tokeniser: Encoding, env: Settings, system_prompt: str, question_prompt: str
) -> Runnable:
    return (
        make_chat_prompt_from_messages_runnable(
            system_prompt=system_prompt,
            question_prompt=question_prompt,
            input_token_budget=env.ai.context_window_size - env.llm_max_tokens,
            tokeniser=tokeniser,
        )
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda s: {"intermediate_docs": [Document(page_content=s)]})
    )


def get_chat_with_docs_graph(
    llm: BaseChatModel,
    all_chunks_retriever: VectorStoreRetriever,
    tokeniser: Encoding,
    env: Settings,
    debug: bool = False,
) -> CompiledGraph:
    app = StateGraph(ChainState)

    app.add_node("get_chat_docs", build_get_docs(env, all_chunks_retriever))
    app.add_node("set_chat_prompt_args", set_prompt_args)
    app.add_node("set_chat_method", set_chat_method)

    app.add_node("no_docs_available", get_no_docs_available(env))
    app.add_node(
        "llm",
        build_llm_chain(
            llm, tokeniser, env, env.ai.chat_with_docs_system_prompt, env.ai.chat_with_docs_question_prompt
        ),
    )
    app.add_node(ChatRoute.chat_with_docs_map_reduce, get_chat_with_docs_map_reduce_graph(llm, tokeniser, env, debug))

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
    return app


def get_chat_with_docs_map_reduce_graph(
    llm: BaseChatModel, tokeniser: Encoding, env: Settings, debug: bool = False
) -> CompiledGraph:
    app = StateGraph(ChatMapReduceState)

    app.add_node(
        "llm_map", build_llm_map_chain(llm, tokeniser, env, env.ai.map_system_prompt, env.ai.chat_map_question_prompt)
    )
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

    return app
