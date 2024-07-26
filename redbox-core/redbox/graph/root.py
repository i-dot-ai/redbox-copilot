import typing
import os
import sys
import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from tiktoken import Encoding
import asyncio

from redbox.chains.graph import set_route
from redbox.graph.search import get_search_graph
from redbox.models.chain import ChainInput, ChainState
from redbox.models.chat import ChatRoute
from redbox.models.settings import Settings
from redbox.chains.components import get_all_chunks_retriever, get_parameterised_retriever, get_chat_llm, get_tokeniser
from redbox.graph.chat import get_chat_graph, get_chat_with_docs_graph


def get_redbox_graph(
    llm: BaseChatModel = None,
    all_chunks_retriever: VectorStoreRetriever = None,
    parameterised_retriever: VectorStoreRetriever = None,
    tokeniser: Encoding = None,
    env: Settings = None,
    debug: bool = False,
):
    _env = env or Settings()
    _all_chunks_retriever = all_chunks_retriever or get_all_chunks_retriever(_env)
    _parameterised_retriever = parameterised_retriever or get_parameterised_retriever(_env)
    _llm = llm or get_chat_llm(_env)
    _tokeniser = tokeniser or get_tokeniser()

    app = StateGraph(ChainState)
    app.set_entry_point("set_route")

    app.add_node("set_route", set_route)
    app.add_conditional_edges("set_route", lambda s: s["route_name"])

    app.add_node(ChatRoute.search, get_search_graph(_llm, _parameterised_retriever, _tokeniser, _env, debug))
    app.add_edge(ChatRoute.search, END)

    app.add_node(ChatRoute.chat, get_chat_graph(_llm, _tokeniser, _env, debug))
    app.add_edge(ChatRoute.chat, END)

    app.add_node(
        ChatRoute.chat_with_docs, get_chat_with_docs_graph(_llm, _all_chunks_retriever, _tokeniser, _env, debug)
    )
    app.add_edge(ChatRoute.chat_with_docs, END)

    return app.compile(debug=debug)


async def run_redbox(
    input: ChainState,
    app: CompiledGraph,
    response_tokens_callback: typing.Callable[[str], None] = lambda _: _,
) -> ChainState:
    final_state = None
    async for event in app.astream_events(input, version="v2"):
        kind = event["event"]
        tags = event.get("tags", [])
        if kind == "on_chat_model_stream" and "response" in tags:
            data = event["data"]
            if data["chunk"].content:
                response_tokens_callback(data["chunk"].content)
        if kind == "on_chain_end" and event["name"] == "LangGraph":
            final_state = ChainState(**event["data"]["output"])
    return final_state


if __name__ == "__main__":
    import os

    logging.basicConfig(stream=sys.stdout, level=os.environ.get("LOG_LEVEL", "INFO"))

    app = get_redbox_graph()
    response = asyncio.run(
        run_redbox(
            ChainState(
                query=ChainInput(
                    question="What are Labour's five missions?",
                    # file_uuids=[],
                    file_uuids=["68e5d196-636e-4847-95ad-6c40ba20e390"],
                    user_uuid="a93a8f40-f261-4f12-869a-2cea3f3f0d71",
                    chat_history=[],
                )
            ),
            app,
        )
    )
    print()
    print(f"{len(response["documents"])} source documents")
    print(f"Used {response["route_name"]}")
    print()
    print(response["response"])