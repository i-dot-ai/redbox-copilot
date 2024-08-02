from langgraph.graph import StateGraph
from langchain_core.vectorstores import VectorStoreRetriever
from tiktoken import Encoding

from redbox.chains.graph import set_route, set_state_field
from redbox.graph.search import get_search_graph
from redbox.models.chain import ChainState
from redbox.models.chat import ChatRoute
from redbox.models.settings import Settings
from redbox.chains.components import get_all_chunks_retriever, get_parameterised_retriever, get_tokeniser
from redbox.graph.chat import get_chat_graph, get_chat_with_docs_graph


async def _default_callback(*args, **kwargs):
    return None


class Redbox:
    FINAL_RESPONSE_TAG = "response_flag"
    SOURCE_DOCUMENTS_TAG = "source_documents_flag"
    ROUTE_NAME_TAG = "route_flag"

    # Non keywords
    ROUTABLE_BUILTIIN = [ChatRoute.chat, ChatRoute.chat_with_docs, ChatRoute.error_no_keyword]

    # Keyword routes
    ROUTABLE_KEYWORDS = {ChatRoute.search: "Search for an answer to the question in the document"}

    def __init__(
        self,
        all_chunks_retriever: VectorStoreRetriever | None = None,
        parameterised_retriever: VectorStoreRetriever | None = None,
        tokeniser: Encoding | None = None,
        env: Settings | None = None,
        debug: bool = False,
    ):
        _env = env or Settings()
        _all_chunks_retriever = all_chunks_retriever or get_all_chunks_retriever(_env)
        _parameterised_retriever = parameterised_retriever or get_parameterised_retriever(_env)
        _tokeniser = tokeniser or get_tokeniser()

        app = StateGraph(ChainState)
        app.set_entry_point("set_route")

        app.add_node("set_route", set_route.with_config(tags=[Redbox.ROUTE_NAME_TAG]))
        app.add_conditional_edges(
            "set_route",
            lambda s: s["route_name"],
            {x: x for x in Redbox.ROUTABLE_BUILTIIN + list(Redbox.ROUTABLE_KEYWORDS.keys())},
        )

        app.add_node(
            ChatRoute.search,
            get_search_graph(
                _parameterised_retriever.with_config(tags=[Redbox.SOURCE_DOCUMENTS_TAG]),
                _tokeniser,
                debug,
            ),
        )
        app.add_node(ChatRoute.chat, get_chat_graph(_tokeniser, debug))
        app.add_node(
            ChatRoute.chat_with_docs,
            get_chat_with_docs_graph(
                _all_chunks_retriever.with_config(tags=[Redbox.SOURCE_DOCUMENTS_TAG]), _tokeniser, _env, debug
            ),
        )
        app.add_node(
            ChatRoute.error_no_keyword,
            set_state_field("response", env.response_no_such_keyword).with_config(tags=[Redbox.FINAL_RESPONSE_TAG]),
        )

        self.graph = app.compile(debug=debug)

    async def run(
        self,
        input: ChainState,
        response_tokens_callback=_default_callback,
        route_name_callback=_default_callback,
        documents_callback=_default_callback,
    ) -> ChainState:
        final_state = None
        async for event in self.graph.astream_events(input, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            if kind == "on_chat_model_stream" and Redbox.FINAL_RESPONSE_TAG in tags:
                await response_tokens_callback(event["data"]["chunk"].content)
            elif kind == "on_chain_end" and Redbox.FINAL_RESPONSE_TAG in tags:
                await response_tokens_callback(event["data"]["output"]["response"])
            elif kind == "on_chain_end" and Redbox.ROUTE_NAME_TAG in tags:
                await route_name_callback(event["data"]["output"]["route_name"])
            elif kind == "on_retriever_end" and Redbox.SOURCE_DOCUMENTS_TAG in tags:
                await documents_callback(event["data"]["output"])
            elif kind == "on_chain_end" and event["name"] == "LangGraph":
                final_state = ChainState(**event["data"]["output"])
        return final_state

    def get_available_keywords(self) -> dict[ChatRoute, str]:
        return Redbox.ROUTABLE_KEYWORDS
