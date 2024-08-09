from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from tiktoken import Encoding

from redbox.graph.root import get_root_graph
from redbox.models.chain import RedboxState
from redbox.models.chat import ChatRoute
from redbox.models.settings import Settings
from redbox.chains.components import get_all_chunks_retriever, get_parameterised_retriever, get_chat_llm
from redbox.graph.root import (
    ROUTABLE_KEYWORDS,
    ROUTE_NAME_TAG,
    FINAL_RESPONSE_TAG,
    SOURCE_DOCUMENTS_TAG,
)


async def _default_callback(*args, **kwargs):
    return None


class Redbox:
    def __init__(
        self,
        llm: BaseChatModel | None = None,
        all_chunks_retriever: VectorStoreRetriever | None = None,
        parameterised_retriever: VectorStoreRetriever | None = None,
        tokeniser: Encoding | None = None,
        env: Settings | None = None,
        debug: bool = False,
    ):
        _env = env or Settings()
        _all_chunks_retriever = all_chunks_retriever or get_all_chunks_retriever(_env)
        _parameterised_retriever = parameterised_retriever or get_parameterised_retriever(_env)
        _llm = llm or get_chat_llm(_env)

        self.graph = get_root_graph(_llm, _all_chunks_retriever, _parameterised_retriever, debug)

    async def run(
        self,
        input: RedboxState,
        response_tokens_callback=_default_callback,
        route_name_callback=_default_callback,
        documents_callback=_default_callback,
    ) -> RedboxState:
        final_state = None
        async for event in self.graph.astream_events(input, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            if kind == "on_chat_model_stream" and FINAL_RESPONSE_TAG in tags:
                await response_tokens_callback(event["data"]["chunk"].content)
            elif kind == "on_chain_end" and FINAL_RESPONSE_TAG in tags:
                await response_tokens_callback(event["data"]["output"])
            elif kind == "on_chain_end" and ROUTE_NAME_TAG in tags:
                await route_name_callback(event["data"]["output"])
            elif kind == "on_retriever_end" and SOURCE_DOCUMENTS_TAG in tags:
                await documents_callback(event["data"]["output"])
            elif kind == "on_chain_end" and event["name"] == "LangGraph":
                final_state = RedboxState(**event["data"]["output"])
        return final_state

    def get_available_keywords(self) -> dict[ChatRoute, str]:
        return ROUTABLE_KEYWORDS
