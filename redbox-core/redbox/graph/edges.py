import logging
import re

from langchain_core.runnables import Runnable

from redbox.models.chain import get_prompts
from redbox.chains.components import get_tokeniser
from redbox.graph.nodes.processes import PromptSet
from redbox.models import ChatRoute
from redbox.models.chain import RedboxState
from redbox.transform import flatten_document_state

log = logging.getLogger()


def calculate_token_budget(state: RedboxState, system_prompt: str, question_prompt: str) -> int:
    tokeniser = get_tokeniser()

    len_question_prompt = len(tokeniser.encode(question_prompt))
    len_system_prompt = len(tokeniser.encode(system_prompt))

    ai_settings = state["request"].ai_settings

    return ai_settings.context_window_size - ai_settings.llm_max_tokens - len_system_prompt - len_question_prompt


def build_documents_bigger_than_context_conditional(prompt_set: PromptSet) -> Runnable:
    """Uses a set of prompts to build the correct conditional for exceeding the context window."""

    def _documents_bigger_than_context_conditional(state: RedboxState) -> bool:
        system_prompt, question_prompt = get_prompts(state, prompt_set)
        token_budget = calculate_token_budget(state, system_prompt, question_prompt)

        if sum(d.metadata["token_count"] for d in flatten_document_state(state["documents"])) > token_budget:
            return True
        else:
            return False

    return _documents_bigger_than_context_conditional


def build_keyword_detection_conditional(*allowed_routes: ChatRoute) -> Runnable:
    """Given a set of permitted routes, will detect them in keywords."""

    def _keyword_detection_conditional(state: RedboxState) -> ChatRoute | str:
        re_keyword_pattern = re.compile(r"@(\w+)")

        route_match = re_keyword_pattern.search(state["request"].question)
        route_name = route_match.group()[1:] if route_match else None

        try:
            route = ChatRoute[route_name]
            if route in allowed_routes:
                return route
        except KeyError:
            if route_name is not None:
                return ChatRoute.error_no_keyword

        return "DEFAULT"

    return _keyword_detection_conditional


def documents_selected_conditional(state: RedboxState) -> bool:
    return len(state["request"].file_uuids) > 0


def multiple_docs_in_group_conditional(state: RedboxState) -> bool:
    for group in state["documents"]:
        if len(state["documents"][group]) > 0:
            return True
    return False
