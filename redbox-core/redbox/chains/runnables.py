import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, chain
from tiktoken import Encoding

from redbox.api.format import format_documents
from redbox.chains.components import get_tokeniser
from redbox.models.chain import ChainChatMessage, RedboxState
from redbox.models.errors import QuestionLengthError
from redbox.models.chain import PromptSet, get_prompts
from redbox.transform import flatten_document_state


log = logging.getLogger()


def build_chat_prompt_from_messages_runnable(prompt_set: PromptSet, tokeniser: Encoding = None):
    @chain
    def _chat_prompt_from_messages(state: RedboxState) -> Runnable:
        """
        Create a ChatPromptTemplate as part of a chain using 'chat_history'.
        Returns the PromptValue using values in the input_dict
        """
        _tokeniser = tokeniser or get_tokeniser()
        system_prompt, question_prompt = get_prompts(state, prompt_set)

        log.debug("Setting chat prompt")
        system_prompt_message = [("system", system_prompt)]
        prompts_budget = len(_tokeniser.encode(system_prompt)) + len(_tokeniser.encode(question_prompt))
        chat_history_budget = (
            state["request"].ai_settings.context_window_size
            - state["request"].ai_settings.llm_max_tokens
            - prompts_budget
        )

        if chat_history_budget <= 0:
            raise QuestionLengthError

        truncated_history: list[ChainChatMessage] = []
        for msg in state["request"].chat_history[::-1]:
            chat_history_budget -= len(tokeniser.encode(msg["text"]))
            if chat_history_budget <= 0:
                break
            else:
                truncated_history.insert(0, msg)

        prompt_template_context = state["request"].model_dump() | {
            "formatted_documents": format_documents(flatten_document_state(state["documents"])),
        }

        return ChatPromptTemplate.from_messages(
            system_prompt_message
            + [(msg["role"], msg["text"]) for msg in truncated_history]
            + [("user", question_prompt)]
        ).invoke(prompt_template_context)

    return _chat_prompt_from_messages
