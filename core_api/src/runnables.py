from operator import itemgetter

from langchain.schema import StrOutputParser
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from core_api.src.format import format_chunks


def make_stuff_document_runnable(
    system_prompt: str,
    llm: ChatLiteLLM,
) -> Runnable:
    """Takes a system prompt and LLM returns a stuff document runnable.

    Runnable takes input of a dict keyed to question, messages and documents.

    Runnable returns a dict keyed to response and sources.
    """
    chat_history = [
        ("system", system_prompt),
        ("placeholder", "{messages}"),
        ("user", "Question: {question}. \n\n Documents: \n\n {documents} \n\n Answer: "),
    ]

    stuff_doc_chain = (
        {
            "question": itemgetter("question"),
            "messages": itemgetter("messages"),
            "documents": itemgetter("documents") | RunnableLambda(format_chunks),
        }
        | ChatPromptTemplate.from_messages(chat_history)
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough()
        | {
            "response": stuff_doc_chain,
            "sources": itemgetter("documents")
        }
    )
