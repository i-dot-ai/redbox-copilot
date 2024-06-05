import logging
import os
from http import HTTPStatus
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.encoders import jsonable_encoder
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_elasticsearch import ApproxRetrievalStrategy, ElasticsearchStore

from core_api.src.auth import get_user_uuid, get_ws_user_uuid
from redbox.llm.prompts.chat import (
    CONDENSE_QUESTION_PROMPT,
    STUFF_DOCUMENT_PROMPT,
    WITH_SOURCES_PROMPT,
)
from redbox.model_db import MODEL_PATH
from redbox.models import Settings
from redbox.models.chat import ChatRequest, ChatResponse, SourceDocument

# === Logging ===

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

env = Settings()


chat_app = FastAPI(
    title="Core Chat API",
    description="Redbox Core Chat API",
    version="0.1.0",
    openapi_tags=[
        {"name": "chat", "description": "Chat interactions with LLM and RAG backend"},
        {"name": "embedding", "description": "Embedding interactions with SentenceTransformer"},
        {"name": "llm", "description": "LLM information and parameters"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

embedding_model = SentenceTransformerEmbeddings(model_name=env.embedding_model, cache_folder=MODEL_PATH)
log.info("Loaded embedding model from environment: %s", env.embedding_model)


# === LLM setup ===

# Create the appropriate LLM, either openai, Azure, anthropic or bedrock
if env.openai_api_key is not None:
    log.info("Creating OpenAI LLM Client")
    llm = ChatLiteLLM(
        streaming=True,
        openai_key=env.openai_api_key,
    )
elif env.azure_openai_api_key is not None:
    log.info("Creating Azure LLM Client")
    log.debug("api_base: %s", env.azure_openai_endpoint)
    log.debug("api_version: %s", env.openai_api_version)

    # this nasty hack is required because, contrary to the docs:
    # using the api_version argument is not sufficient, and we need
    # to use the `OPENAI_API_VERSION` environment variable
    os.environ["OPENAI_API_VERSION"] = env.openai_api_version

    llm = ChatLiteLLM(
        model=env.azure_openai_model,
        streaming=True,
        azure_key=env.azure_openai_api_key,
        api_version=env.openai_api_version,
        api_base=env.azure_openai_endpoint,
    )
elif env.anthropic_api_key is not None:
    msg = "anthropic LLM not yet implemented"
    log.exception(msg)
    raise ValueError(msg)
else:
    msg = "Unknown LLM model specified or missing"
    log.exception(msg)
    raise ValueError(msg)


es = env.elasticsearch_client()
if env.elastic.subscription_level == "basic":
    strategy = ApproxRetrievalStrategy(hybrid=False)
elif env.elastic.subscription_level in ["platinum", "enterprise"]:
    strategy = ApproxRetrievalStrategy(hybrid=True)
else:
    message = f"Unknown Elastic subscription level {env.elastic.subscription_level}"
    raise ValueError(message)


vector_store = ElasticsearchStore(
    es_connection=es,
    index_name=f"{env.elastic_root_index}-chunk",
    embedding=embedding_model,
    strategy=strategy,
    vector_query_field="embedding",
)


@chat_app.post("/vanilla", tags=["chat"], response_model=ChatResponse)
def simple_chat(chat_request: ChatRequest, _user_uuid: Annotated[UUID, Depends(get_user_uuid)]) -> ChatResponse:
    """Get a LLM response to a question history"""

    if len(chat_request.message_history) < 2:  # noqa: PLR2004
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="Chat history should include both system and user prompts",
        )

    if chat_request.message_history[0].role != "system":
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="The first entry in the chat history should be a system prompt",
        )

    if chat_request.message_history[-1].role != "user":
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="The final entry in the chat history should be a user question",
        )

    chat_prompt = ChatPromptTemplate.from_messages((msg.role, msg.text) for msg in chat_request.message_history)
    # Convert to LangChain style messages
    messages = chat_prompt.format_messages()

    response = llm(messages)
    chat_response: ChatResponse = ChatResponse(output_text=response.content)
    return chat_response


async def build_retrieval_chain(
    chat_request: ChatRequest,
    user_uuid: UUID,
):
    question = chat_request.message_history[-1].text
    previous_history = list(chat_request.message_history[:-1])
    previous_history = ChatPromptTemplate.from_messages(
        (msg.role, msg.text) for msg in previous_history
    ).format_messages()

    docs_with_sources_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=WITH_SOURCES_PROMPT,
        document_prompt=STUFF_DOCUMENT_PROMPT,
        verbose=True,
    )

    condense_question_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    standalone_question = condense_question_chain({"question": question, "chat_history": previous_history})["text"]

    search_kwargs = {"filter": {"term": {"creator_user_uuid": str(user_uuid)}}}
    docs = vector_store.as_retriever(search_kwargs=search_kwargs).get_relevant_documents(standalone_question)

    params = {
        "question": standalone_question,
        "input_documents": docs,
    }

    return docs_with_sources_chain, params


@chat_app.post("/rag", tags=["chat"])
async def rag_chat(chat_request: ChatRequest, user_uuid: Annotated[UUID, Depends(get_user_uuid)]) -> ChatResponse:
    """Get a LLM response to a question history and file

    Args:


    Returns:
        StreamingResponse: a stream of the chain response
    """
    chain, params = await build_retrieval_chain(chat_request, user_uuid)
    result = chain(params)

    source_documents = [
        SourceDocument(
            page_content=langchain_document.page_content,
            file_uuid=langchain_document.metadata.get("parent_doc_uuid"),
            page_numbers=langchain_document.metadata.get("page_numbers"),
        )
        for langchain_document in result.get("input_documents", [])
    ]
    return ChatResponse(output_text=result["output_text"], source_documents=source_documents)


@chat_app.websocket("/rag")
async def rag_chat_streamed(websocket: WebSocket):
    await websocket.accept()

    user_uuid = await get_ws_user_uuid(websocket)

    chat_request = ChatRequest.parse_raw(await websocket.receive_text())

    chain, params = await build_retrieval_chain(chat_request, user_uuid)

    async for event in chain.astream_events(params, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            await websocket.send_json({"resource_type": "text", "data": event["data"]["chunk"].content})
        elif kind == "on_chat_model_end":
            await websocket.send_json({"resource_type": "end"})
        elif kind == "on_chain_stream":
            source_documents = [
                jsonable_encoder(
                    SourceDocument(
                        page_content=document.page_content,
                        file_uuid=document.metadata.get("parent_doc_uuid"),
                        page_numbers=document.metadata.get("page_numbers"),
                    )
                )
                for document in event["data"]["chunk"].get("input_documents", [])
            ]
            await websocket.send_json({"resource_type": "documents", "data": source_documents})

    await websocket.close()
