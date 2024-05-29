import logging
#from turtle import down
from typing import Annotated
from uuid import UUID
import os 
from fastapi import Depends, FastAPI, HTTPException
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_elasticsearch import ApproxRetrievalStrategy, ElasticsearchStore

from core_api.src.auth import get_user_uuid
from redbox.llm.prompts.chat import (
    CONDENSE_QUESTION_PROMPT,
    STUFF_DOCUMENT_PROMPT,
    WITH_SOURCES_PROMPT,
)
from redbox.model_db import MODEL_PATH
from redbox.models import EmbeddingModelInfo, Settings
from redbox.models.chat import ChatMessage, ChatRequest, ChatResponse, SourceDocument

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

log.info("Loading embedding model from environment: %s", env.embedding_model)
embedding_model = SentenceTransformerEmbeddings(model_name=env.embedding_model, cache_folder=MODEL_PATH)
log.info("Loaded embedding model from environment: %s", env.embedding_model)


def populate_embedding_model_info() -> EmbeddingModelInfo:
    test_text = "This is a test sentence."
    embedding = embedding_model.embed_documents([test_text])[0]
    embedding_model_info = EmbeddingModelInfo(
        embedding_model=env.embedding_model,
        vector_size=len(embedding),
    )
    return embedding_model_info


embedding_model_info = populate_embedding_model_info()


# === LLM setup ===

    # llm = ChatLiteLLM(
    #     model="gpt-3.5-turbo",
    #     streaming=True,
    # )
# Create the appropriate LLM, either openai, Azure, anthropic or bedrock
llm: ChatLiteLLM = None
if env.llm.type == "openai":
    log.info("Creating OpenAI LLM Client")
    llm = ChatLiteLLM(
        model=env.llm.name,
        streaming=True,
        openai_key=env.openai_api_key,
    )
elif env.llm.type == "azure":
    log.info("Creating Azure LLM Client")
    log.debug(f"api_base: {env.llm.api_base}")
    log.debug(f"api_version: {env.llm.open_api_version}")
    os.environ["OPENAI_API_VERSION"] = env.llm.open_api_version
    llm = ChatLiteLLM(
        model=env.llm.name,
        streaming=True,
        azure_key=env.azure_api_key,
        api_version=env.llm.open_api_version,
        api_base=env.llm.api_base,
        #set_verbosity=True
    )
elif env.llm.type == "anthropic":
    log.info("Anthropic LLM not yet implemented")
    raise ValueError("Anthropic LLM not yet implemented")
elif env.llm.type == "bedrock":
    log.info("Bedrock LLM not yet implemented")
    raise ValueError("Bedrock LLM not yet implemented")
else:
    raise ValueError(f"Unknown LLM model type {env.llm.type}")


es = env.elasticsearch_client()
if env.elastic.subscription_level == "basic":
    strategy = ApproxRetrievalStrategy(hybrid=False)
elif env.elastic.subscription_level in ["platinum", "enterprise"]:
    strategy = ApproxRetrievalStrategy(hybrid=True)
else:
    raise ValueError(f"Unknown Elastic subscription level {env.elastic.subscription_level}")


vector_store = ElasticsearchStore(
    es_connection=es,
    index_name="redbox-data-chunk",
    embedding=embedding_model,
    strategy=strategy,
    vector_query_field="embedding",
)


@chat_app.post("/vanilla", tags=["chat"], response_model=ChatResponse)
def simple_chat(chat_request: ChatRequest, _user_uuid: Annotated[UUID, Depends(get_user_uuid)]) -> ChatResponse:
    """Get a LLM response to a question history"""

    if len(chat_request.message_history) < 2:
        raise HTTPException(
            status_code=422,
            detail="Chat history should include both system and user prompts",
        )

    if chat_request.message_history[0].role != "system":
        raise HTTPException(
            status_code=422,
            detail="The first entry in the chat history should be a system prompt",
        )

    if chat_request.message_history[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The final entry in the chat history should be a user question",
        )

    chat_prompt = ChatPromptTemplate.from_messages((msg.role, msg.text) for msg in chat_request.message_history)
    # Convert to LangChain style messages
    messages = chat_prompt.format_messages()

    response = llm(messages)
    log.info(f"LLM Response: {response.content}")
    #chat_response:ChatResponse = ChatResponse(response_message=ChatMessage(text=response.text, role="ai"))
    chat_response:ChatResponse = ChatResponse(output_text=response.content)
    return chat_response


@chat_app.post("/rag", tags=["chat"])
def rag_chat(chat_request: ChatRequest, user_uuid: Annotated[UUID, Depends(get_user_uuid)]) -> ChatResponse:
    """Get a LLM response to a question history and file

    Args:


    Returns:
        StreamingResponse: a stream of the chain response
    """
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

    docs = vector_store.as_retriever(
        search_kwargs={"filter": {"term": {"creator_user_uuid.keyword": str(user_uuid)}}}
    ).get_relevant_documents(standalone_question)

    result = docs_with_sources_chain(
        {
            "question": standalone_question,
            "input_documents": docs,
        },
    )

    source_documents = [
        SourceDocument(
            page_content=langchain_document.page_content,
            file_uuid=langchain_document.metadata.get("parent_doc_uuid"),
            page_numbers=langchain_document.metadata.get("page_numbers"),
        )
        for langchain_document in result.get("input_documents", [])
    ]
    return ChatResponse(output_text=result["output_text"], source_documents=source_documents)
