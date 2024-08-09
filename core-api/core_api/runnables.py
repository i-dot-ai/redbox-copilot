from langchain_core.runnables import chain

from redbox.models.chain import RedboxState
from redbox.models.chat import ChatResponse
from redbox.transform import map_document_to_source_document, flatten_document_state


@chain
def map_to_chat_response(state: RedboxState):
    """
    Create a ChatResponse at the end of a chain from a dict containing
    'response' a string to use as output_text
    'source_documents' a list of documents to map to source_documents
    """
    return ChatResponse(
        output_text=state["text"],
        source_documents=[
            map_document_to_source_document(d) for d in flatten_document_state(state.get("documents") or {})
        ],
        route_name=state["route_name"],
    )
