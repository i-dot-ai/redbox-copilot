from langchain_core.documents import Document

from redbox.models.chat import SourceDocument
from redbox.models.chain import DocumentState


def map_document_to_source_document(d: Document) -> SourceDocument:
    def map_page_numbers(page_number: int | list[int] | None) -> list[int]:
        if isinstance(page_number, list):
            return page_number
        elif isinstance(page_number, int):
            return [page_number]
        else:
            return []

    return SourceDocument(
        page_content=d.page_content,
        file_uuid=d.metadata["parent_file_uuid"],
        page_numbers=map_page_numbers(d.metadata.get("page_number")),
    )


# This should be unnecessary and indicates we're not chunking correctly
def combine_documents(a: Document, b: Document):
    def listify(metadata: dict, field_name: str) -> list:
        field_value = metadata.get(field_name)
        if isinstance(field_value, list):
            return field_value
        if field_value is None:
            return []
        return [field_value]

    def sorted_list_or_none(obj: list):
        return sorted(set(obj)) or None

    def combine_values(field_name: str):
        return sorted_list_or_none(listify(a.metadata, field_name) + listify(b.metadata, field_name))

    combined_content = a.page_content + b.page_content
    combined_metadata = a.metadata.copy()
    combined_metadata["token_count"] = a.metadata["token_count"] + b.metadata["token_count"]
    combined_metadata["page_number"] = combine_values("page_number")
    combined_metadata["languages"] = combine_values("languages")
    combined_metadata["link_texts"] = combine_values("link_texts")
    combined_metadata["link_urls"] = combine_values("link_urls")
    combined_metadata["links"] = combine_values("links")

    return Document(page_content=combined_content, metadata=combined_metadata)


def structure_documents(docs: list[Document]) -> DocumentState:
    return {
        g_id: {d.metadata["uuid"]: d for d in [d for d in docs if d.metadata["parent_file_uuid"] == g_id]}
        for g_id in [d.metadata["parent_file_uuid"] for d in docs]
    }


def flatten_document_state(documents: DocumentState) -> list[Document]:
    if not documents:
        return []
    return [document for group in documents.values() for document in group.values()]
