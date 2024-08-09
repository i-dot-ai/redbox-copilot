from redbox.models.chain import RedboxState
from redbox.retriever import AllElasticsearchRetriever
from redbox.test.data import RedboxChatTestCase


def test_all_chunks_retriever(
    all_chunks_retriever: AllElasticsearchRetriever, stored_file_all_chunks: RedboxChatTestCase
):
    result = all_chunks_retriever.invoke(RedboxState(request=stored_file_all_chunks.query))

    assert len(result) == len(stored_file_all_chunks.get_docs_matching_query())
    assert {c.page_content for c in result} | {c.page_content for c in stored_file_all_chunks.docs}
    assert {c.metadata["parent_file_uuid"] for c in result} == set(map(str, stored_file_all_chunks.query.file_uuids))
