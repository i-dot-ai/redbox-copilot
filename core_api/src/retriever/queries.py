from typing import Any, TypedDict
from uuid import UUID

from langchain_core.embeddings.embeddings import Embeddings


class ESQuery(TypedDict):
    question: str
    file_uuids: list[UUID]
    user_uuid: UUID


class ESParams(TypedDict):
    size: int
    num_candidates: int
    match_boost: float
    knn_boost: float
    similarity_threshold: float


def get_all(query: ESQuery) -> dict[str, Any]:
    """
    Returns a parameterised elastic query that will return everything it matches.

    As it's used in summarisation, it excludes embeddings.
    """
    query_filter = [
        {
            "bool": {
                "should": [
                    {"term": {"creator_user_uuid.keyword": str(query["user_uuid"])}},
                    {"term": {"metadata.creator_user_uuid.keyword": str(query["user_uuid"])}},
                ]
            }
        }
    ]
    if len(query["file_uuids"]) != 0:
        query_filter.append(
            {
                "bool": {
                    "should": [
                        {"terms": {"parent_file_uuid.keyword": [str(uuid) for uuid in query["file_uuids"]]}},
                        {"terms": {"metadata.parent_file_uuid.keyword": [str(uuid) for uuid in query["file_uuids"]]}},
                    ]
                }
            }
        )
    return {
        "_source": {"excludes": ["*embedding"]},
        "query": {"bool": {"must": {"match_all": {}}, "filter": query_filter}},
    }


def get_some(
    embedding_model: Embeddings, params: ESParams, embedding_field_name: str, query: ESQuery
) -> dict[str, Any]:
    vector = embedding_model.embed_query(query["question"])

    query_filter = [
        {
            "bool": {
                "should": [
                    {"term": {"creator_user_uuid.keyword": str(query["user_uuid"])}},
                    {"term": {"metadata.creator_user_uuid.keyword": str(query["user_uuid"])}},
                ]
            }
        }
    ]

    if len(query["file_uuids"]) != 0:
        query_filter.append(
            {
                "bool": {
                    "should": [
                        {"terms": {"parent_file_uuid.keyword": [str(uuid) for uuid in query["file_uuids"]]}},
                        {"terms": {"metadata.parent_file_uuid.keyword": [str(uuid) for uuid in query["file_uuids"]]}},
                    ]
                }
            }
        )

    return {
        "size": params["size"],
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": query["question"],
                                "boost": params["match_boost"],
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": embedding_field_name,
                            "query_vector": vector,
                            "num_candidates": params["num_candidates"],
                            "filter": query_filter,
                            "boost": params["knn_boost"],
                            "similarity": params["similarity_threshold"],
                        }
                    },
                ],
                "filter": query_filter,
            }
        },
    }
