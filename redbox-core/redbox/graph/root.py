from langgraph.graph import START, END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever

from redbox.graph.edges import (
    build_documents_bigger_than_context_conditional,
    multiple_docs_in_group_conditional,
    build_keyword_detection_conditional,
    documents_selected_conditional,
)
from redbox.graph.nodes.processes import (
    PromptSet,
    build_merge_pattern,
)
from redbox.models.chain import RedboxState
from redbox.models.chat import ChatRoute
from redbox.graph.nodes.processes import (
    build_chat_pattern,
    build_set_route_pattern,
    build_retrieve_pattern,
    build_stuff_pattern,
    build_passthrough_pattern,
    build_set_text_pattern,
    empty_process,
)
from redbox.graph.nodes.sends import build_document_chunk_send, build_document_group_send


# Global constants


FINAL_RESPONSE_TAG = "response_flag"
SOURCE_DOCUMENTS_TAG = "source_documents_flag"
ROUTE_NAME_TAG = "route_flag"
ROUTABLE_KEYWORDS = {ChatRoute.search: "Search for an answer to the question in the document"}


# Subgraphs


def get_chat_graph(
    llm: BaseChatModel,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a subgraph for standard chat."""
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node("p_set_chat_route", build_set_route_pattern(route=ChatRoute.chat))
    builder.add_node("p_chat", build_chat_pattern(llm=llm, prompt_set=PromptSet.Chat, final_response_chain=True))

    # Edges
    builder.add_edge(START, "p_set_chat_route")
    builder.add_edge("p_set_chat_route", "p_chat")
    builder.add_edge("p_chat", END)

    return builder.compile(debug=debug)


def get_search_graph(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a subgraph for retrieval augmented generation (RAG)."""
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node("p_set_search_route", build_set_route_pattern(route=ChatRoute.search))
    builder.add_node("p_condense_question", build_chat_pattern(llm=llm, prompt_set=PromptSet.CondenseQuestion))
    builder.add_node("p_retrieve_docs", build_retrieve_pattern(retriever=retriever))
    builder.add_node(
        "p_stuff_docs", build_stuff_pattern(llm=llm, prompt_set=PromptSet.Search, final_response_chain=True)
    )

    # Edges
    builder.add_edge(START, "p_set_search_route")
    builder.add_edge("p_set_search_route", "p_condense_question")
    builder.add_edge("p_condense_question", "p_retrieve_docs")
    builder.add_edge("p_retrieve_docs", "p_stuff_docs")
    builder.add_edge("p_stuff_docs", END)

    return builder.compile(debug=debug)


def get_chat_with_documents_graph(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a subgraph for chatting with documents."""
    builder = StateGraph(RedboxState)

    # Processes
    builder.add_node("p_pass_question_to_text", build_passthrough_pattern())
    builder.add_node("p_retrieve_docs", build_retrieve_pattern(retriever=retriever))
    builder.add_node("p_set_chat_docs_route", build_set_route_pattern(route=ChatRoute.chat_with_docs))
    builder.add_node("p_set_chat_docs_large_route", build_set_route_pattern(route=ChatRoute.chat_with_docs_map_reduce))
    builder.add_node(
        "p_summarise_each_document", build_merge_pattern(llm=llm, prompt_set=PromptSet.ChatwithDocsMapReduce)
    )
    builder.add_node(
        "p_summarise_document_by_document", build_merge_pattern(llm=llm, prompt_set=PromptSet.ChatwithDocsMapReduce)
    )
    builder.add_node(
        "p_summarise",
        build_stuff_pattern(
            llm=llm,
            prompt_set=PromptSet.ChatwithDocs,
            final_response_chain=True,
        ),
    )
    builder.add_node(
        "p_too_large_error",
        build_set_text_pattern(
            text="These documents are too large to work with.",
            final_response_chain=True,
        ),
    )

    # Decisions
    builder.add_node("d_all_docs_bigger_than_context", empty_process)
    builder.add_node("d_single_doc_summaries_bigger_than_context", empty_process)
    builder.add_node("d_doc_summaries_bigger_than_context", empty_process)
    builder.add_node("d_groups_have_multiple_docs", empty_process)

    # Sends
    builder.add_node("s_chunk", empty_process)
    builder.add_node("s_group_1", empty_process)
    builder.add_node("s_group_2", empty_process)

    # Edges
    builder.add_edge(START, "p_pass_question_to_text")
    builder.add_edge("p_pass_question_to_text", "p_retrieve_docs")
    builder.add_edge("p_retrieve_docs", "d_all_docs_bigger_than_context")
    builder.add_conditional_edges(
        "d_all_docs_bigger_than_context",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocsMapReduce),
        {
            True: "p_set_chat_docs_large_route",
            False: "p_set_chat_docs_route",
        },
    )
    builder.add_edge("p_set_chat_docs_route", "p_summarise")
    builder.add_edge("p_set_chat_docs_large_route", "s_chunk")
    builder.add_conditional_edges(
        "s_chunk", build_document_chunk_send("p_summarise_each_document"), path_map=["p_summarise_each_document"]
    )
    builder.add_edge("p_summarise_each_document", "d_groups_have_multiple_docs")
    builder.add_conditional_edges(
        "d_groups_have_multiple_docs",
        multiple_docs_in_group_conditional,
        {
            True: "s_group_1",
            False: "d_doc_summaries_bigger_than_context",
        },
    )
    builder.add_conditional_edges(
        "s_group_1",
        build_document_group_send("d_single_doc_summaries_bigger_than_context"),
        path_map=["d_single_doc_summaries_bigger_than_context"],
    )
    builder.add_conditional_edges(
        "d_single_doc_summaries_bigger_than_context",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocsMapReduce),
        {
            True: "s_group_2",
            False: "p_too_large_error",
        },
    )
    builder.add_conditional_edges(
        "s_group_2",
        build_document_group_send("p_summarise_document_by_document"),
        path_map=["p_summarise_document_by_document"],
    )
    builder.add_edge("p_summarise_document_by_document", "d_doc_summaries_bigger_than_context")
    builder.add_conditional_edges(
        "d_doc_summaries_bigger_than_context",
        build_documents_bigger_than_context_conditional(PromptSet.ChatwithDocs),
        {
            True: "p_too_large_error",
            False: "p_summarise",
        },
    )
    builder.add_edge("p_too_large_error", END)
    builder.add_edge("p_summarise", END)

    return builder.compile(debug=debug)


# Root graph


def get_root_graph(
    llm: BaseChatModel,
    all_chunks_retriever: VectorStoreRetriever,
    parameterised_retriever: VectorStoreRetriever,
    debug: bool = False,
) -> CompiledGraph:
    """Creates the core Redbox graph."""
    builder = StateGraph(RedboxState)

    # Subgraphs
    chat_subgraph = get_chat_graph(llm=llm, debug=debug)
    rag_subgraph = get_search_graph(llm=llm, retriever=parameterised_retriever, debug=debug)
    cwd_subgraph = get_chat_with_documents_graph(llm=llm, retriever=all_chunks_retriever, debug=debug)

    # Processes
    builder.add_node("p_search", rag_subgraph)
    builder.add_node(
        "p_no_keyword_error",
        build_set_text_pattern(
            text="That keyword isn't recognised",  # TODO: replace with env
            final_response_chain=True,
        ),
    )
    builder.add_node(
        "p_no_keyword_route",
        build_set_route_pattern(route=ChatRoute.error_no_keyword),
    )
    builder.add_node("p_chat", chat_subgraph)
    builder.add_node("p_chat_with_documents", cwd_subgraph)

    # Decisions
    builder.add_node("d_keyword_exists", empty_process)
    builder.add_node("d_docs_selected", empty_process)

    # Edges
    builder.add_edge(START, "d_keyword_exists")
    builder.add_conditional_edges(
        "d_keyword_exists",
        build_keyword_detection_conditional(*ROUTABLE_KEYWORDS.keys()),
        {ChatRoute.search: "p_search", ChatRoute.error_no_keyword: "p_no_keyword_error", "DEFAULT": "d_docs_selected"},
    )
    builder.add_conditional_edges(
        "d_docs_selected",
        documents_selected_conditional,
        {
            True: "p_chat_with_documents",
            False: "p_chat",
        },
    )
    builder.add_edge("p_search", END)
    builder.add_edge("p_no_keyword_error", "p_no_keyword_route")
    builder.add_edge("p_no_keyword_route", END)
    builder.add_edge("p_chat", END)
    builder.add_edge("p_chat_with_documents", END)

    return builder.compile(debug=debug)
