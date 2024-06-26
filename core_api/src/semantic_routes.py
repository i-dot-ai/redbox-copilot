from typing import Annotated

from fastapi import Depends
from langchain_core.runnables import Runnable
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

from core_api.src.build_chains import (
    build_retrieval_chain,
    build_static_response_chain,
    build_summary_chain,
    build_vanilla_chain,
)
from redbox.model_db import MODEL_PATH
from redbox.models.chat import ChatRoute

# === Pre-canned responses for non-LLM routes ===
INFO_RESPONSE = """
I am RedBox, an AI focused on helping UK Civil Servants, Political Advisors and
Ministers triage and summarise information from a wide variety of sources.
"""

ABILITY_RESPONSE = """
* I can help you search over selected documents and do Q&A on them.
* I can help you summarise selected documents.
* I can help you extract information from selected documents.
* I can return information in a variety of formats, such as bullet points.
"""

COACH_RESPONSE = """
I am sorry that didn't work.
You could try rephrasing your task, i.e if you want to summarise a document please use the term,
"Summarise the selected document" or "extract all action items from the selected document."
If you want the results to be returned in a specific format, please specify the format in as much detail as possible.
"""

# === Set up the semantic router ===
info = Route(
    name=ChatRoute.info.value,
    utterances=[
        "What is your name?",
        "Who are you?",
        "What is Redbox?",
    ],
)

ability = Route(
    name=ChatRoute.ability.value,
    utterances=[
        "What can you do?",
        "What can you do?",
        "How can you help me?",
        "What does Redbox do?",
        "What can Redbox do",
        "What don't you do",
        "Please help me",
        "Please help",
        "Help me!",
        "help",
    ],
)

coach = Route(
    name=ChatRoute.coach.value,
    utterances=[
        "That is not the answer I wanted",
        "Rubbish",
        "No good",
        "That's not what I wanted",
        "How can I improve the results?",
    ],
)

gratitude = Route(
    name=ChatRoute.gratitude.value,
    utterances=[
        "Thank you ever so much for your help!",
        "I'm really grateful for your assistance.",
        "Cheers for the detailed response!",
        "Thanks a lot, that was very informative.",
        "Nice one",
        "Thanks!",
    ],
)

summarisation = Route(
    name=ChatRoute.summarise.value,
    utterances=[
        "I'd like to summarise the documents I've uploaded.",
        "Can you help me with summarising these documents?",
        "Please summarise the documents with a focus on the impact on northern England",
        "Please summarise the contents of the uploaded files.",
        "I'd appreciate a summary of the documents I've just uploaded.",
        "Could you provide a summary of these uploaded documents?",
        "Summarise the documents with a focus on macro economic trends.",
    ],
)

extract = Route(
    name=ChatRoute.extract.value,
    utterances=[
        "I'd like to find some information in the documents I've uploaded",
        "Can you help me identify details from these documents?",
        "Please give me all action items from this document",
        "Give me all the action items from these meeting notes",
        "Could you locate some key information in these uploaded documents?",
        "I need to obtain certain details from the documents I have uploaded, please",
        "Please extract all action items from this document",
        "Extract all the sentences with the word 'shall'",
    ],
)

vanilla = Route(
    name=ChatRoute.chat.value,
    utterances=[
        "What is the capital of France?",
        "Write me a fun poem about frogs.",
        "What is the meaning of life?",
        "blah blah blah",
        "How to make jam",
        "Cool beans",
        "Why is the sky blue?",
        "recipe for scones",
        "What?",
        "Huh?",
        "You tell me",
        "Do that but shorter",
        "Do that but in bullet points",
        "markdown table please",
        "Wow",
        "I don't know",
        "Does this work?",
        "Why?",
        "Gosh",
        "What is this?",
    ],
)

__semantic_routing_encoder = None
__routable_chains = None
__semantic_route_layer = None


def get_semantic_routes():
    return (info, ability, coach, gratitude, summarisation, vanilla)


def get_semantic_routing_encoder():
    global __semantic_routing_encoder  # noqa: PLW0603
    if not __semantic_routing_encoder:
        __semantic_routing_encoder = HuggingFaceEncoder(
            name="sentence-transformers/paraphrase-albert-small-v2",
            cache_dir=MODEL_PATH,
        )
    return __semantic_routing_encoder


def get_semantic_route_layer(routes: Annotated[list[Route], Depends(get_semantic_routes)]):
    global __semantic_route_layer  # noqa: PLW0603
    if not __semantic_route_layer:
        __semantic_route_layer = RouteLayer(encoder=get_semantic_routing_encoder(), routes=routes)
    return __semantic_route_layer


def get_routable_chains(
    retrieval_chain: Annotated[Runnable, Depends(build_retrieval_chain)],
    summary_chain: Annotated[Runnable, Depends(build_summary_chain)],
    vanilla_chain: Annotated[Runnable, Depends(build_vanilla_chain)],
):
    global __routable_chains  # noqa: PLW0603
    if not __routable_chains:
        __routable_chains = {
            ChatRoute.info: build_static_response_chain(INFO_RESPONSE, ChatRoute.info),
            ChatRoute.ability: build_static_response_chain(ABILITY_RESPONSE, ChatRoute.ability),
            ChatRoute.coach: build_static_response_chain(COACH_RESPONSE, ChatRoute.coach),
            ChatRoute.gratitude: build_static_response_chain("You're welcome!", ChatRoute.gratitude),
            ChatRoute.chat: vanilla_chain,
            ChatRoute.search: retrieval_chain,
            ChatRoute.summarise: summary_chain,
            # Experimental routes
            ChatRoute.supasearch: retrieval_chain.with_config(
                configurable={
                    "params": {
                        "size": 10,
                        "num_candidates": 100,
                        "match_boost": 1,
                        "knn_boost": 2,
                        "similarity_threshold": 0.7,
                    }
                }
            ),
        }
    return __routable_chains
