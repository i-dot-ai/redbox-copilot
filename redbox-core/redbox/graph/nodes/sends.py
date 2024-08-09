from typing import Callable

from langgraph.constants import Send

from redbox.models.chain import RedboxState


def build_document_group_send(target: str) -> Callable[[RedboxState], list[Send]]:
    def _group_send(state: RedboxState) -> list[Send]:
        group_send_states: list[RedboxState] = [
            RedboxState(
                request=state["request"],
                text=state.get("text"),
                documents={group_key: state["documents"][group_key]},
                route=state.get("route"),
            )
            for group_key in state["documents"]
        ]
        return [Send(node=target, arg=state) for state in group_send_states]

    return _group_send


def build_document_chunk_send(target: str) -> Callable[[RedboxState], list[Send]]:
    def _chunk_send(state: RedboxState) -> list[Send]:
        chunk_send_states: list[RedboxState] = [
            RedboxState(
                request=state["request"],
                text=state.get("text"),
                documents={group_key: {document_key: state["documents"][group_key][document_key]}},
                route=state.get("route"),
            )
            for group_key in state["documents"]
            for document_key in state["documents"][group_key]
        ]
        return [Send(node=target, arg=state) for state in chunk_send_states]

    return _chunk_send
