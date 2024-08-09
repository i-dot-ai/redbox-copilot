from uuid import uuid4
import pytest

from langchain_core.documents import Document

from redbox.models.chain import document_reducer


GROUP_IDS = [uuid4() for i in range(4)]
DOCUMENT_IDS = [uuid4() for i in range(10)]


@pytest.mark.parametrize(
    argnames="a,b,expected",
    ids=[
        "Clear a document",
        "Clear a group",
        "Add a document",
        "Add a group",
        "Add and clear documents in one group",
        "Add and clear a group",
        "Add and clear documents across multiple groups",
    ],
    argvalues=[
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {
                GROUP_IDS[0]: {
                    DOCUMENT_IDS[1]: None,
                }
            },
            {
                GROUP_IDS[0]: {
                    DOCUMENT_IDS[0]: Document("a"),
                },
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
        ),
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {GROUP_IDS[0]: None},
            {GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")}},
        ),
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {
                GROUP_IDS[0]: {
                    DOCUMENT_IDS[4]: Document("e"),
                }
            },
            {
                GROUP_IDS[0]: {
                    DOCUMENT_IDS[0]: Document("a"),
                    DOCUMENT_IDS[1]: Document("b"),
                    DOCUMENT_IDS[4]: Document("e"),
                },
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
        ),
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {GROUP_IDS[0]: {DOCUMENT_IDS[1]: None, DOCUMENT_IDS[4]: Document("e")}},
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[4]: Document("e")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
        ),
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {GROUP_IDS[2]: {DOCUMENT_IDS[4]: Document("e"), DOCUMENT_IDS[5]: Document("f")}},
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
                GROUP_IDS[2]: {DOCUMENT_IDS[4]: Document("e"), DOCUMENT_IDS[5]: Document("f")},
            },
        ),
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {GROUP_IDS[0]: None, GROUP_IDS[2]: {DOCUMENT_IDS[4]: Document("e"), DOCUMENT_IDS[5]: Document("f")}},
            {
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
                GROUP_IDS[2]: {DOCUMENT_IDS[4]: Document("e"), DOCUMENT_IDS[5]: Document("f")},
            },
        ),
        (
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: Document("a"), DOCUMENT_IDS[1]: Document("b")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: Document("c"), DOCUMENT_IDS[3]: Document("d")},
            },
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[0]: None, DOCUMENT_IDS[6]: Document("g")},
                GROUP_IDS[1]: {DOCUMENT_IDS[2]: None, DOCUMENT_IDS[3]: None, DOCUMENT_IDS[7]: Document("h")},
            },
            {
                GROUP_IDS[0]: {DOCUMENT_IDS[1]: Document("b"), DOCUMENT_IDS[6]: Document("g")},
                GROUP_IDS[1]: {DOCUMENT_IDS[7]: Document("h")},
            },
        ),
    ],
)
def test_document_reducer(a, b, expected):
    result = document_reducer(a, b)
    assert result == expected, f"Expected: {expected}. Result: {result}"
