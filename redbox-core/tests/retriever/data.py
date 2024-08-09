from uuid import uuid4


from redbox.models.chain import RedboxQuery
from redbox.models.file import ChunkResolution
from redbox.test.data import TestData, generate_test_cases

ALL_CHUNKS_RETRIEVER_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(question="Irrelevant Question", file_uuids=[uuid4()], user_uuid=uuid4(), chat_history=[]),
            test_data=[TestData(8, 8000)],
            test_id="Successful Path",
        )
    ]
    for test_case in generator
]

PARAMETERISED_RETRIEVER_CASES = [
    test_case
    for generator in [
        generate_test_cases(
            query=RedboxQuery(question="Irrelevant Question", file_uuids=[uuid4()], user_uuid=uuid4(), chat_history=[]),
            test_data=[TestData(8, 8000, ChunkResolution.normal)],
            test_id="Successful Path",
        )
    ]
    for test_case in generator
]
