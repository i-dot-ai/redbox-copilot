import os
import time

import pytest
import requests


def test_upload_to_elastic(file_pdf_path):
    """
    When I POST a file to core-api/file
    I Expect a Chunk with a non-null embedding ... eventually
    """

    file_name = os.path.basename(file_pdf_path)
    files = {"file": (file_name, open(file_pdf_path, "rb"), "application/pdf")}
    response = requests.post(url="http://localhost:5002/file", files=files)
    if response.status_code == 200:
        file_uuid = response.json()["uuid"]

        timeout = 120
        start_time = time.time()
        chunks = []

        while not any(chunk["embedding"] for chunk in chunks) and time.time() - start_time < timeout:
            time.sleep(1)
            chunks = requests.get(f"http://localhost:5002/file/{file_uuid}/chunks").json()

        pytest.fail(reason=f"failed to get embedded chunks within {timeout} seconds")
    else:
        print(response.text)
