from bs4 import BeautifulSoup
from django.conf import settings
from django.test import Client


def test_support_view_contains_contact_emai_and_version_number(client: Client):
    # Given
    version = settings.REDBOX_VERSION

    # When
    response = client.get("/support/")

    # Then
    soup = BeautifulSoup(response.content)
    mailto_links = [
        a.get("href", "").removeprefix("mailto:") for a in soup.find_all("a") if a.get("href", "").startswith("mailto:")
    ]
    assert mailto_links
    assert version in str(response.content)
