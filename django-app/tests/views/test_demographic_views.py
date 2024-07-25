import logging
from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from django.test import Client
from pytest_django.asserts import assertRedirects

from redbox_app.redbox_core.models import BusinessUnit, User

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_check_demographics_redirect_if_unpopulated(client: Client, alice: User):
    # Given
    client.force_login(alice)

    # When
    response = client.get("/check-demographics/", follow=True)

    # Then
    assertRedirects(response, "/demographics/")


@pytest.mark.django_db()
def test_check_demographics_redirect_if_populated(client: Client, user_with_demographic_data: User):
    # Given
    client.force_login(user_with_demographic_data)

    # When
    response = client.get("/check-demographics/", follow=True)

    # Then
    assertRedirects(response, "/chats/")


@pytest.mark.django_db()
def test_view_demographic_details_form(client: Client, user_with_demographic_data: User):
    # Given
    client.force_login(user_with_demographic_data)

    # When
    response = client.get("/demographics/")

    # Then
    assert response.status_code == HTTPStatus.OK
    soup = BeautifulSoup(response.content)
    assert soup.find(id="id_grade").find_all("option", selected=True)[0].text == "Director General"
    assert soup.find(id="id_profession").find_all("option", selected=True)[0].text == "Analysis"
    assert soup.find(id="id_business_unit").find_all("option", selected=True)[0].text == "Paperclip Reconciliation"


@pytest.mark.django_db()
def test_post_to_demographic_details_form(client: Client, alice: User, business_unit: BusinessUnit):
    # Given
    client.force_login(alice)

    # When
    response = client.post(
        "/demographics/",
        {
            "name": "Deryck Lennox-Brown",
            "ai_experience": "Enthusiastic Experimenter",
            "grade": "AO",
            "profession": "AN",
            "business_unit": business_unit.id,
        },
        follow=True,
    )

    # Then
    assertRedirects(response, "/chats/")
    alice.refresh_from_db()
    assert alice.grade == "AO"
