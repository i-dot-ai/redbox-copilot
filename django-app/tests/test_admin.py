import csv
import io
import logging
from http import HTTPStatus
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from django.test import Client
from django.urls import reverse
from yarl import URL

from redbox_app.redbox_core.models import ChatMessage, User

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_chat_history_export(superuser: User, chat_message_with_rating: ChatMessage, client: Client):
    # Given
    client.force_login(superuser)

    # When

    # See https://docs.djangoproject.com/en/dev/ref/contrib/admin/#reversing-admin-urls
    url = reverse("admin:redbox_core_chathistory_changelist")
    data = {"action": "export_as_csv", "_selected_action": [chat_message_with_rating.chat_history.pk]}
    response = client.post(url, data, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8"))))
    assert len(rows) == 1
    row = rows[0]
    assert row["history_name"] == "A chat"
    assert row["history_users"] == "alice@cabinetoffice.gov.uk"
    assert row["message_text"] == "A question?"
    assert row["rating_rating"] == "3"
    assert row["rating_chips"] == "speed, accuracy, blasphemy"


@pytest.mark.django_db()
def test_chat_history_export_without_ratings(superuser: User, chat_message: ChatMessage, client: Client):
    # Given
    client.force_login(superuser)

    # When
    url = reverse("admin:redbox_core_chathistory_changelist")
    data = {"action": "export_as_csv", "_selected_action": [chat_message.chat_history.pk]}
    response = client.post(url, data, follow=True)

    # Then
    assert response.status_code == HTTPStatus.OK
    rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8"))))
    assert len(rows) == 1
    row = rows[0]
    assert row["history_name"] == "A chat"
    assert row["history_users"] == "alice@cabinetoffice.gov.uk"
    assert row["message_text"] == "A question?"
    assert row["rating_rating"] is None
    assert row["rating_chips"] is None


@pytest.mark.django_db()
def test_user_upload(superuser: User, client: Client):
    # Given
    client.force_login(superuser)
    import_file = Path(__file__).parent / "data" / "csv" / "users.csv"

    # When
    url = URL(reverse("admin:redbox_core_user_changelist")) / "import/"
    with import_file.open("rb") as f:
        response = client.post(str(url), {"import_file": f, "format": "0"}, follow=False)

    soup = BeautifulSoup(response.content)
    confirm_form = response.context["confirm_form"]
    action_url = soup.find(id="content").find("form").get("action")

    response = client.post(action_url, confirm_form.initial, follow=True)
    soup = BeautifulSoup(response.content)

    # Then
    assert HTTPStatus(response.status_code).is_success
    assert "3 new, 0 updated, 0 deleted and 0 skipped" in soup.find("li", {"class": "success"}).text
    users = User.objects.all()
    assert len(users) == 4
