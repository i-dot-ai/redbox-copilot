import logging
from http import HTTPStatus

import pytest
from django.test import Client

from redbox_app.redbox_core.models import User

logger = logging.getLogger(__name__)


@pytest.mark.django_db()
def test_declaration_view_get(peter_rabbit: User, client: Client):
    client.force_login(peter_rabbit)
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK, response.status_code
