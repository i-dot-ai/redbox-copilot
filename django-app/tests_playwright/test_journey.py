import logging
import os
import subprocess
from pathlib import Path

from _settings import BASE_URL
from playwright.sync_api import Page
from tests_playwright.pages import LandingPage, SignInConfirmationPage
from yarl import URL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

EMAIL_ADDRESS = os.environ["USER_EMAIL"]
DJANGO_ROOT = Path(__file__).parents[1]


def test_user_journey(page: Page):
    # Landing page
    landing_page = LandingPage(page)

    # Sign in
    sign_in_page = landing_page.navigate_to_sign_in()
    sign_in_page.email = EMAIL_ADDRESS
    sign_in_page.continue_()

    # Use magic link
    magic_link = get_magic_link(EMAIL_ADDRESS, DJANGO_ROOT)
    sign_in_confirmation_page = SignInConfirmationPage(page, magic_link)

    # Documents page
    documents_page = sign_in_confirmation_page.navigate_to_documents_page()
    document_rows = documents_page.get_all_document_rows()
    original_docs_count = len(document_rows)

    # Upload a file
    document_upload_page = documents_page.navigate_to_upload()
    upload_file = DJANGO_ROOT / "files" / "RiskTriggersReport361.pdf"
    documents_page = document_upload_page.upload_document(upload_file)
    document_rows = documents_page.get_all_document_rows()
    logger.debug("document_rows: %s", document_rows)
    assert any(row.filename == upload_file.name for row in document_rows)
    assert len(document_rows) == original_docs_count + 1

    # Delete a file
    document_delete_page = documents_page.delete_latest_document()
    documents_page = document_delete_page.confirm_deletion()
    document_rows = documents_page.get_all_document_rows()
    assert len(document_rows) == original_docs_count

    # Chats page
    chats_page = documents_page.navigate_to_chats()
    chats_page.write_message = "Who put the bomp in the bomp bah bomp bah bomp?"
    chats_page = chats_page.send()
    all_messages = chats_page.all_messages()
    logger.debug("page: %s", chats_page)
    logger.debug("all_messages: %s", all_messages)


def get_magic_link(email_address: str, django_root: Path) -> URL:
    command = ["poetry", "run", "python", "manage.py", "show_magiclink_url", email_address]
    result = subprocess.run(command, capture_output=True, text=True, cwd=django_root, check=True)  # noqa: S603
    magic_link = result.stdout.strip().lstrip("/")
    return BASE_URL / magic_link
