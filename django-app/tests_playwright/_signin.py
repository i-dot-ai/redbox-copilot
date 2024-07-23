import os
import subprocess
from pathlib import Path

from _settings import BASE_URL
from playwright.sync_api import Page
from tests_playwright.pages import HomePage, LandingPage, SignInConfirmationPage
from yarl import URL

EMAIL_ADDRESS = os.environ["USER_EMAIL"]
DJANGO_ROOT = Path(__file__).parents[1]


def sign_in(page: Page) -> "HomePage":
    # Landing page
    landing_page = LandingPage(page)

    # Sign in
    sign_in_page = landing_page.navigate_to_sign_in()
    sign_in_page.email = EMAIL_ADDRESS
    sign_in_page.continue_()

    # Use magic link
    magic_link = get_magic_link(EMAIL_ADDRESS, DJANGO_ROOT)
    sign_in_confirmation_page = SignInConfirmationPage(page, magic_link)
    return sign_in_confirmation_page.navigate_to_documents_page()


def get_magic_link(email_address: str, django_root: Path) -> URL:
    command = ["poetry", "run", "python", "manage.py", "show_magiclink_url", email_address]
    result = subprocess.run(command, capture_output=True, text=True, cwd=django_root, check=True)  # noqa: S603
    magic_link = result.stdout.strip().lstrip("/")
    return BASE_URL / magic_link
