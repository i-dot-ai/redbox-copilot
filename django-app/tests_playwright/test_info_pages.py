from playwright.sync_api import Page
from tests_playwright.pages import LandingPage


def test_support_pages(page: Page):
    # Landing page
    landing_page = LandingPage(page)

    # Privacy page
    landing_page.navigate_to_privacy_page()

    # Accessibility page
    landing_page.navigate_to_accessibility_page()

    # Support page
    landing_page.navigate_to_support_page()
