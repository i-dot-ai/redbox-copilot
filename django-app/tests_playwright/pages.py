import logging
from abc import ABC, abstractmethod
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from _settings import BASE_URL
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import Page, expect
from yarl import URL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BasePage(ABC):
    # All available rules/categories can be found at https://github.com/dequelabs/axe-core/blob/develop/doc/rule-descriptions.md
    # Can't include all as gov.uk design system violates the "region" rule
    AXE_OPTIONS: ClassVar[dict[str, Any]] = {
        "runOnly": {
            "type": "tag",
            "values": [
                "wcag2a",
                "wcag2aa",
                "wcag21aa",
                "wcag22aa",
                "cat.aria",
                "cat.name-role-value",
                "cat.structure",
                "cat.semantics",
                "cat.text-alternatives",
                "cat.forms",
                "cat.sensory-and-visual-cues",
                "cat.tables",
                "cat.time-and-media",
            ],
        }
    }

    def __init__(self, page: Page):
        self.page = page
        self.axe = Axe()
        self.check_title()
        self.check_a11y()

    def check_title(self):
        expected_page_title = self.get_expected_page_title()
        expect(self.page).to_have_title(expected_page_title)
        # expect(self.page).to_have_url("url")

    def check_a11y(self):
        results = self.axe.run(self.page, context=None, options=self.AXE_OPTIONS)
        assert results.violations_count == 0, f"{self.url} - {results.generate_report()}"

    def navigate_to_privacy_page(self) -> "PrivacyPage":
        self.page.get_by_role("link", name="Privacy", exact=True).click()
        return PrivacyPage(self.page)

    def navigate_to_accessibility_page(self) -> "AccessibilityPage":
        self.page.get_by_role("link", name="Accessibility", exact=True).click()
        return AccessibilityPage(self.page)

    def navigate_to_support_page(self) -> "SupportPage":
        self.page.get_by_role("link", name="Support", exact=True).click()
        return SupportPage(self.page)

    @abstractmethod
    def get_expected_page_title(self) -> str: ...

    @property
    def title(self) -> str:
        return self.page.title()

    @property
    def url(self) -> URL:
        return URL(self.page.url)

    def __str__(self) -> str:
        return f'"{self.title}" at {self.url}'


class SignedInBasePage(BasePage, ABC):
    def navigate_to_documents(self) -> "DocumentsPage":
        self.page.get_by_role("link", name="Documents", exact=True).click()
        return DocumentsPage(self.page)

    def navigate_to_chats(self) -> "ChatsPage":
        self.page.get_by_role("link", name="Chats", exact=True).click()
        return ChatsPage(self.page)

    def sign_out(self) -> "LandingPage":
        self.page.get_by_role("link", name="Chats", exact=True).click()
        return LandingPage(self.page)


class LandingPage(BasePage):
    def __init__(self, page):
        page.goto(str(BASE_URL))
        super().__init__(page)

    def get_expected_page_title(self) -> str:
        return "Redbox"

    def navigate_to_sign_in(self) -> "SignInPage":
        self.page.get_by_role("link", name="Sign in", exact=True).click()
        return SignInPage(self.page)


class SignInPage(BasePage):
    def get_expected_page_title(self) -> str:
        return "Sign in - Redbox"

    @property
    def email(self) -> str:
        return self.page.locator("#email").input_value()

    @email.setter
    def email(self, value: str):
        self.page.locator("#email").fill(value)

    def continue_(self) -> "SignInLinkSentPage":
        self.page.get_by_text("Continue").click()
        return SignInLinkSentPage(self.page)


class SignInLinkSentPage(BasePage):
    def get_expected_page_title(self) -> str:
        return "Sign in - link sent - Redbox"


class SignInConfirmationPage(BasePage):
    def __init__(self, page, magic_link: URL):
        page.goto(str(magic_link))
        super().__init__(page)

    def get_expected_page_title(self) -> str:
        return "Sign in - confirmation - Redbox"

    def navigate_to_documents_page(self) -> "DocumentsPage":
        self.page.get_by_role("button", name="Start", exact=True).click()
        return DocumentsPage(self.page)


class HomePage(SignedInBasePage):
    def get_expected_page_title(self) -> str:
        return "Redbox"


class DocumentRow(NamedTuple):
    filename: str
    status: str


class DocumentsPage(SignedInBasePage):
    def get_expected_page_title(self) -> str:
        return "Documents - Redbox"

    def delete_latest_document(self) -> "DocumentDeletePage":
        self.page.get_by_role("button", name="Remove").first.click()
        return DocumentDeletePage(self.page)

    def navigate_to_upload(self) -> "DocumentUploadPage":
        self.page.get_by_role("button", name="Add document").click()
        return DocumentUploadPage(self.page)

    def get_all_document_rows(self) -> list[DocumentRow]:
        cell_texts = self.page.get_by_role("cell").all_inner_texts()
        return [DocumentRow(filename, status) for filename, uploaded_at, status, action in batched(cell_texts, 4)]


class DocumentDeletePage(SignedInBasePage):
    def get_expected_page_title(self) -> str:
        return "Remove document - Redbox"

    def confirm_deletion(self) -> "DocumentsPage":
        self.page.get_by_role("button", name="Remove").click()
        return DocumentsPage(self.page)


class DocumentUploadPage(SignedInBasePage):
    def get_expected_page_title(self) -> str:
        return "Upload a document - Redbox"

    def upload_document(self, upload_file: Path) -> DocumentsPage:
        self.get_file_chooser_by_label().set_files(upload_file)
        self.page.get_by_role("button", name="Upload").click()
        return self.navigate_to_documents()

    def get_file_chooser_by_label(self):
        with self.page.expect_file_chooser() as fc_info:
            self.page.get_by_label("Upload a document").click()
        return fc_info.value


class FeedbackType(Enum):
    HELPFUL = "Helpful"
    NOT_HELPFUL = "Not helpful"


class ChatsPage(SignedInBasePage):
    def get_expected_page_title(self) -> str:
        return "Chats - Redbox"

    @property
    def write_message(self) -> str:
        return self.page.locator("#message").input_value()

    @write_message.setter
    def write_message(self, value: str):
        self.page.locator("#message").fill(value)

    def send(self) -> "ChatsPage":
        self.page.get_by_text("Send").click()
        return ChatsPage(self.page)

    def all_messages(self) -> list[str]:
        return self.page.locator(".rb-chat-message").all_inner_texts()

    def check_feedback_prompt_visible(self, feedback: FeedbackType) -> bool:
        if feedback == FeedbackType.NOT_HELPFUL:
            return self.page.get_by_text("Can you let me know what wasn’t accurate?").is_visible()  # noqa: RUF001
        return self.page.get_by_text("Thank you for your feedback").first.is_visible()

    def give_feedback(self, feedback: FeedbackType):
        self.page.get_by_role("button", name=feedback.value, exact=True).click()


class PrivacyPage(BasePage):
    def get_expected_page_title(self) -> str:
        return "Privacy notice - Redbox"


class AccessibilityPage(BasePage):
    def get_expected_page_title(self) -> str:
        return "Accessibility statement - Redbox"


class SupportPage(BasePage):
    def get_expected_page_title(self) -> str:
        return "Support - Redbox"


def batched(iterable, n):
    # TODO (@brunns): Use library version when we upgrade to Python 3.12.
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    if n < 1:
        message = "n must be at least one"
        raise ValueError(message)
    iterable = iter(iterable)
    while batch := tuple(islice(iterable, n)):
        yield batch
