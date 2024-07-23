from _signin import sign_in
from playwright.sync_api import Page
from tests_playwright.pages import ChatsPage, FeedbackType


def test_response_feedback(page: Page):
    home_page = sign_in(page)

    chats_page: ChatsPage = home_page.navigate_to_chats()
    chats_page.write_message = "This is a test chat"
    chats_page = chats_page.send()

    assert not chats_page.check_feedback_prompt_visible(FeedbackType.HELPFUL)
    assert not chats_page.check_feedback_prompt_visible(FeedbackType.NOT_HELPFUL)

    chats_page.give_feedback(FeedbackType.HELPFUL)
    assert chats_page.check_feedback_prompt_visible(FeedbackType.HELPFUL)
    assert not chats_page.check_feedback_prompt_visible(FeedbackType.NOT_HELPFUL)

    chats_page.give_feedback(FeedbackType.NOT_HELPFUL)
    assert chats_page.check_feedback_prompt_visible(FeedbackType.NOT_HELPFUL)
    assert not chats_page.check_feedback_prompt_visible(FeedbackType.HELPFUL)

    chats_page.give_feedback(FeedbackType.NOT_HELPFUL)
    assert not chats_page.check_feedback_prompt_visible(FeedbackType.HELPFUL)
    assert not chats_page.check_feedback_prompt_visible(FeedbackType.NOT_HELPFUL)
