# Automated Accessibility Testing using Playwright and Axe

The tests are currently located at `django-app/tests_playwright`. All commands below should be run from this directory.

## Setup

`poetry run playwright install`

Add `USER_EMAIL` to your `.env` to allow for logging in. This must be the email address of the user you created for signing in via `createsuperuser` or otherwise.

The tests assume you are running the Django app at http://localhost:8090. If this is not the case you can update the `BASE_URL` in `_settings.py`.

## Running tests

From `django-app/tests_playwright` directory:

`poetry run pytest -s`

Commandline options are documented at https://playwright.dev/python/docs/running-tests
