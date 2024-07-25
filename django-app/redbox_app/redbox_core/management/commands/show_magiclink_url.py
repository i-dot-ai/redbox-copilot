import logging
from argparse import ArgumentParser

from django.core.management import BaseCommand, CommandError
from django.db.models import Max
from magic_link.models import MagicLink

from redbox_app.redbox_core.models import User

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = """Show live MagicLink URL for given user."""

    def add_arguments(self, parser: ArgumentParser):
        parser.add_argument("user_email", type=str)

    def handle(self, **options):
        user_email = options["user_email"]
        logger.debug("user email: %s", user_email)

        try:
            user: User = User.objects.get(email=user_email)
        except User.DoesNotExist as e:
            message = f"No User found with email {user_email}"
            raise CommandError(message) from e

        try:
            latest = MagicLink.objects.filter(user=user).aggregate(Max("created_at"))["created_at__max"]
            logger.debug("latest: %s", latest)
            link: MagicLink = MagicLink.objects.get(user=user, created_at=latest)
        except MagicLink.DoesNotExist as e:
            message = f"No MagicLink found for user {user.email}"
            raise CommandError(message) from e

        logger.debug("link: %s", link)
        if link.is_valid:
            self.stdout.write(self.style.SUCCESS(link.get_absolute_url()))
        else:
            message = f"No active link for user {user.email}"
            raise CommandError(message)
