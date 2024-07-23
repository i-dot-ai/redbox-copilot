import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import BaseCommand

USER = get_user_model()
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


class Command(BaseCommand):
    help = """This should be run once per environment to set the initial superuser.
    Thereafter the superuser should be able to login and assign new staff users via the admin.
    """

    def handle(self, *_args, **_kwargs):
        if email := settings.SUPERUSER_EMAIL:
            user, created = USER.objects.get_or_create(email=email)
            user.is_superuser = True
            user.is_staff = True
            user.save()
            if created:
                logger.info("created superuser")
            return
