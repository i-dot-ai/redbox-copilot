import logging
from datetime import timedelta

from botocore.exceptions import BotoCoreError
from django.conf import settings
from django.core.management import BaseCommand
from django.db.models import Max
from django.utils import timezone
from requests.exceptions import RequestException

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.models import ChatHistory, File, StatusEnum

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)


class Command(BaseCommand):
    help = """This should be run daily per environment to remove expired data.
    It removes Files, ChatMessages and ChatHistories that have exceeded their expiry date.
    """

    def handle(self, *_args, **_kwargs):
        cutoff_date = timezone.now() - timedelta(seconds=settings.FILE_EXPIRY_IN_SECONDS)

        self.stdout.write(self.style.NOTICE(f"Deleting Files expired before {cutoff_date}"))
        counter = 0

        for file in File.objects.filter(last_referenced__lt=cutoff_date):
            logger.debug(
                "Deleting file object %s, last_referenced %s",
                file,
                file.last_referenced,
            )

            try:
                core_api.delete_file(file.core_file_uuid, file.user)
                file.delete_from_s3()

            except RequestException as e:
                logger.exception("Error deleting file object %s using core-api", file, exc_info=e)
                file.status = StatusEnum.errored
                file.save()
            except BotoCoreError as e:
                logger.exception("Error deleting file object %s from storage", file, exc_info=e)
                file.status = StatusEnum.errored
                file.save()
            else:
                file.status = StatusEnum.deleted
                file.save()
                logger.debug("File object %s deleted", file)

                counter += 1

        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {counter} file objects"))

        self.stdout.write(self.style.NOTICE(f"Deleting chats expired before {cutoff_date}"))
        chats_to_delete = ChatHistory.objects.annotate(last_modified_at=Max("chatmessage__modified_at")).filter(
            last_modified_at__lt=cutoff_date
        )
        counter = chats_to_delete.count()
        chats_to_delete.delete()

        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {counter} ChatHistory objects"))
