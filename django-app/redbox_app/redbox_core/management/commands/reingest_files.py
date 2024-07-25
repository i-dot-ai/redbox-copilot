import logging

from django.conf import settings
from django.core.management import BaseCommand
from requests.exceptions import RequestException

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.models import INACTIVE_STATUSES, File, StatusEnum

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)


class Command(BaseCommand):
    help = """This is an ad-hoc command when changes to the AI pipeline (e.g. a new embedding strategy)
    mean we need to regenerate chunks for all the current files.

    It attempts to reupload file data to core-api for reingestion."""

    def handle(self, *_args, **_kwargs):
        self.stdout.write(self.style.NOTICE("Reingesting active files from Django"))
        successes, errors = 0, 0

        for file in File.objects.exclude(status__in=INACTIVE_STATUSES):
            logger.debug("Reingesting file object %s", file)

            try:
                core_api.reingest_file(file.core_file_uuid, file.user)

            except RequestException as e:
                logger.exception("Error reingesting file object %s using core-api", file, exc_info=e)
                file.status = StatusEnum.errored
                file.save()
                errors += 1

            else:
                file.status = StatusEnum.uploaded
                file.save()
                successes += 1

        self.stdout.write(
            self.style.NOTICE(f"Successfully reuploaded {successes} files and failed to reupload {errors} files")
        )
