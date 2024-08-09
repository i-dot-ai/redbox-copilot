import logging
from http import HTTPStatus

from botocore.exceptions import BotoCoreError
from django.conf import settings
from django.core.management import BaseCommand
from requests.exceptions import RequestException

from redbox_app.redbox_core.client import CoreApiClient
from redbox_app.redbox_core.models import File, StatusEnum

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)


def remove_from_django(file: File):
    try:
        file.delete_from_s3()
    except BotoCoreError as e:
        if getattr(e, "text", None) and (
            e.response["Error"]["Code"] == str(HTTPStatus.NOT_FOUND) or e.response["Error"]["Code"] == "NoSuchKey"
        ):
            logger.exception("File %s does not exist in s3, marking as deleted", file, exc_info=e)
            file.status = StatusEnum.deleted
            file.save()
        else:
            logger.exception("S3 error deleting file %s, marking as errored", exc_info=e)
            file.status = StatusEnum.errored
            file.save()

    else:
        logger.info("File %s removed from S3; marking as deleted")
        file.status = StatusEnum.deleted
        file.save()


class Command(BaseCommand):
    help = """This should be run regularly per environment.
    It checks the progress of File uploading and updates the status in Django where necessary.
    """

    def handle(self, *_args, **_kwargs):
        self.stdout.write(self.style.NOTICE("Checking file status"))

        for file in File.objects.filter(status=StatusEnum.processing):
            logger.debug(
                "Chcking file object %s, status %s",
                file,
                file.status,
            )

            try:
                core_file_status_response = core_api.get_file_status(file.core_file_uuid, file.user)
            except RequestException as e:
                if getattr(e.response, "status_code", None) == HTTPStatus.NOT_FOUND:
                    logger.exception("File %s does not exist in core-api, removing from django", file, exc_info=e)
                    remove_from_django(file)

                else:
                    logger.exception(
                        "Error getting status from core_api for %s, setting status to errored", file, exc_info=e
                    )
                    file.status = StatusEnum.errored
                    file.save()

            else:
                file.update_status_from_core(status_label=core_file_status_response.processing_status)
