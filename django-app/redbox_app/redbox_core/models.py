import logging
import uuid
from datetime import UTC, datetime, timedelta

import boto3
from botocore.config import Config
from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.core import validators
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django_use_email_as_username.models import BaseUser, BaseUserManager
from jose import jwt
from yarl import URL

logger = logging.getLogger(__name__)


class UUIDPrimaryKeyBase(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Meta:
        abstract = True


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(editable=False, auto_now_add=True)
    modified_at = models.DateTimeField(editable=False, auto_now=True)

    class Meta:
        abstract = True
        ordering = ["created_at"]


def sanitise_string(string: str | None) -> str | None:
    """We are seeing NUL (0x00) characters in user entered fields, and also in document citations.
    We can't save these characters, so we need to sanitise them."""
    return string.replace("\x00", "\ufffd") if string else string


class BusinessUnit(UUIDPrimaryKeyBase):
    name = models.TextField(max_length=64, null=False, blank=False, unique=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name}"


class User(BaseUser, UUIDPrimaryKeyBase):
    class UserGrade(models.TextChoices):
        AA = "AA", _("AA")
        AO = "AO", _("AO")
        DEPUTY_DIRECTOR = "DD", _("Deputy Director")
        DIRECTOR = "D", _("Director")
        DIRECTOR_GENERAL = "DG", _("Director General")
        EO = "EO", _("EO")
        G6 = "G6", _("G6")
        G7 = "G7", _("G7")
        HEO = "HEO", _("HEO")
        PS = "PS", _("Permanent Secretary")
        SEO = "SEO", _("SEO")
        OT = "OT", _("Other")

    class Profession(models.TextChoices):
        AN = "AN", _("Analysis")
        CM = "CMC", _("Commercial")
        COM = "COM", _("Communications")
        CFIN = "CFIN", _("Corporate finance")
        CF = "CF", _("Counter fraud")
        DDT = "DDT", _("Digital, data and technology")
        EC = "EC", _("Economics")
        FIN = "FIN", _("Finance")
        FEDG = "FEDG", _("Fraud, error, debts and grants")
        HR = "HR", _("Human resources")
        IA = "IA", _("Intelligence analysis")
        IAUD = "IAUD", _("Internal audit")
        IT = "IT", _("International trade")
        KIM = "KIM", _("Knowledge and information management")
        LG = "LG", _("Legal")
        MD = "MD", _("Medical")
        OP = "OP", _("Occupational psychology")
        OD = "OD", _("Operational delivery")
        OR = "OR", _("Operational research")
        PL = "PL", _("Planning")
        PI = "PI", _("Planning inspection")
        POL = "POL", _("Policy")
        PD = "PD", _("Project delivery")
        PR = "PR", _("Property")
        SE = "SE", _("Science and engineering")
        SC = "SC", _("Security")
        SR = "SR", _("Social research")
        ST = "ST", _("Statistics")
        TX = "TX", _("Tax")
        VET = "VET", _("Veterinary")
        OT = "OT", _("Other")

    username = None
    verified = models.BooleanField(default=False, blank=True, null=True)
    invited_at = models.DateTimeField(default=None, blank=True, null=True)
    invite_accepted_at = models.DateTimeField(default=None, blank=True, null=True)
    last_token_sent_at = models.DateTimeField(editable=False, blank=True, null=True)
    password = models.CharField("password", max_length=128, blank=True, null=True)
    business_unit = models.ForeignKey(BusinessUnit, null=True, blank=True, on_delete=models.SET_NULL)
    grade = models.CharField(null=True, blank=True, max_length=3, choices=UserGrade)
    profession = models.CharField(null=True, blank=True, max_length=4, choices=Profession)
    objects = BaseUserManager()

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.email}"

    def save(self, *args, **kwargs):
        self.email = self.email.lower()
        super().save(*args, **kwargs)

    def get_bearer_token(self) -> str:
        """the bearer token expected by the core-api"""
        user_uuid = str(self.id)
        bearer_token = jwt.encode({"user_uuid": user_uuid}, key=settings.SECRET_KEY)
        return f"Bearer {bearer_token}"


class StatusEnum(models.TextChoices):
    uploaded = "uploaded"
    parsing = "parsing"
    chunking = "chunking"
    embedding = "embedding"
    indexing = "indexing"
    complete = "complete"
    unknown = "unknown"
    deleted = "deleted"
    errored = "errored"


class File(UUIDPrimaryKeyBase, TimeStampedModel):
    status = models.CharField(choices=StatusEnum.choices, null=False, blank=False)
    original_file = models.FileField(storage=settings.STORAGES["default"]["BACKEND"])
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_file_name = models.TextField(max_length=2048, blank=True, null=True)
    core_file_uuid = models.UUIDField(null=True)
    last_referenced = models.DateTimeField(blank=True, null=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.original_file_name} {self.user}"

    def save(self, *args, **kwargs):
        if not self.last_referenced:
            if self.created_at:
                #  Needed to populate the initial last_referenced field for existing Files
                self.last_referenced = self.created_at
            else:
                self.last_referenced = timezone.now()
        super().save(*args, **kwargs)

    def delete(self, using=None, keep_parents=False):  # noqa: ARG002  # remove at Python 3.12
        #  Needed to make sure no orphaned files remain in the storage
        self.original_file.storage.delete(self.original_file.name)
        super().delete()

    def delete_from_s3(self):
        """Manually deletes the file from S3 storage."""
        self.original_file.delete(save=False)

    @property
    def file_type(self) -> str:
        name = self.original_file.name
        return name.split(".")[-1]

    @property
    def url(self) -> URL | None:
        #  In dev environment, get pre-signed url from minio
        if settings.ENVIRONMENT.uses_minio:
            s3 = boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
                config=Config(signature_version="s3v4"),
                region_name=settings.AWS_S3_REGION_NAME,
            )

            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": settings.AWS_STORAGE_BUCKET_NAME,
                    "Key": self.name,
                },
            )
            return URL(url)

        if not self.original_file:
            logger.error("attempt to access not existent file %s", self.pk)
            return None

        return URL(self.original_file.url)

    @property
    def name(self) -> str:
        # User-facing name
        return self.original_file_name or self.original_file.name

    @property
    def unique_name(self) -> str:
        # Name used by core-api
        return self.original_file.file.name

    def get_status_text(self) -> str:
        return next(
            (status[1] for status in StatusEnum.choices if self.status == status[0]),
            "Unknown",
        )

    @property
    def expires_at(self) -> datetime:
        return self.last_referenced + timedelta(seconds=settings.FILE_EXPIRY_IN_SECONDS)

    @property
    def expires(self) -> timedelta:
        return self.expires_at - datetime.now(tz=UTC)

    def __lt__(self, other):
        return self.id < other.id


class ChatHistory(UUIDPrimaryKeyBase, TimeStampedModel):
    name = models.TextField(max_length=1024, null=False, blank=False)
    users = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name_plural = "Chat history"

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name} - {self.users}"

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        self.name = sanitise_string(self.name)
        super().save(force_insert, force_update, using, update_fields)


class ChatRoleEnum(models.TextChoices):
    ai = "ai"
    user = "user"
    system = "system"


class Citation(UUIDPrimaryKeyBase, TimeStampedModel):
    file = models.ForeignKey(File, on_delete=models.CASCADE)
    chat_message = models.ForeignKey("ChatMessage", on_delete=models.CASCADE)
    text = models.TextField(null=True, blank=True)
    page_numbers = ArrayField(
        models.PositiveIntegerField(), null=True, blank=True, help_text="location of citation in document"
    )

    def __str__(self):
        return f"{self.file}: {self.text or ''}"

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        self.text = sanitise_string(self.text)
        super().save(force_insert, force_update, using, update_fields)


class ChatMessage(UUIDPrimaryKeyBase, TimeStampedModel):
    chat_history = models.ForeignKey(ChatHistory, on_delete=models.CASCADE)
    text = models.TextField(max_length=32768, null=False, blank=False)
    role = models.CharField(choices=ChatRoleEnum.choices, null=False, blank=False)
    route = models.CharField(max_length=25, null=True, blank=True)
    selected_files = models.ManyToManyField(File, related_name="+", symmetrical=False, blank=True)
    source_files = models.ManyToManyField(File, through=Citation)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.text} - {self.role}"

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        self.text = sanitise_string(self.text)
        super().save(force_insert, force_update, using, update_fields)


class ChatMessageRating(TimeStampedModel):
    chat_message = models.OneToOneField(ChatMessage, on_delete=models.CASCADE, primary_key=True)
    rating = models.PositiveIntegerField(validators=[validators.MinValueValidator(1), validators.MaxValueValidator(5)])
    text = models.TextField(blank=True, null=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.chat_message} - {self.rating} - {self.text}"

    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        self.text = sanitise_string(self.text)
        super().save(force_insert, force_update, using, update_fields)


class ChatMessageRatingChip(UUIDPrimaryKeyBase, TimeStampedModel):
    rating = models.ForeignKey(ChatMessageRating, on_delete=models.CASCADE)
    text = models.CharField(max_length=32)

    class Meta:
        unique_together = "rating", "text"

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.rating} - {self.text}"
