import csv
import logging

from django.conf import settings
from django.contrib import admin
from django.db.models import QuerySet
from django.http import HttpResponse
from import_export.admin import ImportMixin
from requests.exceptions import RequestException

from redbox_app.redbox_core.client import CoreApiClient

from . import models

logger = logging.getLogger(__name__)
core_api = CoreApiClient(host=settings.CORE_API_HOST, port=settings.CORE_API_PORT)


class UserAdmin(ImportMixin, admin.ModelAdmin):
    fields = ["email", "business_unit", "grade", "profession", "is_superuser", "is_staff", "last_login"]
    list_display = ["email", "business_unit", "grade", "profession", "is_superuser", "is_staff", "last_login"]
    list_filter = ["business_unit", "grade", "profession"]
    date_hierarchy = "last_login"

    class Meta:
        model = models.User
        fields = ["email"]
        import_id_fields = ["email"]


class BusinessUnitAdmin(ImportMixin, admin.ModelAdmin):
    fields = ["name"]
    list_display = ["name"]

    class Meta:
        model = models.BusinessUnit
        fields = ["name"]
        import_id_fields = ["name"]


class FileAdmin(admin.ModelAdmin):
    def reupload(self, request, queryset):  # noqa:ARG002
        for file in queryset:
            try:
                logger.info("Re-uploading file to core-api: %s", file)
                core_api.reingest_file(file.core_file_uuid, file.user)
            except RequestException as e:
                logger.exception("Error re-uploading File model object %s.", file, exc_info=e)
                file.status = models.StatusEnum.errored
                file.save()
            else:
                file.status = models.StatusEnum.uploaded
                file.save()

                logger.info("Successfully reuploaded file %s.", file)

    list_display = ["original_file_name", "user", "status", "created_at", "last_referenced"]
    list_filter = ["user", "status"]
    date_hierarchy = "created_at"
    actions = ["reupload"]


class CitationInline(admin.StackedInline):
    model = models.Citation
    ordering = ("modified_at",)

    extra = 1


class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["text", "role", "get_user", "chat_history", "route", "created_at"]
    list_filter = ["role", "route", "chat_history__users"]
    date_hierarchy = "created_at"
    inlines = [CitationInline]

    @admin.display(ordering="chat_history__users", description="User")
    def get_user(self, obj):
        return obj.chat_history.users


class ChatMessageInline(admin.StackedInline):
    model = models.ChatMessage
    ordering = ("modified_at",)
    readonly_fields = ["modified_at", "source_files"]
    extra = 1
    show_change_link = True  # allows users to click through to look at Citations


class ChatHistoryAdmin(admin.ModelAdmin):
    def export_as_csv(self, request, queryset: QuerySet):  # noqa:ARG002
        history_field_names: list[str] = [field.name for field in models.ChatHistory._meta.fields]  # noqa:SLF001
        message_field_names: list[str] = [field.name for field in models.ChatMessage._meta.fields]  # noqa:SLF001
        rating_field_names: list[str] = [field.name for field in models.ChatMessageRating._meta.fields]  # noqa:SLF001

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=chathistory.csv"
        writer = csv.writer(response)

        writer.writerow(
            ["history_" + n for n in history_field_names]
            + ["message_" + n for n in message_field_names]
            + ["rating_" + n for n in rating_field_names]
            + ["rating_chips"]
        )
        chat_history: models.ChatHistory
        chat_message: models.ChatMessage
        chat_message_rating: models.ChatMessageRating
        for chat_history in queryset:
            for chat_message in chat_history.chatmessage_set.all():
                row = [getattr(chat_history, field) for field in history_field_names] + [
                    getattr(chat_message, field) for field in message_field_names
                ]
                if hasattr(chat_message, "chatmessagerating"):
                    chat_message_rating = chat_message.chatmessagerating
                    row += [getattr(chat_message_rating, field) for field in rating_field_names]
                    row += [", ".join(c.text for c in chat_message_rating.chatmessageratingchip_set.all())]
                writer.writerow(row)

        return response

    export_as_csv.short_description = "Export Selected"
    fields = ["name", "users"]
    inlines = [ChatMessageInline]
    list_display = ["name", "users", "created_at"]
    list_filter = ["users"]
    date_hierarchy = "created_at"
    actions = ["export_as_csv"]


class CitationAdmin(admin.ModelAdmin):
    list_display = ["text", "get_user", "chat_message", "file"]
    list_filter = ["chat_message__chat_history__users"]

    @admin.display(ordering="chat_message__chat_history__users", description="User")
    def get_user(self, obj):
        return obj.chat_message.chat_history.users


admin.site.register(models.User, UserAdmin)
admin.site.register(models.File, FileAdmin)
admin.site.register(models.ChatHistory, ChatHistoryAdmin)
admin.site.register(models.ChatMessage, ChatMessageAdmin)
admin.site.register(models.ChatMessageRating)
admin.site.register(models.ChatMessageRatingChip)
admin.site.register(models.Citation, CitationAdmin)
admin.site.register(models.BusinessUnit, BusinessUnitAdmin)
