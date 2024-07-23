from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string

EMAIL_MAPPING = {
    "magic_link": {
        "subject": "Redbox sign-in",
        "template_name": "email/verification.txt",
    },
}


def _send_email(subject: str, template_name: str, to_address: str, context: dict):
    body = render_to_string(template_name, context)
    return send_mail(
        subject=subject,
        message=body,
        from_email=settings.FROM_EMAIL,
        recipient_list=[to_address],
    )


def send_magic_link_email(magic_link: str, to_address: str):
    magic_link_email_values = EMAIL_MAPPING["magic_link"]
    _send_email(
        magic_link_email_values["subject"],
        magic_link_email_values["template_name"],
        to_address,
        context={"url": magic_link},
    )
