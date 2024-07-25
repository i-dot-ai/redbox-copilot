import datetime

import humanize
import jinja2
import pytz
from compressor.contrib.jinja2ext import CompressorExtension
from django.conf import settings
from django.templatetags.static import static
from django.urls import reverse
from django.utils.timezone import template_localtime
from markdown_it import MarkdownIt

# `js-default` setting required to sanitize inputs
# https://markdown-it-py.readthedocs.io/en/latest/security.html
markdown_converter = MarkdownIt("js-default")


def url(path, *args, **kwargs):
    if args and kwargs:
        message = "Use *args or **kwargs, not both."
        raise ValueError(message)
    return reverse(path, args=args, kwargs=kwargs)


def markdown(text, cls=None):
    """
    Converts the given text into markdown.
    The `replace` statement replaces the outer <p> tag with one that contains the given class, otherwise the markdown
    ends up double wrapped with <p> tags.
    Args:
        text: The text to convert to markdown
        cls (optional): The class to apply to the outermost <p> tag surrounding the markdown

    Returns:
        Text converted to markdown
    """
    html = markdown_converter.render(text).strip()
    return html.replace("<p>", f'<p class="{cls or ""}">', 1).replace("</p>", "", 1)


def humanise_expiry(delta: datetime.timedelta) -> str:
    if delta.total_seconds() > 0:
        return f"{humanize.naturaldelta(delta)}"
    else:
        return f"{humanize.naturaldelta(delta)} ago"


def humanize_timedelta(delta: datetime.timedelta):
    return humanize.naturaldelta(delta)


def humanize_short_timedelta(minutes=0, hours_limit=200, too_large_msg=""):
    if minutes > (hours_limit * 60):
        if not too_large_msg:
            return f"More than {hours_limit} hours"
        else:
            return too_large_msg
    else:
        delta = datetime.timedelta(minutes=minutes)
        return humanize.precisedelta(delta, minimum_unit="minutes")


def to_user_timezone(value):
    # Assuming the user's timezone is stored in a variable called 'user_timezone'
    # Replace 'Europe/London' with the actual timezone string for the user
    user_tz = pytz.timezone("Europe/London")
    return value.astimezone(user_tz).strftime("%H:%M %d/%m/%Y")


def environment(**options):
    extra_options = {}
    env = jinja2.Environment(  # nosec: B701 # noqa: S701
        **{
            "autoescape": True,
            "extensions": [CompressorExtension],
            **options,
            **extra_options,
        },
    )
    env.filters.update(
        {
            "static": static,
            "url": url,
            "humanise_expiry": humanise_expiry,
            "template_localtime": template_localtime,
            "to_user_timezone": to_user_timezone,
            "environment": settings.ENVIRONMENT.value,
            "security": settings.MAX_SECURITY_CLASSIFICATION.value,
        }
    )
    env.globals.update(
        {
            "static": static,
            "url": url,
            "humanise_expiry": humanise_expiry,
            "template_localtime": template_localtime,
            "to_user_timezone": to_user_timezone,
            "environment": settings.ENVIRONMENT.value,
            "security": settings.MAX_SECURITY_CLASSIFICATION.value,
        }
    )
    return env
