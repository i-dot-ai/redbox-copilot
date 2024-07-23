import logging

from django.contrib.auth import logout
from django.http import HttpRequest
from django.shortcuts import redirect, render
from magic_link.models import MagicLink
from requests import HTTPError

from redbox_app.redbox_core import email_handler, models
from redbox_app.redbox_core.forms import SignInForm

logger = logging.getLogger(__name__)


def sign_in_view(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("homepage")
    if request.method == "POST":
        form = SignInForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"].lower()

            try:
                user = models.User.objects.get(email=email)
                link = MagicLink.objects.create(user=user, redirect_to="/check-demographics")
                full_link = request.build_absolute_uri(link.get_absolute_url())
                email_handler.send_magic_link_email(full_link, email)
            except models.User.DoesNotExist as e:
                logger.debug("User with email %s not found", email, exc_info=e)
            except HTTPError as e:
                logger.exception("failed to send link to %s", email, exc_info=e)

            return redirect("sign-in-link-sent")

        return render(
            request,
            template_name="sign-in.html",
            context={
                "errors": form.errors,
            },
        )

    return render(
        request,
        template_name="sign-in.html",
        context={"request": request},
    )


def sign_in_link_sent_view(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("homepage")
    return render(
        request,
        template_name="sign-in-link-sent.html",
        context={"request": request},
    )


def signed_out_view(request: HttpRequest):
    logout(request)
    return redirect("homepage")
