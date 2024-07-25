import logging

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import UpdateView

from redbox_app.redbox_core.forms import DemographicsForm
from redbox_app.redbox_core.models import User

logger = logging.getLogger(__name__)


class CheckDemographicsView(View):
    @method_decorator(login_required)
    def get(self, request: HttpRequest) -> HttpResponse:
        user: User = request.user
        if all([user.name, user.ai_experience, user.grade, user.business_unit, user.profession]):
            return redirect("chats")
        else:
            return redirect("demographics")


class DemographicsView(UpdateView):
    model = User
    template_name = "demographics.html"
    form_class = DemographicsForm
    success_url = "/chats/"

    def get_object(self, **kwargs):  # noqa: ARG002
        return self.request.user
