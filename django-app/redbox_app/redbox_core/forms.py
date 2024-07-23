from collections.abc import Mapping
from typing import ClassVar

from django import forms

from redbox_app.redbox_core.models import User


class SignInForm(forms.Form):
    email = forms.EmailField(max_length=100)


class DemographicsForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("business_unit", "grade", "profession")
        labels: ClassVar[Mapping[str, str]] = {
            "business_unit": "Business unit",
            "grade": "Grade",
            "profession": "Profession",
        }
        widgets: ClassVar[Mapping[str, forms.Widget]] = {
            "business_unit": forms.Select(attrs={"class": "govuk-select"}),
            "grade": forms.Select(attrs={"class": "govuk-select"}),
            "profession": forms.Select(attrs={"class": "govuk-select"}),
        }
