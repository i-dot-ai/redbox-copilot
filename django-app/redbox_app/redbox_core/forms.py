from collections.abc import Mapping
from typing import ClassVar

from django import forms

from redbox_app.redbox_core.models import User


class SignInForm(forms.Form):
    email = forms.EmailField(max_length=100)


class DemographicsForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("name", "ai_experience", "business_unit", "grade", "profession")
        labels: ClassVar[Mapping[str, str]] = {
            "name": "Full Name",
            "ai_experience": "How would you describe your level of experience with Generative AI tools?",
            "business_unit": "Business unit",
            "grade": "Grade",
            "profession": "Profession",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["business_unit"].queryset = self.fields["business_unit"].queryset.order_by("name")
