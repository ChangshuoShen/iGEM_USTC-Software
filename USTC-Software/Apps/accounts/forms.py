from django import forms
from django.db import models

from django.contrib.auth.forms import UserCreationForm
from .models import User


class SignUpForm(models.Model):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2')
