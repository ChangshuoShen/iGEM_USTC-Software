from django.urls import path
import Apps.accounts.views as views

urlpatterns = [
    path('login/', views.login),
]

