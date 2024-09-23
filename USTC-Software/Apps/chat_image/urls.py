from django.urls import path
from . import views
app_name = 'chat_image'


urlpatterns = [
    path('', views.FakeTextToImageView.as_view()),
]
