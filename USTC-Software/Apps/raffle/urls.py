from django.urls import path, include
from . import views

app_name = 'raffle'

urlpatterns = [
    path('draw/', views.draw, name='draw'),
]