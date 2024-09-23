from django.urls import path
from . import views


app_name = 'works'
urlpatterns = [
    path('jupyter/', views.jupyter, name='jupyter'),
    
]