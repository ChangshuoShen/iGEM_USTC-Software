from django.urls import path
from . import views

app_name = 'feedback'

urlpatterns = [ 
    path('feedback_page/', views.feedback_page, name='feedback_page'),
    path('send_feedback/', views.send_feedback, name='send_feedback'),
]

