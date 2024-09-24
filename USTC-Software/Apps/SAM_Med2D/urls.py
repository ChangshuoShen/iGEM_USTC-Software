from django.urls import path
from . import views
app_name = 'SAM_Med2D'


urlpatterns = [
    path('signup_login/', views.signup_login, name='signup_login'),
    path('sam_index/', views.sam_index, name='sam_index')
    
]
