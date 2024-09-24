from django.urls import path
from . import views
app_name = 'SAM_Med2D'


urlpatterns = [
    path('sam_index/', views.sam_index, name='sam_index'),
]
