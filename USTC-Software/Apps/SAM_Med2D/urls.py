from django.urls import path
from . import views
app_name = 'SAM_Med2D'


urlpatterns = [
    path('', views.sam_index, name='sam_index'),
    path('upload/', views.upload_image, name='upload'),
]
