from django.urls import path
from . import views
app_name = 'SAM_Med'


urlpatterns = [
    path('', views.sam_index, name='sam_index'),
    path('2d/', views.sam_index, name='2d_to_upload'),
    path('3d/', views.sam_index, name='3d_to_upload'),
    path('upload/', views.upload_image, name='upload'),
]
