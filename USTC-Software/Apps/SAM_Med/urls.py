from django.urls import path
from . import views
app_name = 'SAM_Med'


urlpatterns = [
    path('', views.sam_index, name='sam_index'),
    path('2d/', views.sam_index, name='2d_to_upload'),
    path('3d/', views.sam_index, name='3d_to_upload'),
    path('upload2d/', views.upload_image2d, name='upload_2d'),
    path('upload_3d', views.upload_image2d, name='upload_3d'),
]
