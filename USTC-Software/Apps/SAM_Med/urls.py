from django.urls import path
from . import views
app_name = 'SAM_Med'

urlpatterns = [
    path('', views.sam_index_2d, name='sam_index'),
    path('2d/', views.sam_index_2d, name='2d'),
    path('3d/', views.sam_index_3d, name='3d'),
    path('upload2d/', views.upload_image2d, name='upload_2d'),
    path('upload3d/', views.upload_image3d, name='upload_3d'),
]
