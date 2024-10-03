from django.urls import path
from . import views
app_name = 'rna_seq'


urlpatterns = [
    path('', views.rna_index, name='rna_seq_index'),
    # path('upload/', views.upload_file, name='upload'),
    path('upload/', views.upload_file, name='upload'),
]
