from django.urls import path
from . import views
app_name = 'rna_seq'


urlpatterns = [
    path('', views.rna_index, name='rna_seq_index'),
    # path('upload/', views.upload_file, name='upload'),
    path('upload/', views.upload_file, name='upload'),
    path('download/<int:user_id>', views.download_user_folder, name='download'),
    
    path('process_data/', views.process_data, name='process_data'),
    path('show_progress/', views.show_progress, name='show_progress'),
    path('progress/', views.show_progress1, name='progress'),
]
