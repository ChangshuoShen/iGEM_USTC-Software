from django.urls import path
from . import views

app_name = 'experiment'


urlpatterns = [
    path('download/', views.download, name = 'download'), # 处理好之后下载
    path('upload/', views.upload, name = 'upload'), # 上传实验数据
    path('exp_index', views.exp_index, name = 'exp_index'), # 物化实验主页
    path('explanation/', views.explanation, name = 'explanation'),
    # path('set_language/', views.set_language, name = 'set_language'),
    path('<str:exp_name>/', views.specific_exp, name = 'specific_exp'),    # 具体的一个实验
]
