from django.urls import path
from . import views
app_name = 'accounts'


urlpatterns = [
    path('signup_login/', views.signup_login, name='signup_login'),

    path('send_verification_code/', views.send_verification_code, name='send_verification_code'),
    path('signup_check', views.confirm_signup, name='signup_check'),
    path('login_check/', views.login_check, name='login_check'),
    path('change_pwd/', views.change_pwd, name='change_pwd'),

    path('signup/', views.signup, name='signup'), # 这俩是测试使用的，没有实际意义，暂时不删掉
    path('login/', views.login, name='login'),

    path('logout/', views.logout, name='logout'),
    
    path('profile/', views.profile, name='profile'), # 这里实现用户信息的展示和修改
    path('change_profile/', views.change_profile, name='change_profile'),
    
]
