from django.urls import path
from . import views

app_name = "admin_panel"

urlpatterns = [
    path('dashboard/', views.dashboard, name="dashboard"),
    # path('post_list', views.post_list, name="post_list"),
    path('riddles/', views.riddles, name="riddles"),
    path('share_sth_interesting/', views.share_sth_interesting, name="share_sth_interesting"),
    path('ask_for_help/', views.ask_for_help, name="ask_for_help"),
    path('find_friends', views.find_friends, name="find_friends"),
    path("else/", views.else_list, name="else"),
    path('delete_post/', views.delete_post, name="delete_post"),
    
    path('show_comments/', views.show_comments, name='show_comments'),
    path('delete_comment/', views.delete_comment, name='delete_comment'),
    
    path('user_list/', views.user_list, name='user_list'),
    path('delete_user/', views.delete_user, name='delete_user'),
    
    path('prizes/', views.prize_list, name='prize_list'),
    path('delete_prize/', views.delete_prize, name='delete_prize'),
    path('update_prize/', views.update_prize, name='update_prize'),
    path('clear_all_prizes/', views.clear_all_prizes, name='clear_all_prizes'),
    path('update_all_prizes/', views.update_all_prizes, name='update_all_prizes'),
]

