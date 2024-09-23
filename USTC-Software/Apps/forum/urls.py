from django.urls import path
from . import views
app_name = 'forum'


urlpatterns = [
    path('forum_index/', views.forum_index, name="forum_index"), # 这是论坛的主页
    path('share/', views.share, name="share"), # 编写帖子
    path('submit/', views.submit_sharing, name="submit_sharing"), # 实现post的提交逻辑
    path('post_detail/<int:post_id>', views.show_post_detail, name="post_detail"), # 展示一个发帖的详细信息，内容，作者的基本信息和链接，评论的图
    path('comment_or_reply/', views.comment_or_reply, name="comment_or_reply"),
    # path('user_question/', views.user_question, name="user_question"), # 一个user的相关信息
    # path('users/', views.users, name='users'), # 展示所有的user
    path('like_post/', view=views.like_post, name='like_post'), # 这个实现对一个post点赞的
    
    
    # 这部分是团队一些teaching内容
    path('teaching/', views.teaching_material_index, name='teaching'),
    path('to_upload_teaching_material/', view=views.to_upload_teaching_material, name='to_upload_teaching_material'), # 这个是上传
    path('upload_teaching_material/', views.upload_teaching_material, name='upload_teaching_material'), # 这里上传教学部分的pdf
    path('teaching_detail/<int:material_id>', views.teaching_detail, name='teaching_detail'),
    
    
    # 这部分是分享的一些课程资源
    path('course_resources/', views.course_resources_index, name='course_resources'),
    path('to_upload_course_resource/', views.to_upload_course_resource, name='to_upload_course_resource'),
    path('upload_course_resource/', views.upload_course_resource, name='upload_course_resource'),
    path('course_resources/<int:resource_id>/', views.course_resource_detail, name='course_resource_detail'),
    
    # 这里是开发日志
    path('development_log/', views.development_log_index, name='development_log'),
    path('to_upload_development_log/', views.to_upload_development_log, name='to_upload_development_log'),
    path('upload_development_log/', views.upload_development_log, name='upload_development_log'),
    
    # 整合上传
    path('to_upload/', views.to_upload, name='to_upload')
]

