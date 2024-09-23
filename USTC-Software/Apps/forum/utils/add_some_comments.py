from django.http import HttpResponse
from Apps.accounts.models import User
from ..models import post, Comment
from django.utils import timezone

'''
随便import到一个视图函数中，然后直接执行就有了
'''

def create_comments_for_all_users_and_posts():
    users = User.get_all_users()
    posts = post.get_all_posts()
    
    example_comments = [
        "This is a great post!",
        "I totally agree with your point.",
        "Interesting perspective, thanks for sharing!",
        "Could you provide more details?",
        "I learned something new today, thanks!"
    ]
    
    for user in users:
        for post_instance in posts:
            for content in example_comments:
                Comment.create_comment(
                    post=post_instance,
                    user=user,
                    content=content
                )
    print("Comments have been successfully created for all user-post combinations.")
