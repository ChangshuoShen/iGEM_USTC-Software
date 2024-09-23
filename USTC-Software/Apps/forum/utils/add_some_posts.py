from ..models import post
from Apps.accounts.models import User
from django.utils import timezone
'''
将这段代码随便放进一个视图函数中去执行就可以，就不用管“默认执行环境”了
记得别重复使用
'''

def add_posts():
    # 创建或获取一个示例用户
    user, created = User.objects.get_or_create(
        email='shenchangshuo@icloud.com',
        defaults={
            'password': 'examplepassword',
            'name': 'Example User',
            'register_date': timezone.now()
        }
    )
    if created:
        user.set_password('examplepassword')
        user.save()

    themes = [
        ('Riddle', "What's My Name?", "Guess my name and win a prize!"),
        ('Share Something Interesting', "Interesting Fact", "Did you know that..."),
        ('Ask For Help', "Need Help with Django", "Can someone help me with this Django error?"),
        ('Find Friends', "Looking for Friends", "Hey, anyone interested in hiking this weekend?"),
        ('Else', "Random Thought", "Just had a random thought..."),
    ]

    for theme, title, content in themes:
        for i in range(4, 30):
            post.create_post(
                publisher_id=user,
                post_title=f"{title} {i+1}",
                post_content=f"{content} This is post number {i+1} in the {theme} category.",
                theme=theme
            )
