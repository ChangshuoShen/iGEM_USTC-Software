from django.db import models
from django.utils import timezone
from Apps.accounts.models import User
from django.shortcuts import get_object_or_404
from django.http import Http404
from django.core.paginator import Paginator
from django.db import transaction # 做一些事务锁
# Create your models here.

class post(models.Model):
    class ThemeChoices(models.TextChoices):
        '''
        这里是五类post，在USTC-Software中没用，之后直接删掉
        '''
        THEME_ONE = 'Riddle', 'Riddle'
        THEME_TWO = 'Share Something Interesting', 'Share Something Interesting'
        THEME_THREE = 'Ask For Help', 'Ask For Help'
        THEME_FOUR = 'Find Friends', 'Find Friends'
        THEME_FIVE = 'Else', 'Else'
    
    publisher_id = models.ForeignKey(User, on_delete=models.CASCADE)
    post_title = models.TextField(verbose_name='title', blank = False)
    post_content = models.TextField(verbose_name='post_content', blank = False)
    
    theme = models.CharField(
        verbose_name='theme',
        max_length=50,
        choices=ThemeChoices.choices,
        default=ThemeChoices.THEME_FIVE,
        blank=False
    )
    publish_date = models.DateTimeField(verbose_name='published_date', default=timezone.now)
    post_likes = models.IntegerField(default=0, verbose_name='likes')
    
    def __str__(self):
        return self.post_title
    
    @classmethod
    def create_post(cls, publisher_id, post_title, post_content, theme='Else'):
        post_object = cls.objects.create(
            publisher_id=publisher_id,
            post_title=post_title,
            post_content=post_content,
            theme=theme
        )
        return post_object

    @classmethod
    def get_post_by_id(cls, post_id):
        try:
            post_object = get_object_or_404(cls, id=post_id)
            
            return {
                'post_id': post_id,
                'publisher_id': post_object.publisher_id.id,  # Assuming you want the ID of the publisher
                'post_title': post_object.post_title,
                'post_detail': post_object.post_content,  # post_content key renamed to post_detail
                'theme': post_object.theme,
                'publish_date': post_object.publish_date,
                'post_likes': post_object.post_likes
            }
        except Http404:
            return None
        
    @classmethod
    def get_post_instance_by_id(cls, post_id):
        return get_object_or_404(cls, id=post_id)
    
    @classmethod
    def get_post_counts(cls):
        total_posts = cls.objects.count()
        posts_today = cls.objects.filter(publish_date__date=timezone.now().date()).count()
        posts_yesterday = cls.objects.filter(publish_date__date=(timezone.now() - timezone.timedelta(days=1)).date()).count()
        return total_posts, posts_today, posts_yesterday

    @classmethod
    def get_latest_posts(cls, post_num=5):
        if post_num < 1:
            post_num = 5
        return list(cls.objects.all().order_by('-publish_date').values('post_title', 'post_content', 'theme', 'publish_date', 'post_likes')[:post_num])
    
    @classmethod
    def update_post(cls, post_id, post_title=None, post_content=None, theme=None):
        post_object = get_object_or_404(cls, id=post_id)
        if post_title:
            post_object.post_title = post_title
        if post_content:
            post_object.post_content = post_content
        if theme:
            post_object.theme = theme
        post_object.save()
        return post_object

    @classmethod
    def delete_post(cls, post_id):
        post_object = get_object_or_404(cls, id=post_id)
        post_object.delete()
        
    @classmethod
    def get_all_posts(cls, page=1, items_per_page=10):
        '''
        获取所有的帖子，并支持分页
        '''
        all_posts = cls.objects.all().order_by('-publish_date')
        paginator = Paginator(all_posts, items_per_page)
        page_obj = paginator.get_page(page)
        # 下面的信息不仅要有帖子信息，还要有用户信息
        paginated_posts = {
        'posts': [
            {
                'post_id': post.id,
                'publisher_id': post.publisher_id.id,
                'publisher_name': post.publisher_id.name,
                'publisher_bio': post.publisher_id.bio,
                'post_title': post.post_title,
                'post_detail': post.post_content,
                'publish_date': post.publish_date,
            }
            for post in page_obj
        ],
        'paginator': paginator,
        'page_obj': page_obj,
        }
        return paginated_posts
        
    @classmethod
    def get_posts_by_theme(cls, page, items_per_page = 10):
        """
        获取所有帖子，并按照主题分类，支持分页
        """
        # 创建一个空字典用于存储帖子按主题分类
        posts_by_theme = {}
        paginated_posts_by_theme = {}

        # 遍历所有主题选项
        for i, theme_choice in enumerate(cls.ThemeChoices.choices, start=1):
            theme_name = theme_choice[0]  # 主题名称
            theme_posts = cls.objects.filter(theme=theme_name).order_by('-publish_date')
            paginator = Paginator(theme_posts, items_per_page)
            # 获取对应页的数据
            page_obj = paginator.get_page(page)

            posts_by_theme[i] = [
                {
                    'id': post.id,
                    'publisher_id': post.publisher_id.id,
                    'post_title': post.post_title,
                    'post_detail': post.post_content,
                    'theme': post.theme,
                    'publish_date': post.publish_date,
                    'post_likes': post.post_likes
                }
                for post in page_obj
            ]

            paginated_posts_by_theme[i] = {
                'posts': posts_by_theme[i],
                'paginator': paginator,
                'page_obj': page_obj
            }

        return paginated_posts_by_theme
    
    @classmethod
    def get_posts_for_single_theme(cls, theme):
        posts = post.objects.filter(theme=theme).order_by('-publish_date')
        return list(posts.values('id', 'theme', 'post_title', 'post_content', 'publish_date'))

    

class Comment(models.Model):
    post = models.ForeignKey(post, related_name='comments', on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    content = models.TextField(verbose_name='comment_content', blank=False)
    comment_date = models.DateTimeField(verbose_name='created_at', default=timezone.now)
    comment_likes = models.IntegerField(default=0, verbose_name='likes')

    def __str__(self):
        return f'Comment by {self.user.name} on {self.post.post_title}'

    @classmethod
    def find_comments_on_specific_post(cls, post):
        return cls.objects.filter(post=post)

    @classmethod
    def find_comments_on_specific_post_through_post_id(cls, post_id):
        return cls.objects.filter(post=post_id).order_by('-comment_date')
    
    @classmethod
    def get_comment_counts(cls):
        total_accounts = cls.objects.count()
        accounts_today = cls.objects.filter(comment_date__date=timezone.now().date()).count()
        accounts_yesterday = cls.objects.filter(comment_date__date=(timezone.now() - timezone.timedelta(days=1)).date()).count()
        return total_accounts, accounts_today, accounts_yesterday
    
    @classmethod
    def get_latest_comments(cls, comment_num=5):
        if comment_num < 1:
            comment_num = 5
        return list(cls.objects.all().order_by('-comment_date').values('post', 'user', 'content', 'comment_date')[:comment_num])
    
        # return cls.objects.order_by('-comment_date')[:comment_num]
    
    @classmethod
    def create_comment(cls, post, user, content):
        comment = cls.objects.create(post=post, user=user, content=content)
        return comment

    @classmethod
    def delete_comment(cls, comment_id):
        comment = cls.objects.get(id=comment_id)
        comment.delete()

    @classmethod
    def update_comment(cls, comment_id, content):
        comment = cls.objects.get(id=comment_id)
        comment.content = content
        comment.save()

    @classmethod
    def get_comment_by_id(cls, comment_id):
        return cls.objects.get(id=comment_id)
    
    @classmethod
    def get_all_comments(cls):
        return cls.objects.all().order_by('-comment_date')





class Reply(models.Model):
    '''
    专指针对某个评论的reply
    '''
    comment = models.ForeignKey(Comment, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    reply_content = models.TextField(verbose_name='reply_content', blank=False)
    reply_date = models.DateTimeField(verbose_name='created_at', default=timezone.now)

    def __str__(self):
        return f'Reply by {self.user.name}'

    @classmethod
    def find_replies_on_specific_comment_through_comment_id(cls, comment_id):
        return cls.objects.filter(comment=comment_id).order_by('-reply_date')
    
    @classmethod
    def create_reply(cls, comment, user, content):
        reply = cls.objects.create(comment=comment, user=user, reply_content=content)
        return reply

    @classmethod
    def delete_reply(cls, reply_id):
        reply = cls.objects.get(id=reply_id)
        reply.delete()

    @classmethod
    def update_reply(cls, reply_id, content):
        reply = cls.objects.get(id=reply_id)
        reply.reply_content = content
        reply.save()

    @classmethod
    def get_reply_by_id(cls, reply_id):
        return cls.objects.get(id=reply_id)

class Like(models.Model):
    '''
    这里记录点赞关系
    '''
    post = models.ForeignKey(post, related_name='likes', on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(verbose_name='created_at', default=timezone.now)

    def __str__(self):
        return f'Like by {self.user.name} on {self.post.post_title}'
    
    @classmethod
    def like_post(cls, post_instance, user_instance):
        '''
        注意传入的两个参数都是实例
        '''
        if not cls.objects.filter(post=post_instance, user=user_instance).exists():
            new_like = cls.objects.create(post=post_instance, user=user_instance)
            # 这一行相当于是trigger了
            # post_instance.likes = cls.count_likes_for_post(post_id=post_instance)
            post_instance.post_likes += 1
            new_like.save()
            post_instance.save()

    @classmethod
    def unlike_post(cls, post_instance, user_instance):
        like_queryset = cls.objects.filter(post=post_instance, user=user_instance)
        if like_queryset.exists():
            like_queryset.delete()
            # post_instance.likes = cls.count_likes_for_post(post_id=post_instance)
            post_instance.post_likes -= 1
            post_instance.save()

    @classmethod
    def count_likes_for_post(cls, post_id):
        # 暂时用处不大，但是后面可能会用到 
        return cls.objects.filter(post_id=post_id).count()

    @classmethod
    def count_likes_by_user(cls, user_id):
        return cls.objects.filter(user_id=user_id).count()


# 下面这部分是用来存储Teaching的路径用于展示的
class TeachingMaterial(models.Model):
    title = models.CharField(verbose_name='title', max_length=255)
    pdf_file = models.FileField(verbose_name='pdf_file', upload_to='pdfs/teaching/')
    publish_date = models.DateField(verbose_name='publish_date', default=timezone.now)
    publisher = models.ForeignKey(User, on_delete=models.CASCADE)
    description = models.TextField(verbose_name='description', blank=True, null=True)

    def __str__(self):
        return self.title


# 这里存课程资源
class CourseResource(models.Model):
    title = models.CharField(verbose_name='title', max_length=255)
    pdf_file = models.FileField(verbose_name='pdf_file', upload_to='pdfs/course_resources/')
    publish_date = models.DateField(verbose_name='publish_date', default=timezone.now)
    publisher = models.ForeignKey(User, on_delete=models.CASCADE)
    description = models.TextField(verbose_name='description', blank=True, null=True)

    def __str__(self):
        return self.title
    
    
class DevelopmentLog(models.Model):
    title = models.CharField(verbose_name='Title', max_length=255)
    description = models.TextField(verbose_name='Description')
    log_date = models.DateField(verbose_name='Log Date', default=timezone.now)

    def __str__(self):
        return self.title