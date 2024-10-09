from django.shortcuts import render, HttpResponse, get_object_or_404, redirect
from django.contrib import messages
from .models import post, Comment, Reply, Like, TeachingMaterial, CourseResource, DevelopmentLog
from Apps.accounts.models import User
from django.core.paginator import Paginator
from django.urls import reverse


def forum_index(request):
    # 我们现在重定义一个forum_index，用来直接展示所有的forum的内容，不再关心类别
    page_number = request.GET.get('page', 1)
    items_per_page = 10

    # 使用 Post 类的 get_all_posts 方法获取所有帖子，并支持分页
    posts_data = post.get_all_posts(page=int(page_number), items_per_page=items_per_page)

    return render(request, 'forum.html', {
        'posts_data': posts_data,
    })


def riddle_difficulty_index(request):
    '''
    返回一个长度为3的列表，包括不同难度等级的Riddle帖子：
    0 - 'Easy'
    1 - 'Medium'
    2 - 'Hard'
    '''
    riddles_by_difficulty = post.get_riddles_by_difficulty()

    difficulties_ordered = ['easy', 'medium', 'hard']
    riddle_contents = [riddles_by_difficulty.get(difficulty, []) for difficulty in difficulties_ordered]

    return render(request, 'riddle_index.html', {
        'riddle_difficulty_contents': riddle_contents,
    })

def riddle_category_index(request):
    riddles_by_category = post.get_riddles_by_main_category()
    return render(request, 'riddle_category_index.html', {
        'riddles_by_category': riddles_by_category.items(),
    })

def share(request):
    # 检查用户是否已登录
    if 'user_id' not in request.session or 'email' not in request.session:
        # 用户未登录，重定向到登录页面
        messages.error(request, 'Please log in to leave a comment or reply.')
        return redirect('accounts:signup_login')
    
    print(request.session.__dict__)
    return render(request, 'share.html')

def teaching_material_index(request):
    materials_list = TeachingMaterial.objects.all()
    paginator = Paginator(materials_list, 10)  # 每页显示10条记录
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    return render(request, 'teaching.html', {'page_obj': page_obj})

def to_upload_teaching_material(request):
    return render(request, 'to_upload_teaching_material.html')

def upload_teaching_material(request, material_id=None):
    if request.method == 'POST':
        title = request.POST.get('title')
        pdf_file = request.FILES.get('pdf_file')
        description = request.POST.get('description')
        user_id = request.session.get('user_id')
        user = User.get_user_by_id(user_id)
        if material_id:
            material = TeachingMaterial.objects.get(pk=material_id)
            material.title = title
            material.publisher = user
            material.description = description
            if pdf_file:
                material.pdf_file = pdf_file
            material.save()
        else:
            material = TeachingMaterial.objects.create(
                title=title,
                pdf_file=pdf_file,
                publisher=user,
                description=description
            )
        # return HttpResponse('success')
        return redirect('forum:teaching_detail', material_id=material.id)

    else:
        material = TeachingMaterial.objects.get(pk=material_id) if material_id else None
    return HttpResponse("What???")


def teaching_detail(request, material_id):
    material = TeachingMaterial.objects.get(pk=material_id)
    response = render(request, 'teaching_detail.html', {'material': material})
    response['Content-Security-Policy'] = "frame-ancestors 'self' http://127.0.0.1:8000"
    return response

# 这部分是课程资料

def course_resources_index(request):
    resources_list = CourseResource.objects.all()
    paginator = Paginator(resources_list, 8)  # 每页显示10条记录
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    return render(request, 'course_resources.html', {'page_obj': page_obj})

def to_upload_course_resource(request):
    return render(request, 'to_upload_course_resource.html')

def upload_course_resource(request, resource_id=None):
    if request.method == 'POST':
        title = request.POST.get('title')
        pdf_file = request.FILES.get('pdf_file')
        description = request.POST.get('description')
        user_id = request.session.get('user_id')
        user = User.get_user_by_id(user_id)
        if resource_id:
            resource = CourseResource.objects.get(pk=resource_id)
            resource.title = title
            resource.publisher = user
            resource.description = description
            if pdf_file:
                resource.pdf_file = pdf_file
            resource.save()
        else:
            resource = CourseResource.objects.create(
                title=title,
                pdf_file=pdf_file,
                publisher=user,
                description=description
            )
        return redirect('forum:course_resource_detail', resource_id=resource.id)

    else:
        resource = CourseResource.objects.get(pk=resource_id) if resource_id else None
    return HttpResponse("What???")

def course_resource_detail(request, resource_id):
    resource = CourseResource.objects.get(pk=resource_id)
    response = render(request, 'course_resource_detail.html', {'resource': resource})
    response['Content-Security-Policy'] = "frame-ancestors 'self' http://127.0.0.1:8000"
    return response



# 这里开始是开发日志的相关内容
def development_log_index(request):
    logs_list = DevelopmentLog.objects.all().order_by('-log_date')
    return render(request, 'development_log.html', {'logs_list': logs_list})

def to_upload_development_log(request):
    return render(request, 'to_upload_development_log.html')

def upload_development_log(request, log_id=None):
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description')
        log_date = request.POST.get('log_date')
        user_id = request.session.get('user_id')
        user = User.objects.get(pk=user_id)
        if log_id:
            log = DevelopmentLog.objects.get(pk=log_id)
            log.title = title
            log.description = description
            log.log_date = log_date
            log.created_by = user
            log.save()
        else:
            DevelopmentLog.objects.create(
                title=title,
                description=description,
                log_date=log_date,
                created_by=user
            )
        return redirect('forum:development_log_index')

    else:
        log = DevelopmentLog.objects.get(pk=log_id) if log_id else None
    return render(request, 'to_upload_development_log.html', {'log': log})

# 将所有的上传综合到一个页面
def to_upload(request):
    if request.session.get('email') == 'super@mail':
        return render(request, 'to_upload.html')
    else:
        # 如果不是 "super@mail"，返回一个带有定时重定向的 HttpResponse
        response = HttpResponse("You are not authorized to access this page. You will be redirected to the home page in 5 seconds.")
        response['refresh'] = '3;url=' + reverse('forum:forum_index')  # 设置定时器
        return response

def submit_sharing(request):
    if request.method == 'POST':
        user_id = request.session.get('user_id')
        user = User.get_user_by_id(user_id=user_id)
        title = request.POST.get('title')
        content = request.POST.get('content_copy')  # 确保与表单中的隐藏字段名称一致
        # print(f"Title: {title}, Content: {content}")  # 调试打印语句
        new_post = post.create_post(publisher_id=user, post_title=title, post_content=content)
        messages.success(request, 'Post successfully')
        return show_post_detail(request=request, post_id=new_post.id)
    else:
        return HttpResponse('Error, please try again')



# from .utils.add_some_replies import create_random_replies
def show_post_detail(request, post_id):
    # create_random_replies()
    # return HttpResponse('Show More Information Here')
    post_content = post.get_post_by_id(post_id=post_id)
    post_content['post_id'] = post_id
    
    if post_content:
        # 获取发布者的详细信息
        request.session['present_post_id'] = post_id
        publisher_id = post_content.get('publisher_id')
        user = User.get_user_by_id(publisher_id)
        publisher = {
            'id': user.id,
            'username': user.name,
            'email': user.email,
            'bio': user.bio,
            'gender': user.gender,
        }
    else:
        publisher = None
    
    # 这里开始找所有的comment
    relevant_comments = Comment.find_comments_on_specific_post_through_post_id(post_id=post_id)
    main_comments = []
    for single_comment in relevant_comments:
        # 这里查询该comment对应的所有的reply
        relevant_replies = Reply.find_replies_on_specific_comment_through_comment_id(single_comment.id)
        replies = []
        for single_reply in relevant_replies:
            # 需要传过来的有谁回复的，回复内容和回复时间
            replies.append(
                {
                    'replier': single_reply.user.name,
                    'reply_detail': single_reply.reply_content,
                    'date': single_reply.reply_date,
                }
            )
            # print(replies)
        main_comments.append(
            {
                'id': single_comment.id,
                'commenter': single_comment.user.name,
                'comment_detail': single_comment.content,
                'date': single_comment.comment_date,
                'likes': single_comment.comment_likes,
                'replies': replies,
            }
        )
    
    return render(request, 'post_detail.html', {
        'publisher': publisher,
        'post_content': post_content,
        'main_comments': main_comments,
    })


def like_post(request):
    if request.method == 'POST':
        # 检查用户是否已登录
        # print(request.session.__dict__)
        if 'user_id' not in request.session or 'email' not in request.session:
            # 用户未登录，重定向到登录页面
            messages.error(request, 'Please log in to like it.')
            return redirect('accounts:signup_login')        

        user_id = int(request.session.get('user_id'))
        user_instance = User.get_user_by_id(user_id=user_id)
        
        post_id = int(request.POST.get('post_id'))
        
        post_instance = post.get_post_instance_by_id(post_id=post_id)
        Like.like_post(post_instance=post_instance, user_instance=user_instance)
        return show_post_detail(request, post_id=post_id)
    
    else:
        return HttpResponse('Error???')


def comment_or_reply(request):
    if request.method == 'POST':
        # 检查用户是否已登录
        # print(request.session.__dict__)
        if 'user_id' not in request.session or 'email' not in request.session:
            # 用户未登录，重定向到登录页面
            messages.error(request, 'Please log in to leave a comment or reply.')
            return redirect('accounts:signup_login')
        
        user_id = request.session.get('user_id')
        post_id = request.session.get('present_post_id')
        # 用户已登录
        # print(request.POST)
        comment_or_reply = request.POST.get('comment_or_reply')
        reply = request.POST.get('reply')
        comment_id = request.POST.get('comment_id')
        # 根据reply的值判断是评论还是回复
        
        user_instance = User.get_user_by_id(user_id=user_id)
        if reply in ('true', 'True'):
            # 这是一个回复
            comment_instance = Comment.get_comment_by_id(comment_id=comment_id)
            Reply.create_reply(comment=comment_instance, user=user_instance, content=comment_or_reply)
        else:
            # 这是一个评论
            post_instance = post.get_post_instance_by_id(post_id)
            Comment.create_comment(post=post_instance, user=user_instance, content=comment_or_reply)
        return show_post_detail(request, post_id=post_id)

    return HttpResponse("Nothing to see here")


# from .utils.add_some_comments import create_comments_for_all_users_and_posts
def user_question(request):
    # create_comments_for_all_users_and_posts()
    return render(request, 'user_question.html')


def users(request):
    num = 6
    users_info = [{'name': 'Runge',
              'rank': 'Top',
              'main_contents': '6666666666666666666666',
              }
            ] * num
    
    return render(request, 'user.html', {
        'users_info': users_info
    })

