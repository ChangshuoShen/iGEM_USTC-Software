from django.shortcuts import render, HttpResponse, redirect
from django.urls import reverse
from django.utils import timezone
from Apps.accounts.models import User
from Apps.forum.models import post, Comment, Like
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib import messages
from Apps.raffle.models import Prize
# Create your views here.

THEME_ONE = 'Riddle', 'Riddle'
THEME_TWO = 'Share Something Interesting', 'Share Something Interesting'
THEME_THREE = 'Ask For Help', 'Ask For Help'
THEME_FOUR = 'Find Friends', 'Find Friends'
THEME_FIVE = 'Else', 'Else'

def dashboard(request):
    # 检查 session 中的 email 是否为 "super@mail"
    if request.session.get('email') == 'super@mail':
        # 获取统计数据
        total_accounts, accounts_today, accounts_yesterday = User.get_user_counts()
        total_posts, posts_today, posts_yesterday = post.get_post_counts()
        total_comments, comments_today, comments_yesterday = Comment.get_comment_counts()

        # 获取最新的帖子和用户
        latest_posts = post.get_latest_posts()
        latest_comments = Comment.get_latest_comments()

        context = {
            'total_accounts': total_accounts,
            'accounts_today': accounts_today,
            'accounts_yesterday': accounts_yesterday,
            
            'total_posts': total_posts,
            'posts_today': posts_today,
            'posts_yesterday': posts_yesterday,
            
            'total_comments': total_comments,
            'comments_today': comments_today,
            'comments_yesterday': comments_yesterday,
            
            'latest_posts': latest_posts,
            'latest_comments': latest_comments,
        }

        return render(request, 'dashboard.html', context)
    else:
        # 如果不是 "super@mail"，返回一个带有定时重定向的 HttpResponse
        response = HttpResponse("You are not authorized to access this page. You will be redirected to the home page in 5 seconds.")
        response['refresh'] = '5;url=' + reverse('home')  # 设置定时器
        return response

def riddles(request):
    riddles_list = post.get_posts_for_single_theme('Riddle')
    paginator = Paginator(riddles_list, 20)  # 每页显示20条帖子
    page_number = request.GET.get('page')
    try:
        riddles_page = paginator.page(page_number)
    except PageNotAnInteger:
        riddles_page = paginator.page(1)  # 如果page参数不是整数，则显示第一页
    except EmptyPage:
        riddles_page = paginator.page(paginator.num_pages)  # 如果page参数超出范围，则显示最后一页

    return render(request, 'post-list.html', {
        'this_url': 'admin_panel:riddles',
        'posts': riddles_page,
    })


def share_sth_interesting(request):
    interest_list = post.get_posts_for_single_theme('Share Something Interesting')
    paginator = Paginator(interest_list, 20)
    page_number = request.GET.get('page')
    try:
        interest_page = paginator.page(page_number)
    except PageNotAnInteger:
        interest_page = paginator.page(1)
    except EmptyPage:
        interest_page = paginator.page(paginator.num_pages)

    return render(request, 'post-list.html', {
        'posts': interest_page,
        'this_url': 'admin_panel:share_sth_interesting',
    })

def find_friends(request):
    friends_list = post.get_posts_for_single_theme('Find Friends')
    paginator = Paginator(friends_list, 20)
    page_number = request.GET.get('page')
    try:
        friends_page = paginator.page(page_number)
    except PageNotAnInteger:
        friends_page = paginator.page(1)
    except EmptyPage:
        friends_page = paginator.page(paginator.num_pages)

    return render(request, 'post-list.html', {
        'posts': friends_page,
        'this_url': 'admin_panel:find_friends',
    })

def ask_for_help(request):
    aid_list = post.get_posts_for_single_theme('Ask For Help')
    paginator = Paginator(aid_list, 20)
    page_number = request.GET.get('page')
    try:
        aid_page = paginator.page(page_number)
    except PageNotAnInteger:
        aid_page = paginator.page(1)
    except EmptyPage:
        aid_page = paginator.page(paginator.num_pages)

    return render(request, 'post-list.html', {
        'posts': aid_page,
        'this_url': 'admin_panel:ask_for_help',
    })

def else_list(request):
    else_list = post.get_posts_for_single_theme('Else')
    paginator = Paginator(else_list, 20)
    page_number = request.GET.get('page')
    try:
        else_page = paginator.page(page_number)
    except PageNotAnInteger:
        else_page = paginator.page(1)
    except EmptyPage:
        else_page = paginator.page(paginator.num_pages)

    return render(request, 'post-list.html', {
        'posts': else_page,
        'this_url': 'admin_panel:else',
    })


def delete_post(request):
    if request.method == "POST":
        post_id = int(request.POST.get('post_id'))
        return_url = request.POST.get('return_url')

        post.delete_post(post_id=post_id)
        return redirect(reverse(return_url))
        
    return HttpResponse('Request Failed')


def show_comments(request):
    comment_list = Comment.get_all_comments()
    # print(comment_list)
    paginator = Paginator(comment_list, 20)  # 每页显示20条评论
    page_number = request.GET.get('page')
    try:
        comments_page = paginator.page(page_number)
    except PageNotAnInteger:
        comments_page = paginator.page(1)
    except EmptyPage:
        comments_page = paginator.page(paginator.num_pages)

    return render(request, 'comment-list.html', {
        'comments': comments_page,
        'this_url': 'admin_panel:show_comments',
    })
    
    
def delete_comment(request):
    if request.method == 'POST':
        comment_id = request.POST.get('comment_id')
        return_url = request.POST.get('return_url', 'admin_panel:show_comments')
        Comment.delete_comment(comment_id)
        return redirect(reverse(return_url))
    
    return HttpResponse('Request Failed')

def user_list(request):
    user_list = User.get_all_users()
    paginator = Paginator(user_list, 20)
    page_number = request.GET.get('page')
    try:
        users = paginator.page(page_number)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'user-list.html', {
        'users': users,
        'this_url': 'admin_panel:user_list',
    })
    
    
def delete_user(request):
    if request.method == "POST":
        user_id = request.POST.get('user_id')
        return_url = request.POST.get('return_url', reverse('admin_panel:user_list'))
        user = User.get_user_by_id(user_id=user_id)
        # 停用用户帐户，而不是彻底删除
        user.deactivate_account()
        # messages.success(request, f'User {user.name} has been deactivated.')
        return redirect(return_url)
    
    return redirect(reverse('admin_panel:user_list'))


def prize_list(request):
    prizes = Prize.get_all_prizes()
    return render(request, 'prize-list.html', {'prizes': prizes})

def delete_prize(request):
    if request.method == 'POST':
        prize_id = request.POST.get('prize_id')
        try:
            prize = Prize.objects.get(pk=prize_id)
            prize.delete()
            messages.success(request, 'Prize deleted successfully.')
        except Prize.DoesNotExist:
            messages.error(request, 'Prize does not exist.')
    return redirect('admin_panel:prize_list')

def update_prize(request):
    if request.method == 'POST':
        prize_id = request.POST.get('prize_id')
        new_quantity = request.POST.get('new_quantity')
        if new_quantity.isdigit():
            Prize.update_prize(prize_id, int(new_quantity))
            messages.success(request, 'Prize quantity updated successfully.')
        else:
            messages.error(request, 'Invalid quantity.')
    return redirect('admin_panel:prize_list')

def clear_all_prizes(request):
    if request.method == 'POST':
        Prize.clear_all_prizes()
        messages.success(request, 'All prizes cleared successfully.')
    return redirect('admin_panel:prize_list')


def update_all_prizes(request):
    if request.method == 'POST':
        # 获取所有表单中提交的奖品信息
        for key, value in request.POST.items():
            if key.startswith('name_'):
                prize_id = key.split('_')[1]
                try:
                    prize = Prize.objects.get(pk=prize_id)
                    prize.quantity = int(value)
                    prize.save()
                except (Prize.DoesNotExist, ValueError):
                    pass
        # 更新完成后重定向到奖品列表页面
        return redirect('admin_panel:prize_list')
    else:
        # 如果不是POST请求，直接重定向到奖品列表页面
        return redirect('admin_panel:prize_list')
