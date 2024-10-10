from django.shortcuts import render, HttpResponse, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.hashers import make_password, check_password
# from django.utils import timezone

# 这一部分是实现邮件发送的包依赖
from django.core.mail import send_mail
from django.conf import settings
from Apps.accounts.models import User
# from Apps.forum.models import post

from django.contrib import messages
import random
import string


# login主要是返回一个login的页面
def login(request):
    return render(request, 'login.html')


def signup_login(request):
    # print(request.session.__dict__)
    # print(request.COOKIES)
    if 'user_id' in request.session and 'email' in request.session:
        return redirect('accounts:profile')
    return render(request, 'signup_login.html')


def authenticate_user(email, password):
    # print('执行这个检查用户名和密码的函数了～～～')
    try:
        # 根据邮箱查找用户记录
        # user = User.custom_query('id', 'password', email=email)
        user = User.get_user_by_email(email=email)
        # print(user)
        # 检查密码是否匹配
        if user and check_password(password, user.password):
            # 密码匹配，返回用户对象表示验证成功
            return user
        else:
            # 密码不匹配，返回None表示验证失败
            return None

    except User.DoesNotExist:
        # print('没有这个人，DoesNotExist是django中所有class都有的一个方法，不需要特别定义')
        # 如果找不到对应的用户记录，返回None表示验证失败
        return None


# check是在login中submit之后执行的逻辑，用于检查本次登录有效性，返回对应信息
def login_check(request):
    # request_dict = {
    #     'method': request.method,
    #     'path': request.path,
    #     'GET': dict(request.GET),
    #     'POST': dict(request.POST),
    #     # 其他需要的信息
    # }
    # print(request_dict)
    # 从这里开始执行整个检查逻辑
    if request.method == 'POST':
        email = request.POST.get('email')
        # print('email: ', email)

        forget_btn = request.POST.get('forget_btn')
        
        if forget_btn:
            messages.success(request, 'Verification Sent')
            return send_verification_code(request, action='change_pwd')
            

        # 如果不是那就是正常的执行登陆的逻辑咯
        password = request.POST.get('password')
        # print(password)     # 这里返回的键值对中值都是列表，但是使用GET/POST的get方法之后就是一个单值了

        if email and password:
            # 调用自定义的 authenticate_user 函数验证用户凭据
            user = authenticate_user(email, password)
            if user:
                # 验证成功，将用户对象存储在 session 中表示用户已登录
                request.session['user_id'] = user.id
                request.session['email'] = user.email
                messages.success(request, 'Login successful!')
                
                # del request.session['verification_code']
                
                # 设置用户的cookie，以便下次访问时自动登录
                response = redirect('accounts:profile')
                response.set_cookie('user_id', user.id)
                response.set_cookie('email', user.email)
                return response

            else:
                # 验证失败，显示错误消息并重定向回登录页面
                messages.error(request, 'Invalid email or password!')
                return redirect('accounts:signup_login')
        else:
            # 缺少必要的字段，显示错误消息并重定向回登录页面
            messages.error(request, 'Email and password are required!')
            return redirect('accounts:signup_login')
    else:
        # 不是POST请求，直接重定向回登录页面
        return redirect('accounts:signup_login')


def signup(request):
    return render(request, 'signup.html')


def send_verification_code(request, action='register'):
    if request.method == 'POST':
        email = request.POST.get('email')
        request.session['email'] = email
        if email:
            # 生成随机验证码
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            # print('code: ', code)
            # 发送邮件
            send_mail(
                'Verification Code',
                f'Your verification code is: {code}',
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )

            # 将验证码存储在session中，验证之后注意清除
            request.session['verification_code'] = code
            messages.success(request, f'Verification code sent successfully to {email}!')

            if action == 'register':
                name = request.POST.get('name')
                password = request.POST.get('password')
                request.session['user_data'] = {'name': name, 'email': email, 'password': password}
                return render(request, 'signup_login.html', {'register_verification_sent': True})

            else:   # 否则就是用来修改密码
                return render(request, 'signup_login.html', {'change_pwd_verification_sent': True})
        else:
            messages.error(request, 'Email is required!')
    else:
        return None
    

def verify_verification_code(request, verification_code):
    saved_code = request.session.get('verification_code')
    # print('saved code:', saved_code)
    if saved_code == verification_code:
        # del request.session['verification_code']
        user_data = request.session.get('user_data')
        if user_data:
            return user_data
        else:
            messages.error(request, 'User data not found in session.')
            return True
    else:
        messages.error(request, 'Invalid verification code! Please Try Again')
        return False


def change_pwd(request):
    if request.method == 'POST':
        verification_code = request.POST.get('verification_code')
        new_password = request.POST.get('new_password')
        # print("Begin changing PWD")
        # 验证验证码是否正确
        user_data = verify_verification_code(request, verification_code)
        if user_data:
            # 从session中获取用户的email
            email = request.session.get('email')
            # 根据email查找用户
            user = User.get_user_by_email(email)

            if user:
                # 重置用户密码
                # print('User Found')
                user.update_password(new_password)
                # 清除验证码和email的session信息
                del request.session['verification_code']
                del request.session['email']
                messages.success(request, 'Password changed successfully!')
                return render(request, 'signup_login.html')
            else:
                messages.error(request, 'Email not registered yet!')
        else:
            messages.error(request, 'Invalid verification code!')

    return render(request, 'signup_login.html')


def confirm_signup(request):
    if request.method == 'POST':
        # username = request.POST.get('username')
        # password = request.POST.get('password')
        # email = request.POST.get('email')
        verification_code = request.POST.get('verification_code')
        # print('verification code input: ', verification_code)
        user_data = verify_verification_code(request, verification_code)

        if user_data:
            # print(user_data)
            # 检查该邮箱是否已经注册：
            email_registered = User.get_user_by_email(user_data['email'])
            if email_registered:
                messages.error(request, 'Email has been registered, please login directly')
                return render(request, 'signup_login.html')
            # 使用检索到的数据创建用户
            User.create_user(
                name=user_data['name'],
                raw_password=user_data['password'],
                email=user_data['email']
            )
            # 注册成功后清除 session 中的相关数据
            del request.session['verification_code']
            del request.session['user_data']
            messages.success(request, 'Signup successful! Now Please Login~')
            return redirect('accounts:signup_login')
        else:
            messages.error(request, 'User data not found in session.')

    else:
        messages.error(request, 'Invalid request method.')
    return redirect('accounts:signup_login')


def profile(request):
    request.session.save()  # 强制保存session
    # print(request.session.__dict__) # 只是打印在服务器端看看，不使用log日志
    # print("COOKIES", request.COOKIES)
    email = request.session.get('email')
    user = User.get_user_by_email(email)
    # print("user's birthday: ", user.birthday)
    user_info = {
        'username': user.name,
        'email': user.email,
        'gender': user.gender,
        'birthday': user.birthday,
        'bio': user.bio,
        'likes': user.likes_num,
        'liked_by': user.liked_num,
    }
    
    
    # print(request.COOKIES)
    return render(request, 'profile.html', user_info)

def change_profile(request):
    '''
    这个函数用于在profile页面提交之后用于修改用户对应的一些相关资料
    '''
    if request.method == 'POST':
        # 获取表单数据
        # print(request.POST)  # 打印表单的普通字段数据
        email = request.session.get('email')  # 获取当前登录用户的email
        user = User.get_user_by_email(email)
        
        if not user:
            return HttpResponse('User not found')
        
        username = request.POST.get('username')
        gender = request.POST.get('gender')
        birthday = request.POST.get('birthday')
        bio = request.POST.get('bio')
        
        # 更新用户信息
        if username:
            user.name = username
        if gender:
            user.gender = gender
        if birthday:
            user.birthday = birthday
        if bio:
            user.bio = bio
        user.save()
        messages.success(request, 'Profile changed successfully!')
        # print('信息修改成功')
        
    user_info = {
        'username': user.name,
        'email': user.email,
        'gender': user.gender,
        'birthday': user.birthday,
        'bio': user.bio,
        'likes': user.likes_num,
        'liked_by': user.liked_num,
    }
    return render(request, 'profile.html', user_info)



def logout(request):
    # request.session.flush()
    # 清除 session 中的用户信息
    if 'user_id' in request.session:
        del request.session['user_id']
    if 'email' in request.session:
        del request.session['email']
    
    # 删除相关的 cookie
    response = HttpResponseRedirect(reverse('accounts:signup_login'))  # 替换 'login' 为你的登录页面的 URL 名称
    if 'user_id' in request.COOKIES:
        response.delete_cookie('user_id')
    if 'email' in request.COOKIES:
        response.delete_cookie('email')
        
    return response
    
    response = HttpResponseRedirect(reverse('accounts:signup_login'))
    response.delete_cookie('your_cookie_name')  # 替换 'your_cookie_name' 为你的 cookie 名称
    return response

