from django.shortcuts import render, redirect
from .models import Feedback
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages


def feedback_page(request):
    # 先判断是不是已经注册过了，如果没有注册过，直接提示之后重定向到登陆界面
    user_id = request.session.get('user_id')
    email = request.session.get('email')
    
    print(request.session.__dict__)
    if not user_id or not email:
        messages.error(request, "Please Login before sending a feedback")
        return redirect('accounts:signup_login')
    else:
        return render(request, 'feedback.html')


def send_feedback(request):
    # 这里接受feedback
    if request.method == 'POST':
        email = request.session.get('email')
        satisfaction_level = request.POST.get('rating')
        message = request.POST.get('feedback')
        # print(f'''
        #     message: {message} 
        #     satisfaction_level: {satisfaction_level}
        #     email: {email}
        #       ''')
        if email and satisfaction_level and message:
            # 创建反馈
            Feedback.create_feedback(email=email, satisfaction_level=satisfaction_level, message=message)
            # messages.success(request, "Thank you for your feedback!")
            # 发送确认邮件
            send_mail(
                'Feedback Received',
                'Thank you for your feedback. We appreciate your input!',
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )
            return render(request, 'send_successfully.html')
        else:
            # messages.error(request, "Please fill in all fields.")
            return render(request, 'send_successfully.html')
    else:
        return render(request, 'send_successfully.html')