from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.contrib import messages
from Apps.accounts.models import User

import os
import shutil
import sys
from django.views.decorators.csrf import csrf_protect
import glob
from django.utils import translation

def exp_index(request):
    # 先检查是否登陆成功，后面需要根据用户专门建文件夹存放实验数据和处理结果
    # 检查用户的 cookie 或 session，判断是否已登录
    user_id = request.COOKIES.get('user_id')
    email = request.COOKIES.get('email')
    
    # 如果没有找到用户信息，重定向到登录页面
    if not user_id or not email:
        messages.warning(request, 'Please login before using Physical Chemistry Experiment Tools')
        return redirect('accounts:signup_login')
    
     # 检查用户是否存在于数据库中
    try:
        user = User.get_user_by_email(email)
        if user and user.id == int(user_id):
            # 用户已登录，继续渲染页面
            return render(request, 'exp_index.html')
        else:
            # 用户信息无效，重定向到登录页面
            messages.warning(request, 'Please login before using Physical Chemistry Experiment Tools')
            response = redirect('accounts:signup_login')
            response.delete_cookie('user_id')
            response.delete_cookie('email')
            return response
    except Exception as e:
        # 异常处理，重定向到登录页面
        messages.error(request, 'Error occured!!! Please try again')
        return redirect('accounts:signup_login')


def specific_exp(request, exp_name):
    experiment_title = exp_name
    # print('exp title', experiment_title)
    return render(request, 'experiment_process.html', 
                  {
                    'title': experiment_title,
                    'success':'',                                    
                   })

def explanation(request):
    return render(request, 'upload_explanation.html')

'''
def set_language(request):
    next_url = request.POST.get('next', '/')
    user_language = request.POST.get('language')
    translation.activate(user_language)
    response = redirect(next_url)
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, user_language)
    return response
'''

# 创建一个实验处理函数映射，使用哈希数组进行映射是比分支结构快很多的，一个是O(1)一个是O(n)
EXPERIMENT_PROCESSORS = {
    "恒温槽的装配与性能测定": "experiment.process_txt_files",
    "分光光度法测BPB的电离平衡常数": "experimentTwo.xlsx_process",
    "燃烧热的测定": "experimentThree.process_three",
    "双液系的气液平衡相图": "experimentFour.date_process",
    "旋光物质化学反应反应动力学研究": "experimentFive.experimentFive_process",
    "乙酸乙酯皂化反应动力学研究": "experimentSix.experimentSix_process",
    "聚乙二醇的相变热分析": "experimentSeven.experimentSeven_process",
    "稀溶液粘度法测定聚合物的分子量": "experimentEight.experimentEight_process",
}

def create_user_folder(user_id):
    """根据 user_id 创建专属的实验文件夹"""
    user_folder = os.path.join(settings.EXP_ROOT, str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

@csrf_protect 
def upload(request):
    if request.method == 'POST':
        experiment_title = request.POST.get('file_title')
        # print('experiment title', experiment_title)
        try:
            user_id = request.session.get('user_id')
            if not user_id:
                return redirect('accounts:signup_login')  # 如果未登录，跳转到登录页面

            # 创建用户的专属文件夹
            user_folder = create_user_folder(user_id)

            # 清理用户文件夹中的所有文件
            files_to_remove = glob.glob(os.path.join(user_folder, '*'))
            for file in files_to_remove:
                os.remove(file)
            
            # 删除下载文件（如果存在）
            remove_file_path = os.path.join(user_folder, "download.zip")
            if os.path.exists(remove_file_path):
                os.remove(remove_file_path)
                
            # 保存上传的文件到用户的专属文件夹中
            files = request.FILES.getlist('file')
            for file in files:
                file_name = file.name
                file_path = os.path.join(user_folder, file_name)
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
            
            experiment_title = request.POST.get('file_title')
            folder_start = os.path.join(settings.REFER_ROOT, experiment_title)
            folder_final = [f for f in os.listdir(folder_start) if os.path.isfile(os.path.join(folder_start, f))]

            # 检查上传文件的完整性
            for file in folder_final:
                if not os.path.exists(os.path.join(user_folder, file)):
                    return render(request, 'experiment_process.html', {
                        'title': experiment_title,
                        'success': '未包含完整文件或文件名、拓展名与模版文件不符',
                    })

            # 根据实验标题调用对应的处理函数
            processor = EXPERIMENT_PROCESSORS.get(experiment_title)
            if processor:
                module_name, function_name = processor.rsplit('.', 1)
                module = __import__(f"utils.{module_name}", fromlist=[function_name])
                func = getattr(module, function_name)
                func()  # 调用实验处理函数
            else:
                return render(request, 'experiment_process.html', {
                    'title': experiment_title,
                    'success': '未知实验类型',
                })

            # 删除上传的文件
            for file in files:
                file_path = os.path.join(user_folder, file.name)
                if os.path.exists(file_path):
                    os.remove(file_path)

            return render(request, 'experiment_process.html', {
                'title': experiment_title,
                'success': '数据处理成功',
            })
        except Exception as e:
            # 捕获异常并记录错误
            # print(f"Error occurred: {e}")
            return render(request, 'experiment_process.html', {
                'title': experiment_title,
                'success': '数据处理失败，发生异常。',
            })
    else:
        return render(request, 'exp_index.html')
    

@csrf_protect
def download(request):
    try:
        user_id = request.session.get('user_id')
        if not user_id:
            return redirect('accounts:signup_login')  # 如果未登录，跳转到登录页面

        # 用户专属文件夹路径
        user_folder = os.path.join(settings.EXP_ROOT, str(user_id))

        # 如果用户文件夹中没有文件，则返回错误信息
        if not os.path.exists(user_folder) or not os.listdir(user_folder):
            return HttpResponse("No files available for download", content_type="text/plain")

        # 创建一个压缩包，将用户文件夹中的所有文件压缩
        zip_file_name = 'download'
        zip_file_path = os.path.join(settings.UPLOAD_ROOT, f'{zip_file_name}.zip')
        shutil.make_archive(os.path.join(settings.UPLOAD_ROOT, zip_file_name), 'zip', user_folder)

        # 读取压缩文件并发送给用户
        with open(zip_file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/zip')
            response['Content-Disposition'] = f'attachment; filename="{zip_file_name}.zip"'
            response['Content-Length'] = os.path.getsize(zip_file_path)
            return response
    except Exception as e:
        # 捕获所有异常并返回错误信息
        # print(f"Error occurred during download: {e}")
        return HttpResponse("An error occurred during the download process.", content_type="text/plain")