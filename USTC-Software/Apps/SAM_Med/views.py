import os
from pathlib import Path
import requests
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from Apps.accounts.models import User
import shutil


def sam_index(request):
    # 先检查是否登陆成功，后面需要根据用户专门建文件夹存放实验数据和处理结果
    # 检查用户的 cookie 或 session，判断是否已登录
    user_id = request.COOKIES.get('user_id')
    email = request.COOKIES.get('email')
    
    # 如果没有找到用户信息，重定向到登录页面
    if not user_id or not email:
        messages.warning(request, 'Please login before using Image Segment Tools')
        return redirect('accounts:signup_login')
    
     # 检查用户是否存在于数据库中
    try:
        user = User.get_user_by_email(email)
        if user and user.id == int(user_id):
            # 用户已登录，继续渲染页面
            return render(request, 'sam.html')
        else:
            # 用户信息无效，重定向到登录页面
            messages.warning(request, 'Please login before using Physical Chemistry Experiment Tools')
            response = redirect('accounts:signup_login')
            response.delete_cookie('user_id')
            response.delete_cookie('email')
            return response
    except Exception as e:
        # 异常处理，重定向到登录页面
        messages.error(request, f'Error occured!!!:{e}: Please try again')
        return redirect('accounts:signup_login')
    # return HttpResponse('this is the sam index')
    

# 调用SAM_Med2D的模型
def process_image_2d(
    input_image_path: str,
    output_image_path: str,
    api_url: str='http://127.0.0.1:8002/infer'
    ):
    input_file = Path(input_image_path)
    if not input_file.is_file():
        raise FileNotFoundError(f'file {input_file} not found')
    
    # 构建请求
    with open(input_image_path, 'rb') as image_file:
        files = {'image': (input_file.name, image_file, 'image/png')}
        try:
            response = requests.post(api_url, files=files)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 确保输出目录存在
                output_dir = Path(output_image_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                # 保存图像
                with open(output_image_path, 'wb') as output_file:
                    output_file.write(response.content)
                print(f"the processed image has been saved at {output_image_path}")
                return True
            else:
                print(f"Request Error,code: {response.status_code}, response: {response.text}")
                return False
        except requests.RequestException as e:
            print(f"request error: {e}")
            return False

# 使用示例
# input_image_path = "./SAM-Med2D/data_demo/images/amos_0507_31.png"
# output_image_path = "/home/shenc/Desktop/result.png"
# process_image_2d(input_image_path, output_image_path)

def create_user_folder(user_id):
    """根据 user_id 创建专属的实验文件夹"""
    user_folder = os.path.join(settings.MEDIA_ROOT, str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def clear_directory(directory):
    """清除指定目录中的所有文件和子目录"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def upload_image(request):
    error_message = None

    if request.method == 'POST':
        # 检查用户是否已登录
        user_id = request.session.get('user_id')
        if not user_id:
            messages.warning(request, 'Please login before uploading images.')
            return redirect('accounts:signup_login')

        uploaded_image = request.FILES.get('image')
        if uploaded_image:
            # 创建用户的专属文件夹
            user_folder = create_user_folder(user_id)
            # 确保 temp 和 output 目录存在
            input_dir = os.path.join(user_folder, 'sam_input')
            output_dir = os.path.join(user_folder, 'sam_output')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            # 先清除 temp 和 output 目录中的所有文件
            clear_directory(input_dir)
            clear_directory(output_dir)

            # 保存上传的文件到临时位置
            input_path = os.path.join(input_dir, uploaded_image.name)
            with open(input_path, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)
            # 处理图像
            output_path = os.path.join(output_dir, uploaded_image.name)
            if process_image_2d(input_path, output_path):
                # 设置结果图像 URL
                input_image_url = f'/media/{user_id}/sam_input/{uploaded_image.name}'
                output_image_url = f'/media/{user_id}/sam_output/{uploaded_image.name}'
                # result_image_url = temp_output_path
            else:
                error_message = 'Image processing failed.'
        else:
            error_message = 'No image file was selected.'

    return render(request, 'sam.html', {
        'input_image_url': input_image_url,
        'output_image_url': output_image_url,
        'error_message': error_message
    })