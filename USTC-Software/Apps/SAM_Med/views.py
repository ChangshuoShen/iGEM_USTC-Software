import os
from pathlib import Path
import requests
from django.shortcuts import render, redirect, HttpResponse
from django.conf import settings
from django.contrib import messages
from Apps.accounts.models import User
import shutil

def check_user_login(request):
    """检查用户是否已登录"""
    user_id = request.COOKIES.get('user_id')
    email = request.COOKIES.get('email')

    # 如果没有找到用户信息，重定向到登录页面
    if not user_id or not email:
        messages.warning(request, 'Please login before using Image Segment Tools')
        return None

    # 检查用户是否存在于数据库中
    try:
        user = User.get_user_by_email(email)
        if user and user.id == int(user_id):
            # 用户已登录，返回用户信息
            return user
        else:
            # 用户信息无效，清除cookie并重定向
            messages.warning(request, 'Please login before using Physical Chemistry Experiment Tools')
            response = redirect('accounts:signup_login')
            response.delete_cookie('user_id')
            response.delete_cookie('email')
            return response
    except Exception as e:
        # 处理异常，显示错误消息并重定向
        messages.error(request, f'Error occured: {e}. Please try again')
        return redirect('accounts:signup_login')

def sam_index_2d(request):
    check = check_user_login(request)
    if isinstance(check, HttpResponse): # HttpResponse是所有相应的基类
        return check

    # 如果用户已登录，渲染3D图像分割前端页面
    return render(request, 'sam2d.html')

def sam_index_3d(request):
    check = check_user_login(request)
    if isinstance(check, HttpResponse):
        return check

    # 如果用户已登录，渲染3D图像分割前端页面
    return render(request, 'sam3d.html')

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

# 调用SAM_Med2D的模型
def process_image_2d(
    input_image_path: str,
    output_image_path: str,
    api_url: str='http://127.0.0.2:8000/infer'
):
    input_file = Path(input_image_path)
    if not input_file.is_file():
        raise FileNotFoundError(f'file {input_file} not found')
    # 构建请求
    data = {
        'input_path': str(input_file),
        'output_path': str(output_image_path)
    }
    try:
        response = requests.post(api_url, json=data)
        # 检查响应状态码
        if response.status_code == 200:
            print(f"the processed image has been saved at {output_image_path}")
            return True
        else:
            print(f"Request Error, code: {response.status_code}, response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"request error: {e}")
        return False

def upload_image2d(request):
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
            input_dir = os.path.join(user_folder, 'sam2d_input')
            output_dir = os.path.join(user_folder, 'sam2d_output')
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
            input_image_url = f'/media/{user_id}/sam2d_input/{uploaded_image.name}'
            output_path = os.path.join(output_dir, uploaded_image.name)
            if process_image_2d(input_path, output_path):
                # 设置结果图像 URL
                output_image_url = f'/media/{user_id}/sam2d_output/{uploaded_image.name}'
                # result_image_url = temp_output_path
            else:
                output_image_url = None
                error_message = 'Image processing failed.'
        else:
            error_message = 'No image file was selected.'

    return render(request, 'sam2d.html', {
        'input_image_url': input_image_url,
        'output_image_url': output_image_url,
        'error_message': error_message
    })

# 类似的，添加一个处理3D图像的方法
def process_image_3d(
    input_image_path: str,
    gt_image_path: str,
    output_image_path: str,
    roi_index: int,
    api_url: str = 'http://127.0.0.3:8000/infer'  # 使用你在 curl 中的 api_url
):
    input_file = Path(input_image_path)
    gt_file = Path(gt_image_path)
    
    if not input_file.is_file():
        raise FileNotFoundError(f'Input file {input_file} not found')
    
    if not gt_file.is_file():
        raise FileNotFoundError(f'Ground truth file {gt_file} not found')

    # 构建请求数据
    data = {
        "img_path": str(input_file),  # 输入图像路径
        "gt_path": str(gt_file),      # GT 图像路径
        "category_index": roi_index,  # ROI 索引
        "output_path": output_image_path  # 输出路径
    }

    try:
        # 发送 POST 请求，使用 json 参数
        response = requests.post(api_url, json=data)
        
        # 检查响应状态码
        if response.status_code == 200:
            result = response.json()
            if "output_path" in result:
                print(f"The processed 3D image has been saved at {result['output_path']}\n\n\n")
                return True
            else:
                print("Output path not found in response.")
                return False
        else:
            print(f"Request Error, code: {response.status_code}, response: {response.text}\n\n\n")
            return False
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return False

def upload_image3d(request):
    error_message = None
    input_image_url = None
    output_image_url = None

    if request.method == 'POST':
        # 检查用户是否已登录
        user_id = request.session.get('user_id')
        if not user_id:
            messages.warning(request, 'Please login before uploading images.')
            return redirect('accounts:signup_login')

        input_file = request.FILES.get('input')
        gt_file = request.FILES.get('gt')
        roi_index = request.POST.get('roi_index', 0)  # 从表单中获取 ROI 索引

        if input_file and gt_file:
            # 创建用户的专属文件夹
            user_folder = create_user_folder(user_id)
            # 确保 temp 和 output 目录存在
            input_dir = os.path.join(user_folder, 'sam3d_input')
            output_dir = os.path.join(user_folder, 'sam3d_output')
            
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            # 清除 temp 和 output 目录中的所有文件
            clear_directory(input_dir)
            clear_directory(output_dir)

            # 保存上传的文件到临时位置
            input_path = os.path.join(input_dir, f"image_{input_file.name}")
            gt_path = os.path.join(input_dir, f"gt_{gt_file.name}")

            with open(input_path, 'wb+') as destination:
                for chunk in input_file.chunks():
                    destination.write(chunk)

            with open(gt_path, 'wb+') as destination:
                for chunk in gt_file.chunks():
                    destination.write(chunk)

            # 处理图像
            input_image_url = f'/media/{user_id}/sam3d_input/image_{input_file.name}'
            output_path = os.path.join(output_dir, input_file.name)
            if process_image_3d(input_path, gt_path, output_path, roi_index):
                # 设置结果图像 URL
                output_image_url = f'/media/{user_id}/sam3d_output/{input_file.name}'
            else:
                error_message = '3D image processing failed.'
        else:
            error_message = 'Input file or ground truth file missing.'

    return render(request, 'sam3d.html', {
        'input_image_url': input_image_url,
        'output_image_url': output_image_url,
        'error_message': error_message
    })