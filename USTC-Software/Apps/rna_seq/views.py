import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from Apps.accounts.models import User
from django.core.files.storage import FileSystemStorage
import scanpy as sc
import shutil

def rna_index(request):
    # 检查用户的 session 或 cookie，验证是否已登录
    user_id = request.COOKIES.get('user_id')
    email = request.COOKIES.get('email')

    # 如果没有找到用户信息，重定向到登录页面
    if not user_id or not email:
        messages.warning(request, '请先登录以继续操作')
        return redirect('accounts:signup_login')

    # 检查用户是否存在于数据库中
    try:
        user = User.get_user_by_email(email)
        if user and user.id == int(user_id):
            # 用户已登录，继续渲染页面
            return render(request, 'rna_upload.html')
        else:
            # 用户信息无效，重定向到登录页面
            messages.warning(request, '请重新登录以使用 RNA 实验工具')
            response = redirect('accounts:signup_login')
            response.delete_cookie('user_id')
            response.delete_cookie('email')
            return response
    except Exception as e:
        # 异常处理，重定向到登录页面
        messages.error(request, '发生错误，请重试')
        return redirect('accounts:signup_login')


def create_user_folder(user_id):
    """根据 user_id 创建专属的实验文件夹"""
    user_folder = os.path.join(settings.MEDIA_ROOT, 'rna_seq', str(user_id))
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

def upload_file(request):
    
    if request.method == 'POST':
        user_id = request.session.get('user_id')
        if not user_id:
            messages.warning(request, 'Please login before uploading images.')
            return redirect('accounts:signup_login')
        
        user_folder = create_user_folder(user_id)
        uploaded_files = request.FILES.getlist('data_files')
        # Check if a file was uploaded
        
        if uploaded_files:
            # 如果上传了文件，使用这部分做分析
            adata_list = []
            # If a file is uploaded, handle the file storage
            for data_file in uploaded_files:
                file_path = os.path.join(user_folder, data_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in data_file.chunks():
                        destination.write(chunk)
                adata_list.append(
                    sc.read_h5ad(file_path)
                )

            # Proceed with processing the uploaded file
            # [Continue with your data processing logic]

            return render(request, 'upload.html', {'file_url': file_url})

        # If no file was uploaded, check for dataset selection
        elif 'datasets' in request.POST:
            dataset_choice = request.POST.get('datasets')

            # Load the selected dataset using Scanpy
            if dataset_choice == 'dataset1':
                adata = sc.datasets.pbmc3k()
            elif dataset_choice == 'dataset2':
                adata = sc.datasets.pbmc68k()

            # Proceed with processing the selected dataset
            # [Continue with your data processing logic]
            return render(request, 'upload.html', {'dataset': dataset_choice})

        # If neither file upload nor dataset selection, use the default dataset (pbmc3k)
        else:
            adata = sc.datasets.pbmc3k()

            # Proceed with processing the default dataset
            # [Continue with your data processing logic]
            return render(request, 'upload.html', {'dataset': 'default_pbmc3k'})

    return render(request, 'upload.html')