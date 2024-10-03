import os
from django.shortcuts import render, redirect, HttpResponse
from django.http import JsonResponse
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
        messages.warning(request, 'Please login first for using scRNA-seq')
        return redirect('accounts:signup_login')

    # 检查用户是否存在于数据库中
    try:
        user = User.get_user_by_email(email)
        if user and user.id == int(user_id):
            # 用户已登录，继续渲染页面
            return render(request, 'upload.html')
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
        clear_directory(user_folder)
        uploaded_files = request.FILES.getlist('data_files')
        adata_list = []
        
        # 检查是否上传了文件
        if uploaded_files:
            # 如果上传了文件，使用这部分做分析
            for data_file in uploaded_files:
                file_path = os.path.join(user_folder, data_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in data_file.chunks():    
                        destination.write(chunk)
                # 读取上传的h5ad文件并添加到adata_list
                adata_list.append(sc.read_h5ad(file_path))

        else:
            selected_datasets = request.POST.getlist('datasets')
            if selected_datasets:
                # 根据选中的数据集加载相应的示例数据并添加到 adata_list
                if 'dataset1' in selected_datasets:
                    adata_list.append(sc.datasets.pbmc3k())
                if 'dataset2' in selected_datasets:
                    adata_list.append(sc.datasets.pbmc68k())
            else:
                # 这个时候就默认使用第一个adata数据创建一个adata_list了
                adata_list.append(sc.datasets.pbmc3k())
                
        # 开始选取几个步骤的方法
        batch_correction = request.POST.get('batch_correction')
        clustering = request.POST.get('clustering')
        enrichment_analysis = request.POST.get('enrichment_analysis')
        diff_expr = request.POST.get('diff_expr')
        trajectory_inference = request.POST.get('trajectory_inference')
        
        # 这里开始处理adata_list数据，直接实例化，然后计算所有内容
        

    return render(request, 'upload.html')



def upload_view(request):
    if request.method == 'POST':
        # 获取上传的文件
        files = request.FILES.getlist('data_files')
        if files:
            dataset_message = files[0].name
        # 获取选择的数据集
        else:
            
            datasets = request.POST.getlist('datasets')
            
            # 检查是否同时选中了两个数据集
            if 'dataset1' in datasets and 'dataset2' in datasets:
                dataset_message = "Both datasets selected: Dataset 1 and Dataset 2"
            elif 'dataset1' in datasets:
                dataset_message = "Dataset 1 selected"
            elif 'dataset2' in datasets:
                dataset_message = "Dataset 2 selected"
            else:
                dataset_message = "No dataset selected"
        
        # 获取用户选择的各步骤方法
        batch_correction = request.POST.get('batch_correction')
        clustering = request.POST.get('clustering')
        enrichment_analysis = request.POST.get('enrichment_analysis')
        diff_expr = request.POST.get('diff_expr')
        trajectory_inference = request.POST.get('trajectory_inference')
        
        # 构建返回的结果字典
        results = {
            "dataset_message": dataset_message,
            "batch_correction": batch_correction,
            "clustering": clustering,
            "enrichment_analysis": enrichment_analysis,
            "differential_expression": diff_expr,
            "trajectory_inference": trajectory_inference,
        }
        print(results)
        # 返回 JsonResponse
        return JsonResponse(results)

    return JsonResponse({"error": "Invalid request method."}, status=400)