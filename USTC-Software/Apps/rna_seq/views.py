import os
from django.shortcuts import render, redirect, HttpResponse
from django.http import JsonResponse, StreamingHttpResponse, FileResponse
import subprocess
from django.core.paginator import Paginator
from django.contrib import messages
from django.conf import settings
from Apps.accounts.models import User
from django.core.files.storage import FileSystemStorage
import scanpy as sc
import shutil
from .utils import scRNAseqUtils
import zipfile

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
    display_path = os.path.join(settings.MEDIA_URL, 'rna_seq', str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder, display_path

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
        
        user_folder, display_path = create_user_folder(user_id)
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
            DATASET_PATHS = {
                'dataset1': '/var/www/media/rna_seq/public/pbmc3k_raw.h5ad',
                'dataset2': '/var/www/media/rna_seq/public/pbmc68k_reduced.h5ad',
            }
            selected_datasets = request.POST.getlist('datasets')
            if selected_datasets:
                # 根据选中的数据集加载相应的示例数据并添加到 adata_list
                for dataset in selected_datasets:
                    if dataset in DATASET_PATHS:
                        adata_list.append(sc.read_h5ad(DATASET_PATHS[dataset]))
            else:
                # 这个时候就默认使用第一个adata数据创建一个adata_list了
                adata_list.append(sc.read_h5ad(DATASET_PATHS['dataset1']))

        # 开始选取几个步骤的方法
        batch_correction = request.POST.get('batch_correction')
        clustering = request.POST.get('clustering')
        enrichment_analysis = request.POST.get('enrichment_analysis')
        diff_expr = request.POST.get('diff_expr')
        trajectory_inference = request.POST.get('trajectory_inference')
        
        # 这里开始处理adata_list数据，直接实例化，然后计算所有内容
        adata_process = scRNAseqUtils(
            adata_list,
            user_folder,
            display_path,
            batch_correction,
            clustering,
            enrichment_analysis,
            diff_expr,
            trajectory_inference
        )
        workflow_option = request.POST.get('workflow_option')
        if workflow_option == "qc_and_dim_reduction":
            # 执行简单的质量控制
            adata_process.qc_and_preprocessing()
        
        elif workflow_option == "cluster_analysis_and_annotation":
            # 执行质量控制和数据分析
            adata_process.qc_and_analysis_workflow()
            
        elif workflow_option == "full_workflow":
            # 执行完整的workflow
            adata_process.full_workflow()
                
        return render(
            request,
            'workflow.html',
            {
                'results': adata_process.results,
                'user_id': user_id,
            }
        )
    return render(request, 'upload.html')

def zip_user_folder(user_folder_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for foldername, subfolders, filenames in os.walk(user_folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, user_folder_path)  # 压缩时保留相对路径
                zip_file.write(file_path, arcname)
                
def download_user_folder(request, user_id):
    user_folder_path = f'/var/www/media/rna_seq/{user_id}/'
    zip_file_path = f'/var/www/media/rna_seq/zip_folder/User{user_id}_folder.zip'
    # 如果文件夹不存在，返回错误信息
    if not os.path.exists(user_folder_path):
        return HttpResponse("The folder does not exist.", status=404)
    # 压缩文件夹
    zip_user_folder(user_folder_path, zip_file_path)

    # 检查压缩文件是否成功生成
    if not os.path.exists(zip_file_path):
        return HttpResponse("Failed to create zip file.", status=500)

    # 发送文件作为下载响应
    response = FileResponse(open(zip_file_path, 'rb'))
    response['Content-Disposition'] = f'attachment; filename={user_id}_folder.zip'
    
    return response


# 试一下流式响应
import time

# 全局变量，记录处理进度
num_progress = 0

# 显示进度条页面
def show_progress1(request):
    return render(request, 'progress.html')

# 后台实际处理程序
def process_data(request):
    global num_progress
    num_progress = 0

    total = 10  # 模拟总任务数
    for i in range(total):
        # 模拟任务处理逻辑
        num_progress = (i + 1) * 100 / total  # 更新进度（百分比）
        time.sleep(3)  # 模拟耗时操作
        
    return JsonResponse({'status': 'completed'})

# 前端获取进度
def show_progress(request):
    global num_progress
    return JsonResponse({'progress': num_progress})