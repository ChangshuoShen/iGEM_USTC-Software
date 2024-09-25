import os
from pathlib import Path
import requests
from django.shortcuts import render, redirect
from django.conf import settings


def sam_index(request):
    # return HttpResponse('this is the sam index')
    return render(request, 'image_segment.html')

# 调用SAM_Med2D的模型
def process_image(
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
# process_image(input_image_path, output_image_path)

def upload_image(request):
    result_image_url = None
    error_message = None

    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')
        if uploaded_image:
            # 确保 temp 和 output 目录存在
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # 保存上传的文件到临时位置
            temp_input_path = os.path.join(temp_dir, uploaded_image.name)
            with open(temp_input_path, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)

            # 处理图像
            temp_output_path = os.path.join(output_dir, uploaded_image.name)
            if process_image(temp_input_path, temp_output_path):
                # 删除临时输入文件
                os.remove(temp_input_path)
                # 设置结果图像 URL
                result_image_url = f'/media/output/{uploaded_image.name}'
            else:
                error_message = 'Image processing failed.'
        else:
            error_message = 'No image file was selected.'

    return render(request, 'image_segment.html', {
        'result_image_url': result_image_url,
        'error_message': error_message
    })