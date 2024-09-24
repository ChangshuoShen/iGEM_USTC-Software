from fastapi import FastAPI, Request,File, UploadFile, HTTPException  # 确保UploadFile被正确导入
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn

import json
import jsonify
from datetime import datetime
import requests
from requests.models import Response

import torch
import torchvision.transforms as transforms

import os
import shutil
from uuid import uuid4
from PIL import Image
from jinja2 import Template, Environment, FileSystemLoader
from flask import render_template

import onnxruntime as ort
import numpy as np
from typing import Union
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt

from main import SamDecoder, SamEncoder

# 全局变量
HOST = '127.0.0.1'
PORT = '8080'
port = 8080
DOMAIN = f"http://{HOST}:{PORT}"

# 配置上传目录，全局设置上传目录
UPLOAD_DIRECTORY = "upload"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
PEOCESSED_DIRECTORY = "processed"
if not os.path.exists(PEOCESSED_DIRECTORY):
    os.makedirs(PEOCESSED_DIRECTORY)

# 原先是传入图片路径，输出处理后的图片的url并保存图片，可以运行；后改称输入file类型

def load_model_path_or_repo_id(input_path_or_id: str):
    """
    根据提供的输入加载模型路径或模型 ID。
    
    :param input_path_or_id: 输入的路径或模型 ID
    :return: 返回处理后的模型路径或模型 ID
    """
    if os.path.isdir(input_path_or_id):
        # 如果输入的是一个目录，则返回该目录路径
        return input_path_or_id
    else:
        # 否则，假设输入的是 Hugging Face Model Hub 上的模型 ID
        return input_path_or_id

# 示例使用
input_path_or_id = "/path/to/local/folder"  # 示例本地路径
# input_path_or_id = "hf-username/model-name"  # 示例模型 ID

# 处理输入
processed_input = load_model_path_or_repo_id(input_path_or_id)
print(processed_input)

# 设置设备参数
# Set the device      
DEVICE = "cpu" if torch.backends.mps.is_available() else "mps"
# To run PyTorch code on the GPU, use torch.device(“mps”) analogous to torch.device(“cuda”) on an Nvidia GPU. 
print(f"Using device: {DEVICE}")
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

app = FastAPI() # 创建FastAPI应用
application = app # 将 app 对象暴露为 ASGI 应用程序

# 挂载静态文件目录
app.mount("/upload", StaticFiles(directory=UPLOAD_DIRECTORY), name="upload")
app.mount("/processed", StaticFiles(directory="processed"), name="processed")


# 之前是路由的问题，没有转到/upload下，因此直接把所有东西都写在/下
# 图片上传
@app.get("/")
async def upload_image(request: Request):
    
    global encoder, decoder # 声明全局变量以便在函数内部使用模型
    print('sam decoder: ', SamDecoder)
    encoder = SamEncoder(
        model_path="../../SAM-Med2D/onnx_model/sam-med2d_b.encoder.onnx",
        warmup_epoch=3
    )
    print(f'*****encoder: {encoder}*****')
    decoder = SamDecoder(
        model_path="../../SAM-Med2D/onnx_model/sam-med2d_b.decoder.onnx",
    )
    print(f'*****decoder: {decoder}*****')


    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    image_path = json_post_list.get("image_path")  # 获取请求中的image_path
    # history = json_post_list.get("history", [])  # 获取请求中的历史记录

    file = open(image_path, 'rb')
    original_file_name = os.path.basename(file.name)
    # original_file_name = file.name
    # 读取文件内容
    content = file.read()
    print(f"file: {file}\n")

    # 生成唯一的文件名
    file_extension = original_file_name.split(".")[-1]
    if file_extension not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_name = f"{uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    # 将内容保存到新文件
    with open(file_path, 'wb') as new_file:
        new_file.write(content)
    print(f"文件已保存为 {file_path}")

    # 👇都来自SAM main.py 用已经在main中加载好.onnx的模型处理图片👇

    # SAM自带transform预处理图片
    '''Specifying a specific object with a point'''
    img_file = cv2.imread(file_path)
    img_embeddings = encoder(img_file)
    origin_image_size = img_file.shape[:2]

    point_coords = np.array([[162, 127]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.float32)
    print(f"point_labels: {point_labels}\n")

    masks, _, logits = decoder.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels
    )

    def show_mask(mask, ax, file_name, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        print(f"mask_image:{np.shape(mask_image)}")
        # np.save(f'{file_name}.npy', mask_image)
        ax.imshow(mask_image)
    
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        print(f"pos_points:{pos_points}")
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        

    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca(),file_name)
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')
    # 保存有point和box的处理后的图片到PEOCESSED_DIRECTORY
    path_to_image = os.path.join(PEOCESSED_DIRECTORY, f"seg_{file_name}")
    plt.savefig(path_to_image)

    # 👆都来自SAM main.py 用已经在main中加载好.onnx的模型处理图片👆
    
    # 生成图片url
    path = f"/{PEOCESSED_DIRECTORY}/seg_{file_name}"
    seg_img_url = f"{DOMAIN}{path}"
    print('seg img url', seg_img_url)
    # response = {"image_url": seg_img_url}
    
    # 构建响应JSON
    # 构建响应内容字典
    now = datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    response_data = {
        "image_url": seg_img_url,
        "status": 200,
        "time": time
    }
    print('response data', response_data)
    print('type', type(response_data))
    # 获取预测结果
    # class_id = predicted.item()
    # class_name = f"Class ID: {class_id}"  # 在实际应用中，您应该有一个类名列表

    # 删除原文件
    # os.remove(file_path)

    # 返回响应
    # return FileResponse(path_to_image) # 直接返回静态文件，或StreamingResponse逐块流式传输文件
    #return json.dumps({"response": response})
    # 返回 JSON 响应
    return JSONResponse(content=response_data, status_code=200)

# 启动服务器
# 主函数入口
if __name__ == "__main__":
    # 导入已经预训练过的resnet18模型
    # 由于暂时不需要模型微调而直接下载官方预训练参数，故而注释掉
    # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 这是官方预训练参数地址
    # model.load_state_dict(torch.load('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    
    # 在主函数加载模型
    # Initialize the SAM-Med2D onnx model
    print(SamDecoder)
    encoder = SamEncoder(
        model_path="../../SAM-Med2D/onnx_model/sam-med2d_b.encoder.onnx",
        warmup_epoch=3
    )
    print("*"*5,encoder,"*"*5)
    decoder = SamDecoder(
        model_path="../../SAM-Med2D/onnx_model/sam-med2d_b.decoder.onnx",
    )
    print("*"*5,decoder,"*"*5)

    # 加载预训练的分词器和模型
    # 但所处理的数据并非文本数据，应该也可以不使用分词器

    # 启动FastAPI应用
    uvicorn.run(app='api:application', host=HOST, port=port, reload=True, workers=1)  # 在指定端口和主机上启动应用

