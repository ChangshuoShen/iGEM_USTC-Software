import onnxruntime as ort # ONNX运行时库
from fastapi import FastAPI, UploadFile, File # 导入FastAPI相关模块用于创建web服务
from PIL import Image # 图像处理库
from fastapi.responses import StreamingResponse # 用于流式响应
import io # 处理输入输出流
import numpy as np
from tqdm import tqdm # 进度条显示库
import cv2 # OpenCV图像处理库
from typing import Any, Union # 类型提示
from copy import deepcopy
import matplotlib.pyplot as plt
import os

app = FastAPI()

def show_mask(mask, ax, random_color=False):
    '''
    在轴上绘制mask
    '''
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) # 随机颜色
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6]) # 固定颜色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) # 将掩码与颜色结合
    ax.imshow(mask_image) # 在给定轴上显示掩码图像


def show_points(coords, labels, ax, marker_size=375):
    '''
    在轴上绘制点
    '''
    pos_points = coords[labels == 1] # 正标签点
    neg_points = coords[labels == 0] # 负标签点
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25) # 正标签点绘制为绿色
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white',  linewidth=1.25) # 负标签点绘制为红色


def show_box(box, ax):
    '''
    在轴上绘制边界框
    '''
    x0, y0 = box[0], box[1]  #左上角坐标和宽高控制位置
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2)) # 绘制矩形框
    

# 加载 ONNX 模型到显存
class SamEncoder:
    """Sam encoder model.

    In this class, encoder model will encoder the input image.
    编码器模型将对输入图像进行编码
    Args:
        model_path (str): sam encoder onnx model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        warmup_epoch (int): Warmup, if set 0,the model won`t use random inputs to warmup. default to 3.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 warmup_epoch: int = 3,
                 **kwargs):
        opt = ort.SessionOptions()  #onnxruntime中用于创建会话时配置选项的一个对象
        
        # 选设备
        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading encoder model...")
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)
        
        # 获取模型输入输出名称及形状
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        # 输入图像归一化常数
        self.pixel_mean = np.array([123.675, 116.28, 103.53])
        self.pixel_std = np.array([58.395, 57.12, 57.375])
        self.input_size = (self.input_shape[-1], self.input_shape[-2])

        # 如果需要，执行预热
        if warmup_epoch:
            self.warmup(warmup_epoch)

    def warmup(self, epoch: int) -> None:
        """warmup function
            在机器学习模型的推理过程中，"预热"（warmup）通常指的是在正式开始处理实际数据之前，先用一些虚拟或随机的数据来运行模型几次。
            作用
                * 资源分配和初始化
                * 缓存机制
                * JIT编译（即时）
                * 性能稳定性
        Args:
            epoch (int): warmup epoch.
        """
        x = np.random.random(self.input_shape).astype(np.float32) # 生成随机输入
        print("start warmup!")
        for i in tqdm(range(epoch)): # 这里用个进度条展示
            self.session.run(None, {self.input_name: x}) # 执行推理
        print("warmup finish!")

    def transform(self, img: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to the required input format for vit.
        将输入图像转换为vit所需输入格式
        Args:
            img (np.ndarray): input image, the image type should be BGR.

        Returns:
            np.ndarray: transformed image.
        """
        # BGR -> RGB
        input_image = img[..., ::-1]
        # Normalization
        input_image = (input_image - self.pixel_mean) / self.pixel_std
        # Resize
        input_image = cv2.resize(input_image, self.input_size, cv2.INTER_NEAREST)
        # HWC -> CHW
        input_image = input_image.transpose((2, 0, 1))
        # CHW -> NCHW
        input_image = np.expand_dims(input_image, 0).astype(np.float32)
        return input_image

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use `vit` to extract feature from transformed image.

        Args:
            tensor (np.ndarray): input image with BGR format.

        Returns:
            np.ndarray: image`s feature.
        """
        input_image = self.transform(tensor)
        assert list(input_image.shape) == self.input_shape # 确保输入形状正确
        feature = self.session.run(None, {self.input_name: input_image})[0] # 提取特征
        assert list(feature.shape) == self.output_shape # 确保输出形状正确
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        result = self._extract_feature(img)
        return result



class SamDecoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.
    Sam提示编码器和轻量级解码器
    Args:
        model_path (str): decoder model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        img_size: 256 default
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 img_size: int = 256,
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading decoder model...")
        self.mask_threshold = 0.5  # 掩码阈值
        self.img_size = (img_size, img_size)
        # 初始化ONNX会话
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

    def run(self,
            img_embeddings: np.ndarray, # 来自vit编码器的图像特征
            origin_image_size: Union[list, tuple], # 原始图像尺寸
            point_coords: Union[list, np.ndarray] = None, # 输入点坐标
            point_labels: Union[list, np.ndarray] = None, # 输入点标签
            boxes: Union[list, np.ndarray] = None, # 边界框
            mask_input: np.ndarray = None, # 掩码输入
            return_logits: bool = False): # 是否返回logits
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            img_embeddings (np.ndarray): the image feature from vit encoder.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=4 * embedding.size.

        Returns:
            the segment results.
        """
        
        # 检查是否提供了一个点或边界框
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        # 检查嵌入形状
        if img_embeddings.shape != (1, 256, 16, 16):
            raise ValueError("Got wrong embedding shape!")
        
        # 如果没有提供mask，就初始化一个0掩码
        if mask_input is None:
            mask_input = np.zeros((1, 1, 64, 64), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            # 检查掩码输入形状
            if mask_input.shape != (1, 1, 64, 64):
                raise ValueError("Got wrong mask!")
            
        # 确定点坐标和标签的格式正确
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        # 调整点坐标到新的图像尺寸 
        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, self.img_size).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            # 如果提供了边界框
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4  # 确保拿到4个坐标

            # 调整边框到新的图像尺寸
            boxes = self.apply_boxes(boxes, origin_image_size, self.img_size).reshape((1, -1, 2)).astype(np.float32)
            box_label = np.array([[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32).reshape((1, -1))

            # 将点坐标与边界框合并
            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        # 检查点和标签的形状
        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1
        print(f"point_coords={point_coords}, point_labels={point_labels}")

        # 输入字典
        input_dict = {
            "image_embeddings": img_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(origin_image_size, dtype=np.float32)
        }
        
        # 运行推理
        masks, iou_predictions, low_res_masks = self.session.run(None, input_dict)

        if not return_logits:
            # 应用sigmoid函数并根据阈值转换为二进制mask
            sigmoid_output = self.sigmoid(masks)
            masks = (sigmoid_output > self.mask_threshold).astype(np.float32)

        return masks[0], iou_predictions[0], low_res_masks[0]
    
    @staticmethod
    def sigmoid(x):
        return 0.5 * (np.tanh(0.5 * x) + 1)
    
    # 调整坐标到新尺寸
    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)



@app.get("/") # 这一句是FastAPI框架中用于定义路由route的装饰器语法
async def read_root():
    # async是python中定义异步函数的方式，异步编程是一种允许程序在等待某些操作完成的同时继续执行其他任务的编程范式
    return {"message": "Welcome to the FastAPI server!"}

# FastAPI 路由
@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    # 读取上传文件并解码为OpenCV图像
    file_bytes = np.frombuffer(await image.read(), np.uint8)
    img_file = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # 使用编码器绘画获取图像嵌入
    img_embeddings = encoder_session(img_file)
    # 原始图像尺寸
    origin_image_size = img_file.shape[:2]
    # 定义点坐标和标签
    point_coords = np.array([[162, 127]], dtype=np.float32)
    # 执行解码器会话
    point_labels = np.array([1], dtype=np.float32)
    masks, _, logits = decoder_session.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels
    )
    # 绘制结果
    plt.figure(figsize=(10, 10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')
    # 保存图像到内存缓冲区
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # 作为PNG图像流式响应返回
    return StreamingResponse(buf, media_type="image/png")
"""
    plt.show()

    '''Optimizing Segmentation Results by Point Interaction'''
    new_point_coords = np.array([[169, 140]], dtype=np.float32)
    new_point_labels = np.array([0], dtype=np.float32)
    point_coords = np.concatenate((point_coords, new_point_coords))
    point_labels = np.concatenate((point_labels, new_point_labels))
    mask_inputs = 1. / (1. + np.exp(-logits.astype(np.float32)))

    masks, _, logits = decoder_session.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_inputs,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')

    plt.show()

    '''Specifying a specific object with a bounding box'''
    boxes = np.array([135, 100, 180, 150])
    masks, _, _ = decoder_session.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        boxes=boxes,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_box(boxes, plt.gca())
    plt.axis('off')

    plt.show()

"""
# 运行 FastAPI
if __name__ == "__main__":
    save_path = os.path.join("work_dir", 'ort_demo_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # encoder_session = SamEncoder(model_path="./onnx_model/sam-med2d_b.encoder.onnx", device="cuda", warmup_epoch=3)
    encoder_session = SamEncoder(model_path="/home/shenc/onnx_model/sam-med2d_b.encoder.onnx", device="cuda", warmup_epoch=3)

    decoder_session = SamDecoder(model_path="/home/shenc/onnx_model/sam-med2d_b.decoder.onnx")

    import uvicorn

    # uvicorn.run(app, host="172.17.0.6", port=36505)
    uvicorn.run(app, host='127.0.0.1', port=8002)
