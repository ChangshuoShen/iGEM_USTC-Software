import medim # medical image
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk

def random_sample_next_click(prev_mask, gt_mask):
    """
    Randomly sample one click from ground-truth mask and previous seg mask
    随机从真实标签掩码和之前的分割掩码中采样一个点击点。
    * ground-truth: 指的是图像数据中真实存在的标签或标注信息，通常由专家手动标注或经过验证的自动化系统生成
    * mask: 是一种二进制或多类别的图像，通常用于标识图像中的特定区域
    Arguements:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
                    先前掩码，通称用于标识图像中的特定区域
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
                    真实标签掩码
    """
    prev_mask = prev_mask > 0   # 将之前的掩码转换为bool值
    true_masks = gt_mask > 0    # 获取真实值掩码

    if (not true_masks.any()):
        # 一个真值都没有，抛出异常
        raise ValueError("Cannot find true value in the ground-truth!")

    # 计算假阴性（真实存在但之前预测为不存在的区域）和假阳性（真实不存在但是预测为存在的区域）掩码
    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    # 用于采样的区域掩码取自假阴性或假阳性的部分
    to_point_mask = torch.logical_or(fn_masks, fp_masks)
    # 找出所有满足条件的点坐标并随机选择一个点
    all_points = torch.argwhere(to_point_mask)
    point = all_points[np.random.randint(len(all_points))]

    # 判断选出的这个点是不是假阴性
    if fn_masks[point[0], point[1], point[2]]:
        is_positive = True  # 假阴性标记为正类
    else:
        is_positive = False # 否则标记为负类

    # 将点坐标转化为所需的形状
    sampled_point = point.clone().detach().reshape(1, 1, 3)
    # 转化为张量
    sampled_label = torch.tensor([
        int(is_positive),
    ]).reshape(1, 1)

    return sampled_point, sampled_label


def sam_model_infer(model,
                    roi_image,
                    prompt_generator=random_sample_next_click, # 这个只在roi_gt不是None的时候才会使用
                    roi_gt=None,
                    prev_low_res_mask=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)
    在 SAM-Med3D 模型上进行推理，输入点和对应的标签（每个点的正负标签）。
    
    参数：
        - model: (torch.nn.Module) 已训练的 SAM-Med3D 模型，用于进行推理。
        - roi_image: (torch.Tensor) 裁剪后的输入图像，形状为 [1, 1, 128, 128, 128]。
        - prompt_generator: (Callable) 生成提示点的函数，默认为 random_sample_next_click。
        - roi_gt: (torch.Tensor) 可选的真实标签，形状为 [1, 1, 128, 128, 128]，用于生成提示点。
        - prev_low_res_mask: (torch.Tensor) 之前生成的低分辨率掩码，形状为 [1, 1, D/4, H/4, W/4]，可选。

    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    返回：
        - medsam_seg_mask: (np.array) 生成的分割掩码，形状为 (64, 64, 64)，每个像素为 0 或 1。
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    model = model.to(device)

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor) # 编码图像，获取特征

        # 初始化点坐标和标签
        points_coords, points_labels = torch.zeros(
            1, 0, 3).to(device), torch.zeros(1, 0).to(device)
        new_points_co, new_points_la = torch.Tensor(
            [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64)
        
        if (roi_gt is not None):
        # 如果提供了真实标签，就生成新提示点
            # 如果没有提供前一个低分辨率掩码，则初始化为全零
            prev_low_res_mask = prev_low_res_mask if (
                prev_low_res_mask is not None) else torch.zeros(
                    1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                    4, roi_image.shape[4] // 4)
            # 使用提示生成器生成新点坐标和标签
            new_points_co, new_points_la = prompt_generator(
                torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
            new_points_co, new_points_la = new_points_co.to(
                device), new_points_la.to(device)
            
        # 将新生成的点坐标和标签添加到现有的点集合中
        points_coords = torch.cat([points_coords, new_points_co], dim=1)
        points_labels = torch.cat([points_labels, new_points_la], dim=1)

        # 编码提示点
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            # masks=prev_low_res_mask.to(device),
            masks=None,
        )

        # 解码生成低分辨率掩码
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
        )

        # 将低分辨率掩码插值到原图大小
        prev_mask = F.interpolate(low_res_masks,
                                  size=roi_image.shape[-3:],
                                  mode='trilinear',
                                  align_corners=False)

    # 将概率转换为掩码
    medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)用sigmoid获取概率
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze() # 转换为numpy数组并去除多余维度
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8) # 根据阈值生成最终的分割掩码

    return medsam_seg_mask


def resample_nii(input_path: str,
                 output_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 n=None,
                 reference_image=None, # 注意这个地方可以没有reference_image
                 mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.
    对指定的 nii.gz 文件进行重新采样，调整到目标的体素间距。

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    目标的体素间距，默认是 (1.5, 1.5, 1.5)。
    """
    # Load the nii.gz file using torchio / 读取nii.gz文件，并生成一个 subject 对象
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    
    # 使用 torchio 进行重新采样，指定目标体素间距和插值模式
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    if (n != None):
        image = resampled_subject.img
        tensor_data = image.data
        if (isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[
            1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    save_image.save(output_path)


def read_data_from_nii(img_path, gt_path):
    # 从指定的 NIfTI 图像和标签文件中读取数据，并进行预处理，包括裁剪和归一化
    # sitk: SimpleITK (Insight Segmentation and Registration Toolkit)，是一个用于医学图像处理的库
    sitk_image = sitk.ReadImage(img_path)
    sitk_label = sitk.ReadImage(gt_path)

    # 确保图像和标签的原点，方向一致
    if sitk_image.GetOrigin() != sitk_label.GetOrigin():
        sitk_image.SetOrigin(sitk_label.GetOrigin())
    if sitk_image.GetDirection() != sitk_label.GetDirection():
        sitk_image.SetDirection(sitk_label.GetDirection())

    # 转为numpy格式
    sitk_image_arr, _ = sitk_to_nib(sitk_image)
    sitk_label_arr, _ = sitk_to_nib(sitk_label)

    # 创建subject对象，包括图像和标签
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=sitk_image_arr),
        label=tio.LabelMap(tensor=sitk_label_arr),
    )
    
    # 使用torchio进行裁减和填充，目标尺寸是(128, 128, 128)
    crop_transform = tio.CropOrPad(mask_name='label',target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    # 如果裁剪或填充参数为 None，设置为默认值
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    # 组合裁剪、填充和归一化变换
    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)  # 应用变换，生成ROI

    # 获取图像和标签的ROI，并增加一个通道维度
    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    
    # 计算原始ROI的偏移量，用于后续映射回原始图像空间
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + 128 - padding_params[4] - padding_params[5],
    )
    # 保存图像的元数据信息
    meta_info = {
        "image_path": img_path,
        "image_shape": sitk_image_arr.shape[1:],
        "origin": sitk_label.GetOrigin(),
        "direction": sitk_label.GetDirection(),
        "spacing": sitk_label.GetSpacing(),
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi": ori_roi_offset,
    }
    return (
        img3D_roi,  # 返回图像ROI
        gt3D_roi,   # 返回标签ROI
        meta_info,  # 返回元数据信息
    )


def read_data_from_nii_only_image(img_path):
    # 从指定的 NIfTI 图像和标签文件中读取数据，并进行预处理，包括裁剪和归一化
    # sitk: SimpleITK (Insight Segmentation and Registration Toolkit)，是一个用于医学图像处理的库
    sitk_image = sitk.ReadImage(img_path)
    # 转为numpy格式
    sitk_image_arr, _ = sitk_to_nib(sitk_image)

    # 创建subject对象，只有图像
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=sitk_image_arr)
    )
    # 使用torchio进行裁减和填充，目标尺寸是(128, 128, 128)
    crop_transform = tio.CropOrPad(target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(subject)
    # 如果裁剪或填充参数为 None，设置为默认值
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    # 组合裁剪、填充和归一化变换
    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)  # 应用变换，生成ROI

    # 获取图像的ROI，并增加一个通道维度
    img3D_roi= subject_roi.image.data.clone().detach().unsqueeze(1)
    
    # 计算原始ROI的偏移量，用于后续映射回原始图像空间
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + 128 - padding_params[4] - padding_params[5],
    )
    # 保存图像的元数据信息
    meta_info = {
        "image_path": img_path,
        "image_shape": sitk_image_arr.shape[1:],
        "origin": sitk_image.GetOrigin(),
        "direction": sitk_image.GetDirection(),
        "spacing": sitk_image.GetSpacing(),
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi": ori_roi_offset,
    }
    return (
        img3D_roi,  # 返回图像ROI
        meta_info,  # 返回元数据信息
    )

def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio 会将 1xHxWxD 转换为 DxWxH
    # 所以需要压缩维度并重新转置回 HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)  # 转回图像格式
    
    # 转换并设置元数据信息
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def data_preprocess(img_path, gt_path=None, category_index=None):
    # 这个函数是生成重采样之后的函数
    target_img_path = osp.join(
        osp.dirname(img_path),
        osp.basename(img_path).replace(".nii.gz", "_resampled.nii.gz"))
    # target_gt_path = osp.join(
    #     osp.dirname(gt_path),
    #     osp.basename(gt_path).replace(".nii.gz", "_resampled.nii.gz"))
    
    resample_nii(img_path, target_img_path)
    # resample_nii(gt_path, target_gt_path,
    #             n=category_index, # 这之后，我们不再让用户输入category_index，调用的时候直接别提供了
    #             reference_image=tio.ScalarImage(target_img_path),
    #             mode="nearest")
    
    # 从重采样之后的图像和标签中读取感兴趣的数据和标签
    # roi_image, roi_label, meta_info = read_data_from_nii(
    #     target_img_path, target_gt_path)  
    roi_image, meta_info = read_data_from_nii_only_image(target_img_path)
    # return roi_image, roi_label, meta_info
    return roi_image, meta_info


def data_postprocess(roi_pred, meta_info, output_path, ori_img_path):
    # 将模型预测的局部感兴趣区域(ROI)的结果映射回原始的3D图像空间
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    pred3D_full = np.zeros(meta_info["image_shape"])
    
    # 从 meta_info 中提取出原始ROI（感兴趣区域）的坐标范围
    ori_roi = meta_info["ori_roi"]
    pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                ori_roi[4]:ori_roi[5]] = roi_pred

    # 读取原始图像的 NIfTI 文件，获取其元数据信息
    sitk_image = sitk.ReadImage(ori_img_path)
    
    # 获取原始图像的元数据信息（形状、原点、方向、间距等
    ori_meta_info = {
        "image_path": ori_img_path,
        "image_shape": sitk_image.GetSize(),
        "origin": sitk_image.GetOrigin(),
        "direction": sitk_image.GetDirection(),
        "spacing": sitk_image.GetSpacing(),
    }
    
    # 将ROI预测结果插值回原始图像的形状
    # F.interpolate 用于调整预测图像的大小，使其与原始图像匹配，最近邻插值法保证了类别的保留
    pred3D_full_ori = F.interpolate(
        torch.Tensor(pred3D_full)[None][None],
        size=ori_meta_info["image_shape"],
        mode='nearest').cpu().numpy().squeeze()
    save_numpy_to_nifti(pred3D_full_ori, output_path, meta_info)

def process_med_image(img_path, 
                    #   gt_path, 
                    #   category_index, 
                      output_path, 
                      ckpt_path='/home/shenc/pth_model/sam_med3d_turbo.pth'):
    """
    处理医学图像并生成预测结果

    参数:
    img_path (str): 图像文件路径.
    gt_path (str): 与图像对应的 ground truth 标签路径.
    category_index (int): gt 注释中的目标类别索引.
    output_dir (str): 保存预测结果的目录.
    ckpt_path (str): 模型的权重文件路径,
    """

    # 读取和预处理输入数据
    # roi_image, roi_label, meta_info = data_preprocess(
    #     img_path, gt_path, category_index=category_index)
    roi_image, meta_info = data_preprocess(img_path)
    
    # 预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = medim.create_model("SAM-Med3D",
                               pretrained=True,
                               checkpoint_path=ckpt_path).to(device)
    
    roi_image = roi_image.to(device)
    roi_pred = sam_model_infer(model, roi_image, roi_gt=None)
    data_postprocess(roi_pred, meta_info, output_path, img_path)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# 定义请求体数据模型
class InferenceRequest(BaseModel):
    img_path: str
    # gt_path: str
    # category_index: int
    output_path: str

@app.post("/infer")
async def infer(request: InferenceRequest):
    # 从请求体中提取参数
    img_path = request.img_path
    output_path = request.output_path

    # 校验参数
    if not osp.exists(img_path):
        raise HTTPException(status_code=400, detail="Invalid file path provided.")
    
    try:
        # 调用处理函数
        process_med_image(img_path, output_path)
        return {"message": "Inference completed", "output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.3", port=8000)