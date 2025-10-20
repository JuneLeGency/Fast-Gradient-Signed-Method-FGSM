import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image
from scripts.utils import get_cn_name_by_id, get_name_and_id_from_prediction
# --- 路径定义和模块导入 ---
# 定义项目中的关键路径，使路径管理更清晰健壮
BACKEND_DIR = os.path.dirname(__file__)


# --- 全局资源加载 ---
print("正在加载预训练模型和数据...")

# 加载模型和V1权重，以保持与'pretrained=True'相同的行为和结果
ALEXNET_WEIGHTS = models.AlexNet_Weights.IMAGENET1K_V1
ALEXNET_MODEL = models.alexnet(weights=ALEXNET_WEIGHTS).eval()

RESNET50_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1
RESNET_MODEL = models.resnet50(weights=RESNET50_WEIGHTS).eval()

# 获取与V1权重关联的官方预处理变换
RESNET50_PREPROCESS = RESNET50_WEIGHTS.transforms()
ALEXNET_PREPROCESS = ALEXNET_WEIGHTS.transforms()

print("模型和数据加载完成。")


# --- 预计算和缓存 ---
FGSM_GRADIENT = None
FGSM_INPUT_TENSOR = None


def precompute_fgsm_gradient():
    """仅在首次需要时计算并缓存FGSM的梯度。"""
    global FGSM_GRADIENT, FGSM_INPUT_TENSOR
    if FGSM_GRADIENT is not None:
        return

    image_path = os.path.join(BACKEND_DIR, 'images', 'panda.jpg')
    
    img_tensor = read_image(image_path)

    # 此处保持自定义的Resize以维持原始FGSM脚本的行为
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=ALEXNET_PREPROCESS.mean, std=ALEXNET_PREPROCESS.std),
    ])

    input_tensor = preprocess(img_tensor).unsqueeze(0)
    input_tensor.requires_grad = True
    FGSM_INPUT_TENSOR = input_tensor

    predictions = ALEXNET_MODEL(input_tensor)
    _, target_class_id = get_name_and_id_from_prediction(predictions, ALEXNET_WEIGHTS)
    target = torch.LongTensor([target_class_id])

    loss = torch.nn.CrossEntropyLoss()
    loss_val = loss(predictions, target)
    ALEXNET_MODEL.zero_grad()
    loss_val.backward()

    FGSM_GRADIENT = input_tensor.grad.data.sign()
    print("梯度计算并缓存完成。")


# --- 核心功能函数 ---

def predict_image(image_path):
    try:
        img_tensor = read_image(image_path)
        display_image = transforms.ToPILImage()(img_tensor)
    except (FileNotFoundError, RuntimeError):
        return None, f"错误：找不到或无法读取图像文件 {image_path}"

    # Get the height and width from the tensor shape (C, H, W)
    img_height, img_width = img_tensor.shape[1], img_tensor.shape[2]

    # Determine the crop size from the official transforms
    crop_size_val = RESNET50_PREPROCESS.crop_size
    if not isinstance(crop_size_val, int):
        crop_size_val = crop_size_val[0]

    # Conditionally apply preprocessing
    if (img_height, img_width) == (crop_size_val, crop_size_val):
        # For already-cropped images, just convert dtype and normalize
        preprocess = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=RESNET50_PREPROCESS.mean, std=RESNET50_PREPROCESS.std),
        ])
    else:
        # For other images, use the full official pipeline
        preprocess = RESNET50_PREPROCESS
        
    input_batch = preprocess(img_tensor).unsqueeze(0)

    with torch.no_grad():
        output = RESNET_MODEL(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    result_text = "--- Top 5 预测结果 ---\n"
    for i in range(top5_prob.size(0)):
        class_id = top5_catid[i].item()
        class_name_cn = get_cn_name_by_id(class_id, RESNET50_WEIGHTS)
        probability = top5_prob[i].item() * 100
        result_text += f"第{i+1}名: {class_name_cn:<10} 置信度: {probability:.2f}%\n"

    return display_image, result_text


def denormalize_image(tensor):
    mean = np.array(RESNET50_PREPROCESS.mean)
    std = np.array(RESNET50_PREPROCESS.std)
    tensor = tensor.clone().detach().squeeze(0).cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))


def generate_fgsm_attack(epsilon):
    precompute_fgsm_gradient()

    perturbation = epsilon * FGSM_GRADIENT
    adversarial_tensor = FGSM_INPUT_TENSOR.data + perturbation

    adversarial_output = ALEXNET_MODEL(adversarial_tensor)
    adv_prob, adv_catid_tensor = torch.topk(torch.nn.functional.softmax(adversarial_output[0], dim=0), 1)
    adv_class_id = adv_catid_tensor[0].item()

    original_pil = denormalize_image(FGSM_INPUT_TENSOR)
    adversarial_pil = denormalize_image(adversarial_tensor)

    perturbation_vis = perturbation.squeeze(0).detach().cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (
                perturbation_vis.max() - perturbation_vis.min())
    perturbation_pil = Image.fromarray((np.transpose(perturbation_vis, (1, 2, 0)) * 255).astype(np.uint8))

    # 原始结果文本
    with torch.no_grad():
        orig_output = ALEXNET_MODEL(FGSM_INPUT_TENSOR)
        orig_class_name, orig_class_id = get_name_and_id_from_prediction(orig_output, ALEXNET_WEIGHTS)
        orig_prob = torch.nn.functional.softmax(orig_output[0], dim=0)[orig_class_id].item() * 100
        original_text = f"原始预测: {orig_class_name}\n置信度: {orig_prob:.2f}%"

    # 攻击结果文本
    adv_class_name = get_cn_name_by_id(adv_class_id, ALEXNET_WEIGHTS)
    adv_probability = adv_prob[0].item() * 100
    adversarial_text = f"攻击后预测: {adv_class_name}\n置信度: {adv_probability:.2f}%"

    return original_pil, perturbation_pil, adversarial_pil, original_text, adversarial_text


def generate_targeted_attack(progress_callback, target_class_id=504):
    image_path = os.path.join(BACKEND_DIR, 'images', 'panda.jpg')
    img_tensor = read_image(image_path)

    preprocess_no_norm = transforms.Compose([
        transforms.Resize(RESNET50_PREPROCESS.resize_size),
        transforms.CenterCrop(RESNET50_PREPROCESS.crop_size),
        transforms.ConvertImageDtype(torch.float32),
    ])
    normalize = transforms.Normalize(mean=RESNET50_PREPROCESS.mean, std=RESNET50_PREPROCESS.std)

    input_tensor_no_norm = preprocess_no_norm(img_tensor).unsqueeze(0)
    original_normalized_tensor = normalize(input_tensor_no_norm.squeeze(0)).unsqueeze(0)

    with torch.no_grad():
        predictions = RESNET_MODEL(original_normalized_tensor)
        orig_class_name, original_class_id = get_name_and_id_from_prediction(predictions, RESNET50_WEIGHTS)
        orig_prob = torch.nn.functional.softmax(predictions[0], dim=0)[original_class_id].item() * 100
        original_text = f"原始预测: {orig_class_name}\n置信度: {orig_prob:.2f}%"

    actual_class = torch.LongTensor([original_class_id])
    required_class_id = target_class_id
    required_class = torch.LongTensor([required_class_id])
    required_class_name = get_cn_name_by_id(required_class_id, RESNET50_WEIGHTS)

    delta = torch.zeros_like(input_tensor_no_norm, requires_grad=True)
    optimizer = optim.SGD([delta], lr=0.01)
    epsilon = 0.05
    num_iterations = 100

    for i in range(num_iterations + 1):
        perturbed_image_normalized = normalize((input_tensor_no_norm + delta).squeeze(0)).unsqueeze(0)
        predictions = RESNET_MODEL(perturbed_image_normalized)

        loss = torch.nn.CrossEntropyLoss()
        loss_maximize = loss(predictions, actual_class)
        loss_minimize = loss(predictions, required_class)
        total_loss = loss_minimize - loss_maximize

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        delta.data.clamp_(-epsilon, epsilon)

        if progress_callback and i % 5 == 0:
            progress_callback(i / num_iterations)

    if progress_callback:
        progress_callback(1.0)

    final_adv_output = RESNET_MODEL(perturbed_image_normalized)
    adv_prob, adv_catid = torch.topk(torch.nn.functional.softmax(final_adv_output[0], dim=0), 1)

    adv_class_id = adv_catid[0].item()
    adv_class_name = get_cn_name_by_id(adv_class_id, RESNET50_WEIGHTS)
    adv_probability = adv_prob[0].item() * 100
    adversarial_text = f"攻击后预测: {adv_class_name}\n置信度: {adv_probability:.2f}%\n(目标: {required_class_name})"

    original_pil = denormalize_image(original_normalized_tensor)
    adversarial_pil = denormalize_image(perturbed_image_normalized)

    perturbation_vis = delta.squeeze(0).detach().cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (
                perturbation_vis.max() - perturbation_vis.min())
    perturbation_pil = Image.fromarray((np.transpose(perturbation_vis, (1, 2, 0)) * 255).astype(np.uint8))

    return original_pil, perturbation_pil, adversarial_pil, original_text, adversarial_text
