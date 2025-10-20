import json
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- 全局资源加载 ---
print("正在加载预训练模型和数据...")

ALEXNET_MODEL = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
RESNET_MODEL = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
# --- 图像预处理定义 ---
script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, "imagenet_class_index_cn.json")
with open(json_path, encoding='utf-8') as f:
    ID_CLASSNAME = {int(k): v for k, v in json.load(f).items()}

# --- 可重用的图像变换 ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 标准归一化变换
TRANSFORM_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# 标准的 "Resize -> CenterCrop" 变换
TRANSFORM_RESIZE_CROP = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# 转换为张量
TRANSFORM_TO_TENSOR = transforms.ToTensor()

# 完整的标准预处理流程
PREPROCESS_STANDARD = transforms.Compose([
    TRANSFORM_RESIZE_CROP,
    TRANSFORM_TO_TENSOR,
    TRANSFORM_NORMALIZE,
])

# 仅包含 "ToTensor" 和 "Normalize" 的预处理流程 (用于224x224图像)
PREPROCESS_TENSOR_NORM = transforms.Compose([
    TRANSFORM_TO_TENSOR,
    TRANSFORM_NORMALIZE,
])

print("模型和数据加载完成。")

# --- 预计算和缓存 ---
# 为了与原始脚本行为一致，我们预先计算并缓存FGSM的梯度
FGSM_GRADIENT = None
FGSM_INPUT_TENSOR = None

def precompute_fgsm_gradient():
    """仅在首次需要时计算并缓存FGSM的梯度。"""
    global FGSM_GRADIENT, FGSM_INPUT_TENSOR
    if FGSM_GRADIENT is not None:
        return

    print("首次执行FGSM，正在预计算梯度...")
    image_path = os.path.join(script_dir, 'images', 'panda.jpg')
    image = Image.open(image_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        TRANSFORM_TO_TENSOR,
        TRANSFORM_NORMALIZE,
    ])

    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor.requires_grad = True
    FGSM_INPUT_TENSOR = input_tensor

    predictions = ALEXNET_MODEL(input_tensor)
    _, top1_catid = torch.topk(predictions, 1)
    target_class_id = top1_catid[0].item()
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
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return None, f"错误：找不到文件 {image_path}"

    # 根据图像尺寸决定预处理步骤
    if image.size == (224, 224):
        preprocess = PREPROCESS_TENSOR_NORM
    else:
        preprocess = PREPROCESS_STANDARD
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = RESNET_MODEL(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    result_text = "--- Top 5 预测结果 ---" + "\n"
    for i in range(top5_prob.size(0)):
        class_name = ID_CLASSNAME[top5_catid[i].item()][1]
        probability = top5_prob[i].item() * 100
        result_text += f"第{i+1}名: {class_name:<10} 置信度: {probability:.2f}%\n"
        
    return image, result_text

def denormalize_image(tensor):
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    tensor = tensor.clone().detach().squeeze(0).cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def generate_fgsm_attack(epsilon):
    precompute_fgsm_gradient() # 确保梯度已计算

    # 使用缓存的梯度和输入张量
    perturbation = epsilon * FGSM_GRADIENT
    adversarial_tensor = FGSM_INPUT_TENSOR.data + perturbation
    
    adversarial_output = ALEXNET_MODEL(adversarial_tensor)
    adv_prob, adv_catid = torch.topk(torch.nn.functional.softmax(adversarial_output[0], dim=0), 1)

    original_pil = denormalize_image(FGSM_INPUT_TENSOR)
    adversarial_pil = denormalize_image(adversarial_tensor)
    
    perturbation_vis = perturbation.squeeze(0).detach().cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (perturbation_vis.max() - perturbation_vis.min())
    perturbation_pil = Image.fromarray((np.transpose(perturbation_vis, (1, 2, 0)) * 255).astype(np.uint8))

    # 原始结果文本
    with torch.no_grad():
        orig_output = ALEXNET_MODEL(FGSM_INPUT_TENSOR)
        _, orig_catid = torch.topk(orig_output, 1)
        orig_class_id = orig_catid[0].item()
        orig_class_name = ID_CLASSNAME[orig_class_id][1]
        orig_prob = torch.nn.functional.softmax(orig_output[0], dim=0)[orig_class_id].item() * 100
        original_text = f"原始预测: {orig_class_name}\n置信度: {orig_prob:.2f}%"

    # 攻击结果文本
    adv_class_name = ID_CLASSNAME[adv_catid[0].item()][1]
    adv_probability = adv_prob[0].item() * 100
    adversarial_text = f"攻击后预测: {adv_class_name}\n置信度: {adv_probability:.2f}%"

    return original_pil, perturbation_pil, adversarial_pil, original_text, adversarial_text

def generate_targeted_attack(progress_callback, target_class_id=504):
    image_path = os.path.join(script_dir, 'images', 'panda.jpg')
    image = Image.open(image_path).convert('RGB')

    preprocess_no_norm = transforms.Compose([
        TRANSFORM_RESIZE_CROP,
        TRANSFORM_TO_TENSOR,
    ])
    normalize = TRANSFORM_NORMALIZE

    input_tensor_no_norm = preprocess_no_norm(image).unsqueeze(0)
    original_normalized_tensor = normalize(input_tensor_no_norm.squeeze(0)).unsqueeze(0)

    with torch.no_grad():
        predictions = RESNET_MODEL(original_normalized_tensor)
        _, top1_catid = torch.topk(predictions, 1)
        original_class_id = top1_catid[0].item()
        orig_prob = torch.nn.functional.softmax(predictions[0], dim=0)[original_class_id].item() * 100
        orig_class_name = ID_CLASSNAME[original_class_id][1]
        original_text = f"原始预测: {orig_class_name}\n置信度: {orig_prob:.2f}%"

    actual_class = torch.LongTensor([original_class_id])
    required_class_id = target_class_id
    required_class = torch.LongTensor([required_class_id])
    required_class_name = ID_CLASSNAME[required_class_id][1]

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
    
    adv_class_name = ID_CLASSNAME[adv_catid[0].item()][1]
    adv_probability = adv_prob[0].item() * 100
    adversarial_text = f"攻击后预测: {adv_class_name}\n置信度: {adv_probability:.2f}%\n(目标: {required_class_name})"

    original_pil = denormalize_image(original_normalized_tensor)
    adversarial_pil = denormalize_image(perturbed_image_normalized)
    
    perturbation_vis = delta.squeeze(0).detach().cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (perturbation_vis.max() - perturbation_vis.min())
    perturbation_pil = Image.fromarray((np.transpose(perturbation_vis, (1, 2, 0)) * 255).astype(np.uint8))

    return original_pil, perturbation_pil, adversarial_pil, original_text, adversarial_text