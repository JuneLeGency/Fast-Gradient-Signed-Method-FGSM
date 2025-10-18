import torch
import torchvision.models as models
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import json
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim

# --- 全局资源加载 ---
# 为了避免在每次调用函数时都重新加载模型和数据，我们在模块级别加载一次
print("正在加载预训练模型和数据...")

# 模型
ALEXNET_MODEL = models.alexnet(pretrained=True).eval()
RESNET_MODEL = models.resnet50(pretrained=True).eval()

# 类别索引 (中文)
script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, "imagenet_class_index_cn.json")
with open(json_path, encoding='utf-8') as f:
    ID_CLASSNAME = {int(k): v for k, v in json.load(f).items()}

print("模型和数据加载完成。")

# --- 图像预处理定义 ---
# ResNet50 的标准预处理
PREPROCESS_RESNET = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# AlexNet 的标准预处理
PREPROCESS_ALEXNET = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 核心功能函数 ---

def predict_image(image_path):
    """
    对单个图像进行预测，返回图像和Top 5预测结果文本。
    Args:
        image_path (str): 图像文件的路径。
    Returns:
        tuple: (PIL.Image, str) 包含原始图像和格式化后的Top 5预测结果字符串。
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return None, f"错误：找不到文件 {image_path}"

    input_tensor = PREPROCESS_RESNET(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = RESNET_MODEL(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    result_text = "--- Top 5 预测结果 ---"
    for i in range(top5_prob.size(0)):
        class_name = ID_CLASSNAME[top5_catid[i].item()][1]
        probability = top5_prob[i].item() * 100
        result_text += f"第{i+1}名: {class_name:<10} 置信度: {probability:.2f}%\n"
        
    return image, result_text

def denormalize_image(tensor):
    """反归一化图像张量以便显示"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().squeeze(0).cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def generate_fgsm_attack(epsilon):
    """
    执行FGSM非定向攻击。
    Args:
        epsilon (float): 扰动强度。
    Returns:
        tuple: 包含三张PIL Image（原图，扰动，对抗样本）和两个结果字符串（攻击前，攻击后）。
    """
    image_path = os.path.join(script_dir, 'images', 'panda.jpg')
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    input_tensor = PREPROCESS_ALEXNET(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch.requires_grad = True

    # 原始预测
    output = ALEXNET_MODEL(input_batch)
    _, top1_catid = torch.topk(output, 1)
    target_class_id = top1_catid[0].item()
    
    loss = F.nll_loss(output, top1_catid[0])
    ALEXNET_MODEL.zero_grad()
    loss.backward()
    
    # 生成扰动
    gradient_sign = input_batch.grad.data.sign()
    perturbation = epsilon * gradient_sign
    adversarial_tensor = input_batch + perturbation
    
    # 对抗样本预测
    adversarial_output = ALEXNET_MODEL(adversarial_tensor)
    adv_prob, adv_catid = torch.topk(torch.nn.functional.softmax(adversarial_output[0], dim=0), 1)

    # 准备返回结果
    original_pil = denormalize_image(input_batch)
    adversarial_pil = denormalize_image(adversarial_tensor)
    
    # 可视化扰动
    perturbation_vis = perturbation.squeeze(0).detach().cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (perturbation_vis.max() - perturbation_vis.min())
    perturbation_pil = Image.fromarray((np.transpose(perturbation_vis, (1, 2, 0)) * 255).astype(np.uint8))

    # 原始结果文本
    orig_class_name = ID_CLASSNAME[target_class_id][1]
    orig_prob = torch.nn.functional.softmax(output[0], dim=0)[target_class_id].item() * 100
    original_text = f"原始预测: {orig_class_name}\n置信度: {orig_prob:.2f}%"

    # 攻击结果文本
    adv_class_name = ID_CLASSNAME[adv_catid[0].item()][1]
    adv_probability = adv_prob[0].item() * 100
    adversarial_text = f"攻击后预测: {adv_class_name}\n置信度: {adv_probability:.2f}%"

    return original_pil, perturbation_pil, adversarial_pil, original_text, adversarial_text

# 定向攻击的逻辑比较复杂，暂时不包含在第一个版本的GUI中，以确保核心功能的稳定交付。
# 如果需要，我们可以在后续迭代中添加。
