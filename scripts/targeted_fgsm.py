import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from utils import return_class_name, return_class_accuracy, visualize
import os
import ssl

# --- SSL 证书问题修复 ---
ssl._create_default_https_context = ssl._create_unverified_context

# 获取当前脚本所在目录
script_dir = os.path.dirname(__file__)

# --- 模型和数据加载 ---
resnet = models.resnet50(pretrained=True).eval()

json_path = os.path.join(script_dir, "..", "backend", "imagenet_class_index_cn.json")
with open(json_path, encoding='utf-8') as f:
    id_classname = json.load(f)

# --- 图像预处理 ---
image_path = os.path.join(script_dir, "..", "backend", "images", "panda.jpg")
image = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm = transforms.Normalize(mean=mean, std=std)

input_tensor_no_norm = preprocess(image).unsqueeze(0)
original_normalized_tensor = norm(input_tensor_no_norm.squeeze(0)).unsqueeze(0)

# --- 原始图像预测 ---
with torch.no_grad():
    predictions = resnet(original_normalized_tensor)
    (target_class, target_dim) = return_class_name(predictions, id_classname)
    target_acc = return_class_accuracy(predictions, target_dim)
    print(f"原始图像预测: 类别 = {target_class}, 置信度 = {target_acc}%")

# --- 目标攻击设置 ---
actual_class = torch.LongTensor([target_dim])
required_class_id = 504 # 咖啡杯 (coffee_mug)
required_class = torch.LongTensor([required_class_id])
required_class_name = id_classname[str(required_class.item())][1]
print(f"攻击目标: 让模型将图像识别为 '{required_class_name}'")

# --- 优化扰动 ---
delta = torch.zeros_like(input_tensor_no_norm, requires_grad=True)
optimizer = optim.SGD([delta], lr=0.01)
epsilon = 0.05 
num_iterations = 100
print(f"\n开始进行目标攻击 (迭代 {num_iterations} 次), Epsilon = {epsilon}...")

for i in range(num_iterations + 1):
    perturbed_image = input_tensor_no_norm + delta
    perturbed_image_normalized = norm(perturbed_image.squeeze(0)).unsqueeze(0)
    
    predictions = resnet(perturbed_image_normalized)
    
    loss = torch.nn.CrossEntropyLoss()
    loss_maximize = loss(predictions, actual_class)
    loss_minimize = loss(predictions, required_class)
    total_loss = loss_minimize - loss_maximize

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    delta.data.clamp_(-epsilon, epsilon)
    
    if i % 10 == 0:
        adversarial_class, _ = return_class_name(predictions, id_classname)
        adversarial_acc = return_class_accuracy(predictions, required_class_id)
        acc_of_original = return_class_accuracy(predictions, target_dim)
        
        print(f"--- 迭代次数: {i} ---")
        print(f"当前预测类别: {adversarial_class}")
        print(f"目标类别 '{required_class_name}' 的置信度: {adversarial_acc:.2f}%")
        print(f"原始类别 '{target_class}' 的置信度: {acc_of_original:.2f}%")
        
        if i % 20 == 0:
            print("生成可视化图像...")
            visualize(original_normalized_tensor, perturbed_image_normalized, epsilon, delta.detach(), target_class, required_class_name, target_acc, adversarial_acc, acc_of_original)

print("\n目标攻击演示完成。")
