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
# 在 macOS 上，可能会遇到 SSL 证书验证失败的问题，导致无法下载预训练模型。
# 以下代码通过创建一个未经验证的 HTTPS 上下文来绕过此问题。
# 这在本地、受信任的环境中是安全的。
ssl._create_default_https_context = ssl._create_unverified_context

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 模型和数据加载 ---
# 加载预训练的 ResNet50 模型
resnet = models.resnet50(pretrained=True)
# 设置为评估模式
resnet.eval()

# 加载 ImageNet 类别索引文件
json_path = os.path.join(script_dir, "imagenet_class_index_cn.json")
with open(json_path) as f:
    id_classname = json.load(f)

# --- 图像预处理 ---
# 打开原始图像
image_path = os.path.join(script_dir, "images", "panda.jpg")
image = Image.open(image_path)

# 定义图像预处理（不包含归一化，因为扰动是在原始像素上优化的）
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 定义归一化，将在把图像送入模型前应用
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm = transforms.Normalize(mean=mean, std=std)

# 预处理图像并增加 batch 维度
input_image = preprocess(image).unsqueeze(0)
input_image = Variable(input_image) # 这个张量不直接计算梯度

# 创建一个归一化后的原始图像副本，用于获取初始预测
original_image_normalized = norm(input_image.squeeze(0)).unsqueeze(0)

# --- 原始图像预测 ---
predictions = resnet(original_image_normalized)
(target_class, target_dim) = return_class_name(predictions, id_classname)
target_acc = return_class_accuracy(predictions, target_dim)
print(f"原始图像预测: 类别 = {target_class}, 置信度 = {target_acc}%")

# --- 目标攻击设置 ---
# 原始图像的真实类别
actual_class = torch.LongTensor([target_dim])
# 我们希望模型误分类的目标类别（这里是 504: 'coffee_mug' 咖啡杯）
required_class_id = 504
required_class = torch.LongTensor([required_class_id])
required_class_name = id_classname[str(required_class.item())][1]
print(f"攻击目标: 让模型将图像识别为 '{required_class_name}'")

# --- 优化扰动 ---
# 初始化一个与输入图像同样大小的零张量，用于存储扰动（delta）
# 我们将优化这个 delta，所以它的 requires_grad=True
delta = torch.zeros_like(input_image, requires_grad=True)
# 定义优化器，使用 SGD 来更新 delta
optimizer = optim.SGD([delta], lr=0.01) # 学习率可以调整

# 扰动大小的上限
epsilon = 0.05 
print(f"\n开始进行目标攻击 (迭代 100 次), Epsilon = {epsilon}..\n")

for i in range(101):
    # 将扰动 delta 添加到原始图像上
    perturbed_image = input_image + delta
    # 在送入模型前进行归一化
    perturbed_image_normalized = norm(perturbed_image.squeeze(0)).unsqueeze(0)
    
    # 模型对扰动后的图像进行预测
    predictions = resnet(perturbed_image_normalized)
    
    # 定义损失函数
    loss = torch.nn.CrossEntropyLoss() 
    # 损失1: 与原始类别的交叉熵。我们希望最大化这个损失。
    loss_maximize = loss(predictions, actual_class) 
    # 损失2: 与目标类别的交叉熵。我们希望最小化这个损失。
    loss_minimize = loss(predictions, required_class) 
    
    # 总损失 = 最小化目标损失 - 最大化原始损失
    total_loss = loss_minimize - loss_maximize

    # 优化步骤
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 限制扰动 delta 的大小在 [-epsilon, epsilon] 范围内
    delta.data.clamp_(-epsilon, epsilon)
    
    # --- 打印和可视化 ---
    if i % 10 == 0:
        # 获取当前预测结果
        adversarial_class, adv_class_index = return_class_name(predictions, id_classname)
        # 获取对目标类别的置信度
        adversarial_acc = return_class_accuracy(predictions, required_class_id)
        # 获取对原始类别的置信度
        acc_of_original = return_class_accuracy(predictions, target_dim)
        
        print(f"--- 迭代次数: {i} ---")
        print(f"当前预测类别: {adversarial_class}")
        print(f"目标类别 '{required_class_name}' 的置信度: {adversarial_acc:.2f}%")
        print(f"原始类别 '{target_class}' 的置信度: {acc_of_original:.2f}%")
        
        # 每 20 次迭代进行一次可视化
        if i % 20 == 0:
            print("生成可视化图像...")
            visualize(original_image_normalized, perturbed_image_normalized, epsilon, delta.detach(), target_class, required_class_name, target_acc, adversarial_acc, acc_of_original)

print("\n目标攻击演示完成。")
