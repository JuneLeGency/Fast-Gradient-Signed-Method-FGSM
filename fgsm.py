import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
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
# 加载预训练的 AlexNet 模型
alexnet = models.alexnet(pretrained=True)
# 设置为评估模式。这会固定模型的权重，我们只更新输入图像。
alexnet.eval() 

# 加载 ImageNet 类别索引文件
json_path = os.path.join(script_dir, "imagenet_class_index_cn.json")
with open(json_path) as f:
    id_classname = json.load(f)

# --- 图像预处理 ---
# 打开要攻击的图像
image_path = os.path.join(script_dir, "images", "panda.jpg")
image = Image.open(image_path)

# 定义图像的预处理步骤
# 1. 调整图像大小
# 2. 转换为张量
# 3. 使用 ImageNet 的均值和标准差进行归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 预处理图像并增加一个维度（batch size）
input_image = preprocess(image).unsqueeze(0)
# 将输入图像包装成一个需要计算梯度的变量，因为我们需要根据损失来更新图像
input_image = Variable(input_image, requires_grad=True)

# --- 原始图像预测 ---
# 对原始图像进行预测
predictions = alexnet(input_image)
# 获取预测的类别名称和索引
(target_class, target_dim) = return_class_name(predictions, id_classname)
# 获取原始类别的置信度
target_acc = return_class_accuracy(predictions, target_dim)

print(f"原始图像预测: 类别 = {target_class}, 置信度 = {target_acc}%")

# --- FGSM 攻击准备 ---
# 创建目标标签张量
target = Variable(torch.LongTensor([target_dim]), requires_grad=False)
# 定义损失函数
loss = torch.nn.CrossEntropyLoss()
# 计算模型预测与真实标签之间的损失
loss_val = loss(predictions, target)
# 反向传播，计算损失相对于输入图像的梯度
loss_val.backward(retain_graph=True)
# 获取梯度的符号，这是 FGSM 攻击的核心
grads = torch.sign(input_image.grad.data)

# --- 执行攻击并可视化 ---
# 定义一组 epsilon 值（扰动大小）
# epsilon 越大，扰动越明显，但攻击效果（降低模型准确率）也越强
epsilon = [0, 0.007, 0.01, 0.05, 0.1, 0.2] 
print("\n开始进行 FGSM 攻击...")

for i, eps in enumerate(epsilon):
    if eps == 0:
        print("\n--- 未添加扰动 (Epsilon = 0) ---")
        # 可视化原始图像
        # 为了统一处理，我们创建一个“未扰动”的图像副本进行可视化
        adv_image_tensor = input_image.clone().detach()
        predictions_adv = alexnet.forward(Variable(adv_image_tensor))
        adversarial_class, class_index = return_class_name(predictions_adv, id_classname)
        adversarial_acc = return_class_accuracy(predictions_adv, class_index)
        acc_of_original = return_class_accuracy(predictions_adv, target_dim)
        
        print(f"预测类别: {adversarial_class}, 置信度: {adversarial_acc}%")
        print(f"原始类别 '{target_class}' 的置信度降至: {acc_of_original}%")
        
        # 创建一个零梯度用于可视化
        zero_grads = torch.zeros_like(input_image.grad.data)
        visualize(input_image, adv_image_tensor, eps, zero_grads, target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original)
        continue

    print(f"\n--- 添加扰动 (Epsilon = {eps}) ---")
    # FGSM 方法：在原始图像上添加基于梯度符号的扰动
    adversarial_image = input_image.data + eps * grads
    
    # 将扰动后的图像重新输入模型
    predictions_adv = alexnet.forward(Variable(adversarial_image))

    # 获取对抗样本的预测结果
    adversarial_class, class_index = return_class_name(predictions_adv, id_classname)
    adversarial_acc = return_class_accuracy(predictions_adv, class_index)

    # 检查原始类别的置信度变化
    acc_of_original = return_class_accuracy(predictions_adv, target_dim)
    
    print(f"攻击后预测类别: {adversarial_class}, 置信度: {adversarial_acc}%")
    print(f"原始类别 '{target_class}' 的置信度降至: {acc_of_original}%")

    # 可视化结果
    visualize(input_image, adversarial_image, eps, eps * grads, target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original)

print("\nFGSM 攻击演示完成。")
