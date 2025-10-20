import os
import ssl

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.io import read_image

from utils import return_class_accuracy, visualize, get_name_and_id_from_prediction

# --- SSL 证书问题修复 ---
ssl._create_default_https_context = ssl._create_unverified_context

# 获取当前脚本所在目录
script_dir = os.path.dirname(__file__)

# --- 模型和数据加载 ---
print("正在加载预训练的 AlexNet 模型 (V1 权重)...")
weights = models.AlexNet_Weights.IMAGENET1K_V1
alexnet = models.alexnet(weights=weights)
alexnet.eval() 
preprocess_official = weights.transforms()

# --- 图像预处理 ---
image_path = os.path.join(script_dir, "..", "backend", "images", "panda.jpg")
img_tensor = read_image(image_path)

# 此处保持自定义的Resize以维持原始FGSM脚本的行为
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=preprocess_official.mean, std=preprocess_official.std),
])

input_image = preprocess(img_tensor).unsqueeze(0)
input_image = Variable(input_image, requires_grad=True)

# --- 原始图像预测 ---
predictions = alexnet(input_image)
target_class, target_dim = get_name_and_id_from_prediction(predictions, weights)
target_acc = return_class_accuracy(predictions, target_dim)

print(f"原始图像预测: 类别 = {target_class}, 置信度 = {target_acc}%")

# --- FGSM 攻击准备 ---
target = Variable(torch.LongTensor([target_dim]), requires_grad=False)
loss = torch.nn.CrossEntropyLoss()
loss_val = loss(predictions, target)
loss_val.backward(retain_graph=True)
grads = torch.sign(input_image.grad.data)

# --- 执行攻击并可视化 ---
epsilon = [0, 0.007, 0.01, 0.05, 0.1, 0.2] 
print("\n开始进行 FGSM 攻击...")

for i, eps in enumerate(epsilon):
    if eps == 0:
        print(f"\n--- 未添加扰动 (Epsilon = {eps}) ---")
        adv_image_tensor = input_image.clone().detach()
        predictions_adv = alexnet.forward(Variable(adv_image_tensor))
        adversarial_class, class_index = get_name_and_id_from_prediction(predictions_adv, weights)
        adversarial_acc = return_class_accuracy(predictions_adv, class_index)
        acc_of_original = return_class_accuracy(predictions_adv, target_dim)
        
        print(f"预测类别: {adversarial_class}, 置信度: {adversarial_acc}%")
        print(f"原始类别 '{target_class}' 的置信度降至: {acc_of_original}%")
        
        zero_grads = torch.zeros_like(input_image.grad.data)
        visualize(input_image, adv_image_tensor, eps, zero_grads, target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original)
        continue

    print(f"\n--- 添加扰动 (Epsilon = {eps}) ---")
    adversarial_image = input_image.data + eps * grads
    predictions_adv = alexnet.forward(Variable(adversarial_image))

    adversarial_class, class_index = get_name_and_id_from_prediction(predictions_adv, weights)
    adversarial_acc = return_class_accuracy(predictions_adv, class_index)
    acc_of_original = return_class_accuracy(predictions_adv, target_dim)
    
    print(f"攻击后预测类别: {adversarial_class}, 置信度: {adversarial_acc}%")
    print(f"原始类别 '{target_class}' 的置信度降至: {acc_of_original}%")

    visualize(input_image, adversarial_image, eps, eps * grads, target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original)

print("\nFGSM 攻击演示完成。\n")