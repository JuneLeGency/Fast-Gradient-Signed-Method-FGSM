import os
import ssl

import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image

from utils import return_class_accuracy, visualize, get_name_and_id_from_prediction, get_cn_name_by_id

# --- SSL 证书问题修复 ---
ssl._create_default_https_context = ssl._create_unverified_context

# 获取当前脚本所在目录
script_dir = os.path.dirname(__file__)

# --- 模型和数据加载 ---
print("正在加载预训练的 ResNet50 模型 (V1 权重)...")
weights = models.ResNet50_Weights.IMAGENET1K_V1
resnet = models.resnet50(weights=weights).eval()
preprocess_official = weights.transforms()

# --- 图像预处理 ---
image_path = os.path.join(script_dir, "..", "backend", "images", "panda.jpg")
img_tensor = read_image(image_path)

# 拆分官方预处理流程，以便注入攻击
preprocess_no_norm = transforms.Compose([
    transforms.Resize(preprocess_official.resize_size),
    transforms.CenterCrop(preprocess_official.crop_size),
    transforms.ConvertImageDtype(torch.float32),
])
norm = transforms.Normalize(mean=preprocess_official.mean, std=preprocess_official.std)

input_tensor_no_norm = preprocess_no_norm(img_tensor).unsqueeze(0)
original_normalized_tensor = norm(input_tensor_no_norm.squeeze(0)).unsqueeze(0)

# --- 原始图像预测 ---
with torch.no_grad():
    predictions = resnet(original_normalized_tensor)
    target_class, target_dim = get_name_and_id_from_prediction(predictions, weights)
    target_acc = return_class_accuracy(predictions, target_dim)
    print(f"原始图像预测: 类别 = {target_class}, 置信度 = {target_acc}%")

# --- 目标攻击设置 ---
actual_class = torch.LongTensor([target_dim])
required_class_id = 504 # 咖啡杯 (coffee_mug)
required_class = torch.LongTensor([required_class_id])
required_class_name = get_cn_name_by_id(required_class_id, weights)
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
        adversarial_class, _ = get_name_and_id_from_prediction(predictions, weights)
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


# --- 保存并验证最终的对抗样本图像 ---
print("\n--- 保存并验证最终的对抗样本图像 ---")

final_perturbed_image_tensor = (input_tensor_no_norm + delta).squeeze(0)
final_perturbed_image_tensor.data.clamp_(0, 1)

to_pil = transforms.ToPILImage()
final_pil_image = to_pil(final_perturbed_image_tensor.cpu())

results_dir = os.path.join(script_dir, "..", "results_images")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
save_path = os.path.join(results_dir, f"{required_class_name}.png")
final_pil_image.save(save_path)
print(f"最终的对抗样本图像已保存到: {save_path}")

# --- 加载并验证保存的图像 ---
print("\n开始验证保存的对抗样本图像...")
verify_image_tensor = read_image(save_path)

verify_preprocess = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    norm, # 复用上面定义的norm
])
verify_tensor = verify_preprocess(verify_image_tensor).unsqueeze(0)

with torch.no_grad():
    verify_predictions = resnet(verify_tensor)
    verify_class, verify_dim = get_name_and_id_from_prediction(verify_predictions, weights)
    verify_acc = return_class_accuracy(verify_predictions, verify_dim)
    print(f"修正后验证结果: 预测类别 = '{verify_class}', 置信度 = {verify_acc:.2f}%")