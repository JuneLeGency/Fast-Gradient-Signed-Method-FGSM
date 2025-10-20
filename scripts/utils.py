import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
from matplotlib.font_manager import FontProperties
import sys

# --- Matplotlib 中文显示设置 ---
# 通过加载本地字体文件来确保中文的正确显示
font_path = os.path.join(os.path.dirname(__file__), 'Alibaba-PuHuiTi-Medium.ttf')
if os.path.exists(font_path):
    my_font = FontProperties(fname=font_path)
else:
    # 如果找不到字体，就尝试使用一个常见的系统备用字体
    # 在macOS上是'PingFang SC'，在Windows上是'SimHei'
    if sys.platform == 'darwin':
        fallback_font = 'PingFang SC'
    elif sys.platform == 'win32':
        fallback_font = 'SimHei'
    else:
        fallback_font = 'sans-serif' # Linux上的通用备用
    print(f"警告：字体文件未找到于 {font_path}。将尝试使用备用字体 {fallback_font}。")
    try:
        my_font = FontProperties(fname=fallback_font)
        # 测试一下备用字体是否真的能找到
        plt.figure(figsize=(0.1, 0.1))
        plt.text(0, 0, "测试", fontproperties=my_font)
        plt.close()
    except RuntimeError:
        print(f"警告：备用字体 {fallback_font} 也无法加载。中文可能无法显示。")
        my_font = None

def return_class_name(predictions, id_classname):
  """
  根据模型的预测返回最可能的类别名称和索引。
  """
  max_dim = predictions.argmax(dim=1).item()
  class_name = id_classname[str(max_dim)][1]
  return class_name , max_dim


def return_class_accuracy(predictions, class_id):
  """
  计算模型对特定类别的预测准确率（置信度）。
  """
  prob = F.softmax(predictions, dim=1)
  accuracy = prob[0, class_id] * 100
  return torch.round(accuracy).item()


def visualize(image, adv_image, epsilon, gradients,  target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original):
    """
    可视化原始图像、扰动和对抗样本图像。
    """
    # 反归一化图像以便显示
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    # 处理原始图像
    image = image.squeeze(0).detach().cpu()
    image = image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    image = np.transpose( image , (1,2,0))   
    image = np.clip(image, 0, 1)

    # 处理对抗样本图像
    adv_image = adv_image.squeeze(0).detach().cpu()
    adv_image = adv_image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    adv_image = np.transpose( adv_image , (1,2,0))  
    adv_image = np.clip(adv_image, 0, 1)

    # 处理梯度（扰动）
    gradients = gradients.squeeze(0).detach().cpu().numpy()
    gradients = np.transpose(gradients, (1,2,0))
    # 将扰动从 [-eps, eps] 范围归一化到 [0, 1] 范围以便于可视化
    if gradients.max() > gradients.min():
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
    else:
        gradients = np.clip(gradients, 0, 1)

    # 创建一个1x3的子图来显示图像
    figure, ax = plt.subplots(1,3, figsize=(18,8))
    
    # 显示原始图像
    ax[0].imshow(image)
    ax[0].set_title('原始图像 (Original Image)', fontproperties=my_font, fontsize=20)
    ax[0].axis("off")

    # 显示扰动
    ax[1].imshow(gradients)
    ax[1].set_title(f'扰动 (Perturbation) epsilon: {epsilon}', fontproperties=my_font, fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # 显示对抗样本图像
    ax[2].imshow(adv_image)
    ax[2].set_title('对抗样本 (Adversarial Example)', fontproperties=my_font, fontsize=20)
    ax[2].axis("off")

    # 在图像下方显示预测结果
    ax[0].text(0.5,-0.13, f"预测类别: {target_class}\n置信度: {target_acc}%", size=15, ha="center", transform=ax[0].transAxes, fontproperties=my_font)
    ax[2].text(0.5,-0.13, f"预测为 '{adversarial_class}' 的置信度: {adversarial_acc}%\n预测为 '{target_class}' 的置信度: {acc_of_original}%", size=15, ha="center", transform=ax[2].transAxes, fontproperties=my_font)
    
    # 显示图像
    plt.show()
