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
# 通过计算脚本的绝对路径来定位项目根目录下的字体文件
font_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'Alibaba-PuHuiTi-Medium.ttf')
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

# --- 类别名称查找 (带缓存) ---
_EN_TO_CN_MAP = None
_MAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'en_to_cn_mapping.json')

def _load_mapping():
    """将英中映射文件加载到全局变量中，只加载一次。"""
    global _EN_TO_CN_MAP
    if _EN_TO_CN_MAP is None:
        print("首次调用，正在加载英中名称映射文件...")
        try:
            with open(_MAP_PATH, "r", encoding="utf-8") as f:
                _EN_TO_CN_MAP = json.load(f)
        except Exception as e:
            print(f"错误：无法加载映射文件 {_MAP_PATH}: {e}")
            _EN_TO_CN_MAP = {} # 避免后续调用时重复尝试加载

def get_name_and_id_from_prediction(predictions, weights):
    """
    从模型输出中获取最高置信度的类别ID和中文名。
    """
    _load_mapping() # 确保映射已加载
    
    max_dim = predictions.argmax(dim=1).item()
    
    english_name = weights.meta["categories"][max_dim]
    chinese_name = _EN_TO_CN_MAP.get(english_name, english_name)
    
    return chinese_name, max_dim

def get_cn_name_by_id(class_id, weights):
    """
    根据给定的类别ID获取中文名。
    """
    _load_mapping() # 确保映射已加载
    english_name = weights.meta["categories"][class_id]
    return _EN_TO_CN_MAP.get(english_name, english_name)

def get_all_classes_with_cn_names(weights):
    """
    返回一个包含所有类别及其翻译名称的列表，用于前端选择框。
    """
    _load_mapping() # 确保映射已加载
    english_categories = weights.meta["categories"]
    class_list = [
        {
            "value": i,
            "label": f"{_EN_TO_CN_MAP.get(english_name, english_name)} ({english_name})"
        }
        for i, english_name in enumerate(english_categories)
    ]
    return class_list




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
