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

# --- Matplotlib 中文显示设置 ---
# 通过加载本地字体文件来确保中文的正确显示
font_path = os.path.join(os.path.dirname(__file__), 'Alibaba-PuHuiTi-Medium.ttf')
if os.path.exists(font_path):
    my_font = FontProperties(fname=font_path)
else:
    print(f"警告：字体文件未找到于 {font_path}。将使用默认字体，中文可能无法显示。")
    my_font = None

def return_class_name(predictions, id_classname):
  """
  根据模型的预测返回最可能的类别名称和索引。

  Args:
    predictions: 模型的预测输出张量。
    id_classname: ImageNet 类别索引到名称的映射字典。

  Returns:
    一个元组，包含类别名称（字符串）和类别索引（整数）。
  """
  max_dim = predictions.argmax(dim=1).item()
  class_name = id_classname[str(max_dim)][1]
  return class_name , max_dim


def return_class_accuracy(predictions, class_id):
  """
  计算模型对特定类别的预测准确率（置信度）。

  Args:
    predictions: 模型的预测输出张量。
    class_id: 要计算准确率的类别索引。

  Returns:
    四舍五入后的准确率（0-100之间的浮点数）。
  """
  prob = F.softmax(predictions, dim=1)
  accuracy = prob[0, class_id] * 100
  return torch.round(accuracy).item()


def visualize(image, adv_image, epsilon, gradients,  target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original):
    """
    可视化原始图像、扰动和对抗样本图像。

    Args:
      image: 原始图像张量。
      adv_image: 对抗样本图像张量。
      epsilon: 扰动大小。
      gradients: 计算出的梯度（扰动）。
      target_class: 原始图像的预测类别。
      adversarial_class: 对抗样本的预测类别。
      target_acc: 原始图像的预测准确率。
      adversarial_acc: 对抗样本的预测准确率。
      acc_of_original: 对抗样本中原始类别的准确率。
    """
    # 反归一化图像以便显示
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    # 处理原始图像
    image = image.squeeze(0) 
    image = image.detach() 
    image = image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    image = np.transpose( image , (1,2,0))   
    image = np.clip(image, 0, 1)

    # 处理对抗样本图像
    adv_image = adv_image.squeeze(0)
    adv_image = adv_image.detach().mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    adv_image = np.transpose( adv_image , (1,2,0))  
    adv_image = np.clip(adv_image, 0, 1)

    # 处理梯度（扰动）
    gradients = gradients.squeeze(0).numpy()
    gradients = np.transpose(gradients, (1,2,0))
    gradients = np.clip(gradients, 0, 1)

    # 创建一个1x3的子图来显示图像
    figure, ax = plt.subplots(1,3, figsize=(18,8))
    
    # 显示原始图像
    ax[0].imshow(image)
    ax[0].set_title('原始图像 (Original Image)', fontproperties=my_font, fontsize=20)
    ax[0].axis("off")

    # 显示扰动
    ax[1].imshow(gradients)
    ax[1].set_title('扰动 (Perturbation) epsilon: {}'.format(epsilon), fontproperties=my_font, fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # 显示对抗样本图像
    ax[2].imshow(adv_image)
    ax[2].set_title('对抗样本 (Adversarial Example)', fontproperties=my_font, fontsize=20)
    ax[2].axis("off")

    # 在图像下方显示预测结果
    ax[0].text(0.5,-0.13, "预测类别: {}\n置信度: {}%".format(target_class, target_acc), size=15, ha="center", transform=ax[0].transAxes, fontproperties=my_font)
    ax[2].text(0.5,-0.13, "预测为 '{}' 的置信度: {}%\n预测为 '{}' 的置信度: {}%".format(adversarial_class, adversarial_acc, target_class, acc_of_original), size=15, ha="center", transform=ax[2].transAxes, fontproperties=my_font)
    
    # 显示图像
    plt.show()
