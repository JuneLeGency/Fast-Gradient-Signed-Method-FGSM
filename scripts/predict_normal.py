import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.font_manager import FontProperties

def predict_image(image_path):
    """
    加载预训练的 ResNet50 模型，对单张图片进行预测，并显示Top 5的结果。
    """
    font_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'Alibaba-PuHuiTi-Medium.ttf')
    if os.path.exists(font_path):
        my_font = FontProperties(fname=font_path)
    else:
        print(f"警告：字体文件未找到于 {font_path}。将使用默认字体，中文可能无法显示。")
        my_font = None

    # --- 模型和数据加载 ---
    print("正在加载预训练的 ResNet50 模型...")
    # 加载预训练的 ResNet50 模型
    resnet = models.resnet50(pretrained=True)
    # 设置为评估模式
    resnet.eval()

    # 加载 ImageNet 类别索引文件（中文版）
    json_path = os.path.join(os.path.dirname(__file__), "..", "backend", "imagenet_class_index_cn.json")
    with open(json_path, encoding='utf-8') as f:
        id_classname = {int(k): v for k, v in json.load(f).items()}

    # --- 图像预处理 ---
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误：找不到图像文件 {image_path}")
        return

    # 定义图像的预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 预处理图像并增加一个维度（batch size）
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # --- 执行预测 ---
    print("正在对图像进行预测...")
    with torch.no_grad():
        output = resnet(input_batch)

    # --- 展示结果 ---
    # 将输出转换为概率
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 打印 Top 5 预测结果
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("\n--- Top 5 预测结果 ---")
    for i in range(top5_prob.size(0)):
        class_name = id_classname[top5_catid[i].item()][1]
        probability = top5_prob[i].item() * 100
        print(f"第{i+1}名: {class_name:<15} 置信度: {probability:.2f}%")
    
    # 可视化图像和最高置信度的预测结果
    top1_class_name = id_classname[top5_catid[0].item()][1]
    top1_prob = top5_prob[0].item() * 100

    plt.imshow(image)
    title_text = f"原始图像预测结果\n类别: {top1_class_name}, 置信度: {top1_prob:.2f}%"
    plt.title(title_text, fontproperties=my_font, fontsize=16)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        # 如果没有提供参数，默认使用 images/panda.jpg
        image_file = os.path.join(os.path.dirname(__file__), 'images', 'panda.jpg')
        print(f"未指定图像文件，将使用默认图像: {image_file}")
    
    predict_image(image_file)
