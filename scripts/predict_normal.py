import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.font_manager import FontProperties
from torchvision.io import read_image


def predict_image(image_path):
    """
    加载预训练的 ResNet50 模型，对单张图片进行预测，并显示Top 5的结果。
    此版本使用 torchvision.io.read_image 以获得与官方示例一致的结果。
    """
    print("\n--- Custom predict_image Function ---")
    font_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'Alibaba-PuHuiTi-Medium.ttf')
    if os.path.exists(font_path):
        my_font = FontProperties(fname=font_path)
    else:
        print(f"警告：字体文件未找到于 {font_path}。将使用默认字体，中文可能无法显示。")
        my_font = None

    # --- 模型和数据加载 ---
    print("正在加载预训练的 ResNet50 模型 (V1 权重)...")
    weights = ResNet50_Weights.IMAGENET1K_V1
    resnet = resnet50(weights=weights)
    resnet.eval()

    # 获取官方预处理流程
    preprocess = weights.transforms()

    # 加载 ImageNet 类别索引文件（中文版）
    json_path = os.path.join(os.path.dirname(__file__), "..", "backend", "imagenet_class_index_cn.json")
    with open(json_path, encoding='utf-8') as f:
        id_classname = {int(k): v for k, v in json.load(f).items()}

    # --- 图像预处理 ---
    try:
        # 使用 torchvision.io.read_image 以获得与官方示例一致的结果
        # read_image 返回一个 (C, H, W) 的 uint8 张量
        img_tensor = read_image(image_path)
    except (FileNotFoundError, RuntimeError):
        print(f"错误：找不到或无法读取图像文件 {image_path}")
        return
    
    # 为了显示，需要将tensor转回PIL Image
    display_image = transforms.ToPILImage()(img_tensor)

    # 预处理图像张量并增加一个维度（batch size）
    # 官方的preprocess可以直接处理PIL Image或Tensor
    input_batch = preprocess(img_tensor).unsqueeze(0)

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
        print(f"第{i + 1}名: {class_name:<15} 置信度: {probability:.2f}%")

    # 可视化图像和最高置信度的预测结果
    top1_class_name = id_classname[top5_catid[0].item()][1]
    top1_prob = top5_prob[0].item() * 100

    plt.imshow(display_image)
    title_text = f"原始图像预测结果\n类别: {top1_class_name}, 置信度: {top1_prob:.2f}%"
    plt.title(title_text, fontproperties=my_font, fontsize=16)
    plt.axis('off')
    plt.show()


def official_sample(path: str):
    print("\n--- Official PyTorch Sample (Corrected) ---")
    # 使用 read_image 读取图片，decode_image 用于解码内存中的字节流
    img = read_image(path)

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    # 使用官方的类别名称
    category_name = weights.meta["categories"][class_id]
    print(f"Official Sample - {category_name}: {100 * score:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(os.path.dirname(__file__), '../backend/images', 'horse.jpeg')
        print(f"未指定图像文件，将使用默认图像: {image_file}")

    predict_image(image_file)
    official_sample(image_file)