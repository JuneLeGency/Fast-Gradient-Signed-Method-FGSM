import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from matplotlib.font_manager import FontProperties
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

from utils import get_cn_name_by_id  # Import the new utility function


def predict_image(image_path):
    """
    加载预训练的 ResNet50 模型，对单张图片进行预测，并显示Top 5的结果。
    """
    print("\n--- Custom predict_image Function (Refactored) ---")
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
    
    # --- 图像预处理 ---
    try:
        img_tensor = read_image(image_path)
    except (FileNotFoundError, RuntimeError):
        print(f"错误：找不到或无法读取图像文件 {image_path}")
        return
    
    display_image = transforms.ToPILImage()(img_tensor)

    # Get the height and width from the tensor shape (C, H, W)
    img_height, img_width = img_tensor.shape[1], img_tensor.shape[2]

    # Get the official preprocessing pipeline
    preprocess_official = weights.transforms()

    # Determine the crop size from the official transforms
    crop_size_val = preprocess_official.crop_size
    if not isinstance(crop_size_val, int):
        crop_size_val = crop_size_val[0]

    # Conditionally apply preprocessing
    if (img_height, img_width) == (crop_size_val, crop_size_val):
        print(f"图像尺寸为 {(img_height, img_width)}，将跳过缩放和裁剪步骤。")
        # For already-cropped images, just convert dtype and normalize
        preprocess = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=preprocess_official.mean, std=preprocess_official.std),
        ])
    else:
        print(f"图像尺寸为 {(img_height, img_width)}，将使用官方标准预处理流程。")
        # For other images, use the full official pipeline
        preprocess = preprocess_official

    input_batch = preprocess(img_tensor).unsqueeze(0)

    # --- 执行预测 ---
    print("正在对图像进行预测...")
    with torch.no_grad():
        output = resnet(input_batch)

    # --- 展示结果 ---
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    print("\n--- Top 5 预测结果 ---")
    for i in range(top5_prob.size(0)):
        class_id = top5_catid[i].item()
        # 使用新的工具函数获取中文名
        chinese_name = get_cn_name_by_id(class_id, weights)
        probability = top5_prob[i].item() * 100
        print(f"第{i + 1}名: {chinese_name:<15} 置信度: {probability:.2f}%")

    # 可视化图像和最高置信度的预测结果
    top1_class_id = top5_catid[0].item()
    top1_chinese_name = get_cn_name_by_id(top1_class_id, weights)
    top1_prob = top5_prob[0].item() * 100

    plt.imshow(display_image)
    title_text = f"原始图像预测结果\n类别: {top1_chinese_name}, 置信度: {top1_prob:.2f}%"
    plt.title(title_text, fontproperties=my_font, fontsize=16)
    plt.axis('off')
    plt.show()


def official_sample(path: str):
    print("\n--- Official PyTorch Sample (Refactored) ---")
    img = read_image(path)
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    
    english_name = weights.meta["categories"][class_id]
    # 使用新的工具函数获取中文名
    chinese_name = get_cn_name_by_id(class_id, weights)
    
    print(f"Official Sample - {english_name} ({chinese_name}): {100 * score:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(os.path.dirname(__file__), '../backend/images', 'panda.jpg')
        print(f"未指定图像文件，将使用默认图像: {image_file}")

    predict_image(image_file)
    official_sample(image_file)