# 人工智能对抗攻击演示案例：FGSM

## 1. 一键打包脚本 (推荐)

为了最方便地在不同平台分发此应用，我们提供了一个“一键打包”脚本 `build.py`。

### **用法**

无论是在 Windows 还是 macOS 上，只需在项目根目录 `Fast-Gradient-Signed-Method-FGSM/` 下打开终端，然后运行：

```bash
python build.py
```

此脚本会自动完成以下所有操作：
1.  安装所有必要的项目依赖。
2.  安装打包工具 `PyInstaller`。
3.  根据您当前的操作系统，执行打包命令。

打包完成后，您会在 `dist/` 文件夹内找到最终的应用程序：
-   在 **macOS** 上，是 `StudyPandaAttackDemo.app`。
-   在 **Windows** 上，是 `StudyPandaAttackDemo.exe`。

您可以将这个文件直接分发给其他老师使用。

---

## 2. 图形化界面 (GUI) 用法

如果您想直接运行程序而不是打包，请按以下步骤操作。

### **如何启动GUI**

1.  **激活虚拟环境**：
    首先，请确保您在 `Fast-Gradient-Signed-Method-FGSM` 目录下打开终端，并激活虚拟环境：
    ```bash
    source .venv/bin/activate
    ```

2.  **运行主程序**：
    接着，运行 `app_gui.py` 脚本来启动程序：
    ```bash
    python app_gui.py
    ```

### **界面功能说明**

应用启动后，您会看到一个包含三个选项卡的窗口：

1.  **正常识别选项卡**：
    -   点击 **“选择图片”** 按钮，可以从 `images/` 目录或其他位置选择一张图片。
    -   程序会自动对图片进行识别，并在下方文本框中显示置信度最高的前5个结果。

2.  **非定向攻击 (FGSM) 选项卡**：
    -   此功能固定攻击 `images/panda.jpg` 图片。
    -   拖动 **“扰动强度 (Epsilon)”** 滑块可以选择不同的攻击强度。
    -   点击 **“开始攻击”** 按钮，界面会同时展示“原始图像”、“扰动”和“对抗样本”三张图，并在下方显示攻击前后的识别结果对比。

3.  **定向攻击选项卡**：
    -   此功能目前正在开发中。

---

## 3. 案例简介

本项目旨在通过一个经典的对抗攻击案例，向学生普及“人工智能内生安全”的基本概念。我们使用的攻击方法是**快速梯度符号攻击（Fast Gradient Sign Method, FGSM）**，这是由 Ian Goodfellow 等人在2014年提出的、最早也是最著名的对抗攻击算法之一。

**核心思想**：神经网络通过梯度下降法来调整权重、最小化损失函数，从而学习识别图像。FGSM 的思想则巧妙地“反其道而行之”：它不调整模型权重，而是利用模型对于输入图像的梯度信息，对原始输入图像进行微小的、人眼难以察觉的修改，以达到**最大化损失函数**的目的。这样生成的“对抗样本”在人类看来与原图几乎没有区别，但却能让一个训练有素的神经网络模型做出完全错误的分类判断。

---

## 4. 环境准备

如果您不使用 `build.py` 脚本，而是希望手动设置环境，请遵循以下步骤：

1.  **创建与激活环境**：
    首先，请在 `Fast-Gradient-Signed-Method-FGSM` 目录下打开终端，并创建一个新的虚拟环境：
    ```bash
    python3 -m venv .venv
    ```
    然后激活该环境：
    ```bash
    source .venv/bin/activate
    ```

2.  **安装依赖** (推荐方式):
    本项目使用 `pyproject.toml` 管理依赖。请使用 `uv`（或 `pip`）通过以下命令在项目根目录一键安装所有依赖：
    ```bash
    uv pip install .
    ```

3.  **中文字体**：为了在结果图片中正确显示中文，项目中已包含“阿里巴巴普惠体”字体文件 (`Alibaba-PuHuiTi-Medium.ttf`)，并已在代码中配置加载，无需额外安装。

---

## 5. 文件结构说明

```
Fast-Gradient-Signed-Method-FGSM/
├── images/                     # 存放用于测试的原始图片
├── .venv/                      # Python 虚拟环境
├── build.py                    # 一键打包脚本
├── app_gui.py                  # 图形化界面 (GUI) 的主程序
├── attack_core.py              # 封装了核心攻击与识别逻辑的模块
├── predict_normal.py           # (命令行) 用于演示正常的图像识别
├── fgsm.py                     # (命令行) 用于演示非定向的FGSM攻击
├── targeted_fgsm.py            # (命令行) 用于演示定向的FGSM攻击
├── utils.py                    # (命令行) 辅助工具脚本，包含绘图等函数
├── imagenet_class_index_cn.json # ImageNet 1000个类别的中文翻译
├── Alibaba-PuHuiTi-Medium.ttf  # 用于显示中文的字体文件
├── pyproject.toml              # 项目配置文件 (包含依赖)
└── README_CN.md                # 本说明文档
```

---

## 6. 高级用法：命令行

如果您或您的学生熟悉命令行，也可以直接运行原始的脚本文件来分步进行演示。

### 步骤一：正常图像识别 (证明模型有效)

`predict_normal.py` 脚本可以接受一个图片路径作为参数。如果不提供参数，它将默认识别 `images/panda.jpg`。

1.  **识别默认的熊猫图片**：
    ```bash
    python predict_normal.py
    ```

2.  **识别其他图片**：
    ```bash
    python predict_normal.py images/dog.jpeg
    ```

### 步骤二：非定向攻击 (让模型犯错)

`fgsm.py` 脚本将对熊猫图片进行非定向攻击。

```bash
python fgsm.py
```

### 步骤三：定向攻击 (指鹿为马)

`targeted_fgsm.py` 脚本的目标是让模型把“大熊猫”识别为“咖啡杯”。

```bash
python targeted_fgsm.py
```