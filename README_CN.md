# 人工智能对抗攻击演示平台 (Web版)

## 1. 项目简介

本项目是一个基于Web的教学演示平台，旨在通过经典的**快速梯度符号攻击（FGSM）**案例，向学生直观地展示人工智能模型的脆弱性，普及“AI内生安全”的基本概念。

为了让所有学生都能零门槛地访问和体验，我们将原始的命令行工具彻底重构为B/S架构（浏览器/服务器）的Web应用。老师只需在本地或服务器上运行本项目，学生即可通过浏览器访问统一的网址进行互动实验，无需在自己的电脑上安装任何复杂的环境。

平台包含三大核心功能模块：

1.  **正常识别**：用户可以上传任意图片，测试AI模型在正常情况下的识别能力。
2.  **非定向攻击**：用户可调整“扰动强度”，对一张固定的熊猫图片施加攻击，观察模型识别率的显著下降，甚至得出荒谬的结论。
3.  **定向攻击**：一个更高级的攻击模式。系统将通过迭代优化的方式，生成一种特制的“扰动”，迫使模型不仅识别错误，而且会以极高的置信度将熊猫图片识别为一个完全不相关的物体——“咖啡杯”。此过程带有实时进度条，方便观察。

## 2. 技术架构

本项目采用标准的前后端分离架构：

-   **后端 (Backend)**：使用 **Python** + **FastAPI** 框架构建，负责处理所有核心计算任务（模型加载、图像处理、攻击算法实现），并通过 REST API 和 WebSocket 与前端通信。
-   **前端 (Frontend)**：使用 **JavaScript** + **React** 框架构建，负责在浏览器中渲染所有用户界面和交互，并与后端进行数据交换。

## 3. 如何运行本项目

您需要在两台终端（或一个终端的两个标签页）中，分别启动后端服务和前端应用。

### 步骤一：启动后端服务

1.  **进入后端目录**：
    打开一个新终端，进入 `backend` 文件夹。
    ```bash
    cd /Users/gencylee/temp/StudyPandaAttackDemo/Fast-Gradient-Signed-Method-FGSM/backend
    ```

2.  **激活虚拟环境**：
    ```bash
    source ../.venv/bin/activate
    ```

3.  **安装/更新依赖**：
    如果您是第一次运行，或依赖有变动，请先安装依赖。
    ```bash
    uv pip install -e . # 注意：这里使用 -e 以便正确安装项目
    ```

4.  **启动服务**：
    使用 `uvicorn` 启动 FastAPI 应用。`--reload` 参数可以在您修改代码后自动重启服务。
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    当您看到类似 `Application startup complete.` 的提示时，表示后端服务已在 `http://localhost:8000` 上成功运行。

### 步骤二：启动前端应用

1.  **进入前端目录**：
    打开**另一个新终端**，进入 `frontend` 文件夹。
    ```bash
    cd /Users/gencylee/temp/StudyPandaAttackDemo/Fast-Gradient-Signed-Method-FGSM/frontend
    ```

2.  **安装依赖**：
    如果您是第一次运行，需要先安装前端项目的所有依赖库。
    ```bash
    npm install
    ```

3.  **启动应用**：
    ```bash
    npm start
    ```
    此命令会自动在您的浏览器中打开一个新的标签页，地址通常是 `http://localhost:3000`。您现在就可以开始使用了。

---

## 4. 文件结构

```
Fast-Gradient-Signed-Method-FGSM/
├── backend/                      # 后端服务代码
│   ├── images/
│   ├── main.py                 # FastAPI 主程序
│   ├── attack_core.py          # 核心算法模块
│   └── ...
├── frontend/                     # 前端 React 应用代码
│   ├── src/
│   │   ├── App.js              # React 主组件
│   │   └── App.css             # 应用样式
│   └── package.json
├── .venv/                        # Python 虚拟环境
├── pyproject.toml              # Python 项目配置文件
└── README_CN.md                # 本说明文档
```
