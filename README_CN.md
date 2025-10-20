# 人工智能对抗攻击演示平台

## 1. 项目简介

本项目旨在通过一个经典的对抗攻击案例，向学生普及“人工智能内生安全”的基本概念。为了适应不同的教学场景，本项目提供了两个版本：

1.  **Web 应用版 (推荐)**：一个前后端分离的Web应用。老师只需在本地或服务器上运行，所有学生即可通过浏览器访问统一的网址进行互动实验，无需在自己的设备上安装任何环境，是多人教学和分发的最佳选择。

2.  **原生GUI版**：一个使用 CustomTkinter 构建的独立桌面应用。它功能完善，适合在单台已配置好环境的电脑上进行本地演示。

两个版本均包含三大核心功能：正常图像识别、非定向攻击 (FGSM) 和带实时进度的定向攻击。

---

## 2. 如何使用

**基本要求**: 您的电脑需要预先安装 [Python 3.8+](https://www.python.org/) 和 [Node.js 16+](https://nodejs.org/)。

### 首次环境设置

如果您是第一次使用本项目，请先在项目根目录下打开终端，运行 `setup` 命令来创建虚拟环境并安装所有依赖：

```bash
python3 build_and_package.py setup
```

--- 

### 方案A：运行 Web 应用 (推荐)

环境设置好后，执行以下步骤：

1.  **构建前端** (仅在修改前端代码后需要)：
    ```bash
    python3 build_and_package.py build_frontend
    ```

2.  **启动服务**：
    ```bash
    python3 build_and_package.py run
    ```

3.  **访问应用**：服务启动后，在浏览器中打开 [http://localhost:8000](http://localhost:8000) 即可。

### 方案B：运行原生GUI应用

环境设置好后，直接运行 `app_gui.py` 即可：

```bash
# 确保虚拟环境已激活
source .venv/bin/activate 

# 运行GUI程序
python3 app_gui.py
```

#### 打包原生GUI应用 (可选)

如果您想将原生GUI版打包成一个独立的可执行文件（macOS下为 `.app`，Windows下为 `.exe`），可以运行 `build_gui.py` 脚本：

```bash
python3 build_gui.py
```
打包好的文件会出现在 `dist/` 目录下。

---

## 3. 文件结构

```
Fast-Gradient-Signed-Method-FGSM/
├── backend/              # Web应用 - 后端代码与共享资源 (images, fonts, data)
├── frontend/             # Web应用 - 前端 React 应用代码
│
├── app_gui.py            # 原生GUI - 主程序
├── build_gui.py          # 原生GUI - 打包脚本
│
├── build_and_package.py  # Web应用 - 自动化构建与运行脚本
├── scripts/              # 存放旧的、仅用于命令行的脚本 (已归档)
│
├── pyproject.toml        # Python 项目配置文件 (包含所有依赖)
└── README_CN.md          # 本说明文档
```