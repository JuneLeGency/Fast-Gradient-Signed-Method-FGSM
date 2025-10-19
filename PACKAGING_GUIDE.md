# Windows 应用打包指南 (.exe)

本指南将引导您如何在 Windows 环境下，将此项目打包成一个独立的 `.exe` 可执行文件。最终生成的文件可以在任何现代 Windows 电脑上直接运行，无需预先安装 Python 或任何库。

---

### 步骤一：准备环境

1.  **安装 Python**：
    - 如果您的电脑尚未安装 Python，请从 [Python 官网](https://www.python.org/downloads/windows/) 下载并安装最新稳定版本的 Python。
    - **重要**：在安装时，请务必勾选 “Add Python to PATH” 选项。

2.  **获取项目文件**：
    - 从 GitHub 仓库下载本项目的源代码，并解压到一个固定的位置（例如 `C:\Users\YourName\Desktop\StudyPandaAttackDemo`）。

3.  **打开命令提示符 (CMD) 或 PowerShell**：
    - 按下 `Win` 键，输入 `cmd` 或 `powershell`，然后打开它。
    - 使用 `cd` 命令进入您刚刚解压的项目中的 `Fast-Gradient-Signed-Method-FGSM` 文件夹。例如：
      ```powershell
      cd C:\Users\YourName\Desktop\StudyPandaAttackDemo\Fast-Gradient-Signed-Method-FGSM
      ```

---

### 步骤二：创建虚拟环境并安装依赖

在项目文件夹中，执行以下命令：

1.  **创建虚拟环境**：
    ```powershell
    python -m venv .venv
    ```

2.  **激活虚拟环境**：
    ```powershell
    .venv\Scripts\activate
    ```
    激活成功后，您会看到命令行提示符前面出现 `(.venv)` 的字样。

3.  **安装依赖库**：
    我们将使用 `pip` 来安装所有项目依赖。`pyproject.toml` 文件已经定义好了一切。
    ```powershell
    pip install .
    ```
    这个命令会自动读取 `pyproject.toml` 文件并安装所有必需的库。

---

### 步骤三：安装打包工具

我们将使用业界标准的 `PyInstaller` 工具进行打包。

```powershell
pip install pyinstaller
```

---

### 步骤四：执行打包命令

这是最关键的一步。此命令会收集您的 Python 脚本以及所有需要用到的资源文件（如字体、图片、数据文件），并将它们全部打包到一个 `.exe` 文件中。

请在依然处于激活的虚拟环境中的命令行里，执行以下命令：

```powershell
pyinstaller --onefile --windowed --name StudyPandaAttackDemo --add-data "Alibaba-PuHuiTi-Medium.ttf;." --add-data "imagenet_class_index_cn.json;." --add-data "images;images" app_gui.py
```

**命令解释**：
- `--onefile`: 创建一个单一的可执行文件。
- `--windowed`: 运行程序时不显示黑色的命令行窗口。
- `--name StudyPandaAttackDemo`: 指定生成的文件名为 `StudyPandaAttackDemo.exe`。
- `--add-data "源;目标"`: 这是在告诉打包工具需要将哪些额外文件打包进去。**注意**：Windows下源和目标的分隔符是分号 `;`。

打包过程需要几分钟，请耐心等待。

---

### 步骤五：找到并分发可执行文件

打包完成后，您会在项目文件夹下看到一个新生成的 `dist` 文件夹。

您需要的独立可执行文件就在里面：

`dist\StudyPandaAttackDemo.exe`

您可以将这个 `.exe` 文件单独复制出来，发送给任何使用 Windows 的老师，他们无需任何额外操作，双击即可运行我们制作的程序。
