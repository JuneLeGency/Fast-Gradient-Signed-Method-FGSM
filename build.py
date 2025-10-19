import subprocess
import sys
import os

def run_command(command):
    """运行命令并实时打印输出，如果失败则退出。"""
    print(f"--- 正在执行: {' '.join(command)} ---")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            print(f"--- 命令执行失败，退出码: {rc} ---")
            sys.exit(rc)
    except FileNotFoundError:
        print(f"错误：命令 '{command[0]}' 未找到。请确保它已安装并位于您的 PATH 中。")
        sys.exit(1)
    print("--- 命令执行成功 ---")

def main():
    """主构建流程"""
    # 确保我们在脚本所在的目录中运行
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 步骤 1: 如果有虚拟环境，先激活。对于打包脚本，我们直接使用全局或已激活的python
    print(">>> 步骤 1: 安装项目依赖 (从 pyproject.toml)...")
    # 使用 sys.executable 确保我们用的是当前运行此脚本的 Python 解释器关联的 pip
    run_command([sys.executable, "-m", "pip", "install", "."])

    # 步骤 2: 安装 PyInstaller
    print(">>> 步骤 2: 安装 PyInstaller...")
    run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 步骤 3: 运行 PyInstaller 进行打包
    print(">>> 步骤 3: 开始构建应用程序...")
    
    name = "StudyPandaAttackDemo"
    entry_script = "app_gui.py"
    
    # 根据不同平台设置 --add-data 的分隔符
    data_separator = ':'
    if sys.platform == "win32":
        data_separator = ';'

    pyinstaller_command = [
        "pyinstaller",
        "--windowed",
        "--clean",  # 清理之前的构建缓存
        f"--name={name}",
        f"--add-data=Alibaba-PuHuiTi-Medium.ttf{data_separator}.",
        f"--add-data=imagenet_class_index_cn.json{data_separator}.",
        f"--add-data=images{data_separator}images",
        entry_script,
    ]

    run_command(pyinstaller_command)

    print(f"\n>>> 构建完成! <<<")
    if sys.platform == "darwin":
        print(f"应用程序位于: dist/{name}.app")
    elif sys.platform == "win32":
        print(f"可执行文件位于: dist\\{name}.exe")
    else:
        print(f"可执行文件位于: dist/{name}")

if __name__ == "__main__":
    main()
