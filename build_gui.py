import subprocess
import sys
import os

def run_command(command, cwd=None, shell=False):
    """Runs a command and prints its output in real-time."""
    print(f"--- 正在于 '{cwd or os.getcwd()}' 目录下执行: {' '.join(command)} ---")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, encoding='utf-8', cwd=cwd, shell=shell)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"--- 命令执行失败: {e} ---")
        sys.exit(1)
    print("--- 命令执行成功 ---")

def main():
    """主构建流程：安装依赖并打包GUI应用。"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print(">>> 步骤 1/2: 安装项目依赖 (pip install .)...")
    run_command([sys.executable, "-m", "pip", "install", "."], cwd=project_root)

    print("\n>>> 步骤 2/2: 开始使用 PyInstaller 打包GUI应用...")
    name = "StudyPandaAttackDemo_GUI"
    entry_script = "app_gui.py"
    
    data_separator = ':' if sys.platform != "win32" else ';'

    pyinstaller_command = [
        "pyinstaller",
        "--noconfirm",
        "--windowed",
        f"--name={name}",
        "--paths=.", # 将根目录添加到路径中，以便找到 backend 包
        f"--add-data=backend/Alibaba-PuHuiTi-Medium.ttf{data_separator}backend",
        f"--add-data=backend/imagenet_class_index_cn.json{data_separator}backend",
        f"--add-data=backend/images{data_separator}backend/images",
        entry_script,
    ]

    run_command(pyinstaller_command)
    print(f"\n打包完成！应用位于 dist/ 目录。")

if __name__ == "__main__":
    main()