import os
import subprocess
import sys
import argparse

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")

# --- Helper Functions ---
def is_windows():
    return sys.platform == "win32"

def run_command(command, cwd=None, shell=False):
    """Runs a command and prints its output in real-time."""
    print(f"\n--- 正在于 '{cwd or os.getcwd()}' 目录下执行: {' '.join(command)} ---")
    try:
        # shell=True is needed for commands like `source` or `&&` on Windows
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

def get_python_executable():
    """获取虚拟环境中的python解释器路径"""
    if is_windows():
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

# --- Main Functions ---

def setup_environment():
    """创建虚拟环境并安装所有依赖"""
    print(">>> 步骤 1/3: 创建 Python 虚拟环境...")
    if not os.path.exists(VENV_DIR):
        run_command([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("虚拟环境已存在，跳过创建。")

    python_executable = get_python_executable()
    print("\n>>> 步骤 2/3: 安装后端 Python 依赖...")
    run_command([python_executable, "-m", "pip", "install", "."], cwd=PROJECT_ROOT)

    print("\n>>> 步骤 3/3: 安装前端 Javascript 依赖...")
    run_command(["npm", "install"], cwd=FRONTEND_DIR, shell=is_windows())
    
    print("\n环境设置完成！")

def build_frontend():
    """构建前端静态文件"""
    print(">>> 正在构建前端静态文件 (npm run build)...")
    run_command(["npm", "run", "build"], cwd=FRONTEND_DIR, shell=is_windows())
    print("前端构建完成！产物位于 frontend/build 目录。")

def run_packaging():
    """使用 PyInstaller 打包应用"""
    print(">>> 步骤 1/2: 确保 PyInstaller 已安装...")
    python_executable = get_python_executable()
    run_command([python_executable, "-m", "pip", "install", "pyinstaller"])

    print("\n>>> 步骤 2/2: 开始使用 PyInstaller 打包...")
    name = "StudyPandaAttackDemo"
    entry_script = os.path.join(BACKEND_DIR, "main.py")
    
    data_separator = ':' if not is_windows() else ';'

    pyinstaller_command = [
        "pyinstaller",
        "--noconfirm",
        "--windowed",
        f"--name={name}",
        f"--add-data={os.path.join(BACKEND_DIR, 'Alibaba-PuHuiTi-Medium.ttf')}{data_separator}backend",
        f"--add-data={os.path.join(BACKEND_DIR, 'imagenet_class_index_cn.json')}{data_separator}backend",
        f"--add-data={os.path.join(BACKEND_DIR, 'images')}{data_separator}backend/images",
        f"--add-data={os.path.join(FRONTEND_DIR, 'build')}{data_separator}frontend/build",
        entry_script,
    ]

    run_command(pyinstaller_command)
    print(f"\n打包完成！应用位于 dist/ 目录。")

def run_server():
    """以开发模式启动服务"""
    print(">>> 正在启动开发服务器...")
    python_executable = get_python_executable()
    run_command([python_executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"], cwd=BACKEND_DIR)

# --- Argument Parser ---
def main():
    parser = argparse.ArgumentParser(description="人工智能对抗攻击演示平台的构建和打包工具。 সন")
    subparsers = parser.add_subparsers(dest="command", required=True, help="可用的子命令")

    parser_setup = subparsers.add_parser("setup", help="创建虚拟环境并安装所有前后端依赖。 সন")
    parser_setup.set_defaults(func=setup_environment)

    parser_build = subparsers.add_parser("build_frontend", help="仅构建前端静态文件。 সন")
    parser_build.set_defaults(func=build_frontend)

    parser_package = subparsers.add_parser("package", help="将应用打包为独立可执行文件。 সন")
    parser_package.set_defaults(func=run_packaging)

    parser_all = subparsers.add_parser("all", help="执行从环境设置、前端构建到最终打包的完整流程。 সন")
    parser_all.set_defaults(func=lambda: (setup_environment(), build_frontend(), run_packaging()))

    parser_run = subparsers.add_parser("run", help="以开发模式启动后端服务。 সন")
    parser_run.set_defaults(func=run_server)

    args = parser.parse_args()
    args.func()

if __name__ == "__main__":
    main()
