import subprocess
import sys
import os

def run_command(command, cwd=None):
    """在指定目录下运行命令并实时打印输出，如果失败则退出。"""
    print(f"--- 正在于 '{cwd or os.getcwd()}' 目录下执行: {' '.join(command)} ---")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', cwd=cwd)
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
    print("--- 命令执行成功 ---
")

def main():
    """主构建流程：构建前端，安装后端。"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_root, "frontend")
    backend_dir = os.path.join(project_root, "backend")

    # 步骤 1: 安装前端依赖
    print(">>> 步骤 1/4: 安装前端依赖 (npm install)...")
    run_command(["npm", "install"], cwd=frontend_dir)

    # 步骤 2: 构建前端静态文件
    print(">>> 步骤 2/4: 构建前端静态文件 (npm run build)...")
    run_command(["npm", "run", "build"], cwd=frontend_dir)

    # 步骤 3: 安装后端依赖
    print(">>> 步骤 3/4: 安装后端 Python 依赖 (pip install .)...")
    # 使用 sys.executable 确保我们用的是当前运行此脚本的 Python 解释器关联的 pip
    run_command([sys.executable, "-m", "pip", "install", "."], cwd=project_root)

    # 步骤 4: 完成
    print(">>> 步骤 4/4: 构建完成! <<<")
    print("\n项目已准备就绪。请按以下步骤启动统一服务：")
    print("1. 确保您的终端位于项目根目录中。")
    print("2. 运行以下命令：")
    print("   uvicorn backend.main:app --host 0.0.0.0 --port 8000")
    print("\n3. 在浏览器中打开 http://localhost:8000 即可访问应用。")

if __name__ == "__main__":
    main()
