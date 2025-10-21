

import os
import subprocess
import sys
import argparse

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
APP_NAME = "StudyPandaAttackDemo_GUI"
ENTRY_SCRIPT = "app_gui.py"

# --- Helper Functions ---
def is_windows():
    """Check if the current operating system is Windows."""
    return sys.platform == "win32"

def run_command(command, cwd=None):
    """
    Runs a command and prints its output in real-time.
    Uses shell=True on Windows for better compatibility.
    """
    shell = is_windows()
    print(f"\n--- Running command in '{cwd or PROJECT_ROOT}': {' '.join(command)} ---")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=cwd,
            shell=shell
        )
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"--- Command failed: {e} ---")
        sys.exit(1)
    print("--- Command successful ---")

def get_python_executable():
    """Get the path to the Python executable in the virtual environment."""
    if not os.path.exists(VENV_DIR):
        print(f"Error: Virtual environment not found at '{VENV_DIR}'.")
        print("Please run 'uv venv .venv' to create it first.")
        sys.exit(1)
        
    if is_windows():
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

# --- Main Build Function ---
def build_application():
    """
    Installs dependencies and packages the GUI application using PyInstaller.
    """
    python_executable = get_python_executable()

    print(">>> Step 1/3: Installing project dependencies...")
    run_command([python_executable, "-m", "pip", "install", "."])

    print("\n>>> Step 2/3: Installing PyInstaller...")
    run_command([python_executable, "-m", "pip", "install", "pyinstaller"])

    print("\n>>> Step 3/3: Packaging application with PyInstaller...")
    
    # Platform-specific path separator for --add-data
    data_separator = ';' if is_windows() else ':'

    # Define data to be bundled with the application
    data_to_bundle = [
        'backend/Alibaba-PuHuiTi-Medium.ttf',
        'backend/en_to_cn_mapping.json',
        'backend/images'
    ]

    pyinstaller_executable = os.path.join(os.path.dirname(python_executable), "pyinstaller")

    pyinstaller_command = [
        pyinstaller_executable,
        "--noconfirm",    # Overwrite previous builds without asking
        "--windowed",     # Create a GUI app without a console window
        f"--name={APP_NAME}",
        "--clean",        # Clean PyInstaller cache and remove temporary files
    ]

    # Add data files to the command
    for data_path in data_to_bundle:
        destination_folder = os.path.dirname(data_path)
        pyinstaller_command.append(f"--add-data={data_path}{data_separator}{destination_folder}")

    pyinstaller_command.append(ENTRY_SCRIPT)

    run_command(pyinstaller_command)

    print(f"\nBuild complete! Application is in the 'dist/{APP_NAME}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Build script for the {APP_NAME} application."
    )
    parser.add_argument(
        "action",
        choices=["build"],
        help="The action to perform. 'build' will package the application."
    )
    args = parser.parse_args()

    if args.action == "build":
        build_application()
