#!/usr/bin/env python
"""
Setup Environment Script for LangGraph 101 project.

This script checks for required dependencies and helps set up the environment
for the LangGraph 101 project.
"""
import os
import sys
import subprocess
import time
import importlib.util
from typing import List, Tuple
import shutil

# Required packages for error handling and resilience
REQUIRED_PACKAGES = {
    "python-dotenv": "dotenv",
    "psutil": "psutil",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "streamlit": "streamlit"
}

def check_package(package_name: str, import_name: str) -> bool:
    """Check if a package is installed.

    Args:
        package_name: The pip package name
        import_name: The import module name

    Returns:
        True if installed, False otherwise
    """
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except ImportError:
        return False


def check_dependencies() -> List[str]:
    """Check for missing dependencies.

    Returns:
        List of missing package names
    """
    missing_packages = []

    for package_name, import_name in REQUIRED_PACKAGES.items():
        if not check_package(import_name, import_name):
            missing_packages.append(package_name)

    return missing_packages


def install_packages(packages: List[str]) -> bool:
    """Install missing packages.

    Args:
        packages: List of package names to install

    Returns:
        True if successful, False otherwise
    """
    if not packages:
        return True

    print(f"Installing packages: {', '.join(packages)}")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False


def check_directories() -> None:
    """Check if required directories exist, create if needed."""
    directories = [
        "analytics_data",
        "error_logs",
        "content_output"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")


def check_env_file() -> Tuple[bool, bool]:
    """Check if .env file exists, create from template if needed.

    Returns:
        Tuple[env_exists, was_created]
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    template_path = os.path.join(os.path.dirname(__file__), ".env.template")

    if os.path.exists(env_path):
        return True, False

    if os.path.exists(template_path):
        # Copy template to .env
        shutil.copy(template_path, env_path)
        return True, True

    # Create minimal .env
    with open(env_path, "w") as f:
        f.write("# Created by setup_environment.py\n\n")
        f.write("# ===== API Keys =====\n")
        f.write("API_KEY=your_gemini_api_key_here\n")
        f.write("TAVILY_API_KEY=your_tavily_api_key_here\n\n")
        f.write("# ===== System Settings =====\n")
        f.write("ENVIRONMENT=development\n\n")

    return True, True


def main():
    """Main function."""
    print("=" * 60)
    print("LangGraph 101 Environment Setup")
    print("=" * 60)
    print()

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("WARNING: Python 3.8 or higher is recommended")

    # Check dependencies
    print("\nChecking dependencies...")
    missing_packages = check_dependencies()

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")

        # Prompt to install
        install = input("Install missing packages? (y/n): ")
        if install.lower() in ('y', 'yes'):
            success = install_packages(missing_packages)
            if success:
                print("All packages installed successfully")
            else:
                print("Some packages failed to install")
                print("Please install them manually:")
                print(f"  pip install {' '.join(missing_packages)}")
    else:
        print("All required packages are installed")

    # Check directories
    print("\nChecking directories...")
    check_directories()

    # Check .env file
    print("\nChecking .env file...")
    env_exists, was_created = check_env_file()

    if was_created:
        print("Created .env file from template")
        print("Please edit the .env file to set your API keys and configuration")
    elif env_exists:
        print(".env file already exists")

    # Final message
    print("\n" + "=" * 60)
    print("Environment setup complete!")
    print("\nTo run the application:")
    print("  streamlit run streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
