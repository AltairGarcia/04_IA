#!/usr/bin/env python
"""
Installation script for LangGraph 101 project.
Installs all required dependencies, sets up the environment, and runs initial system checks.
"""
import os
import sys
import subprocess
import argparse
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if the Python version meets requirements."""
    min_version = (3, 9)
    current_version = sys.version_info[:2]

    if current_version < min_version:
        print(f"Error: Python {min_version[0]}.{min_version[1]} or higher is required.")
        print(f"Current version: Python {current_version[0]}.{current_version[1]}")
        return False

    print(f"✅ Python version OK: {current_version[0]}.{current_version[1]}")
    return True

def get_package_lists() -> Tuple[List[str], List[str], List[str]]:
    """Get lists of required packages for different installations.

    Returns:
        Tuple of (core_packages, web_packages, dev_packages)
    """
    # Core packages required for basic functionality
    core_packages = [
        "python-dotenv",
        "requests",
        "psutil",
        "matplotlib",
        "pandas",
        "numpy",
        "jsonschema",
        "langchain",
        "langchain-core",
        "langchain-community",
        "langgraph"
    ]

    # Web interface packages
    web_packages = [
        "streamlit",
        "streamlit-extras",
        "plotly",
        "altair"
    ]

    # Development packages
    dev_packages = [
        "pytest",
        "pytest-cov",
        "black",
        "flake8",
        "mypy",
        "isort",
        "pre-commit"
    ]

    return core_packages, web_packages, dev_packages

def install_packages(packages: List[str], upgrade: bool = False) -> bool:
    """Install the specified packages using pip.

    Args:
        packages: List of package names to install
        upgrade: Whether to upgrade existing packages

    Returns:
        True if installation succeeded, False otherwise
    """
    if not packages:
        return True

    cmd = [sys.executable, "-m", "pip", "install"]

    if upgrade:
        cmd.append("--upgrade")

    cmd.extend(packages)

    print(f"\nInstalling {len(packages)} packages...")
    print(" ".join(cmd))

    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def setup_environment() -> bool:
    """Set up the necessary environment for the project.

    Returns:
        True if setup succeeded, False otherwise
    """
    # Create required directories
    dirs = ["analytics_data", "error_logs", "content_output"]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create .env file template if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# LangGraph 101 Environment Variables

# API Keys
GEMINI_API_KEY=
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
STABILITY_API_KEY=
SERP_API_KEY=
PEXELS_API_KEY=

# Email Configuration
SMTP_SERVER=
SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_PASSWORD=
EMAIL_SENDER=
EMAIL_RECIPIENTS=comma,separated,emails

# System Configuration
LOG_LEVEL=INFO
DEBUG=False
""")
        print("✅ Created .env template file")
        print("⚠️  Remember to fill in your API keys in the .env file")
    else:
        print("ℹ️  Using existing .env file")

    return True

def run_system_check() -> bool:
    """Run a system check to verify installation.

    Returns:
        True if system check passed, False otherwise
    """
    try:
        print("\nRunning system check...")
        from system_initialization import check_dependencies, initialize_all_systems

        # Check for missing dependencies
        missing = check_dependencies()
        if missing:
            print(f"⚠️  Missing optional dependencies: {', '.join(missing)}")
            print("These can be installed with:")
            print(f"pip install {' '.join(missing)}")
        else:
            print("✅ All dependencies installed")

        # Initialize systems (but don't start threads)
        result = initialize_all_systems()

        if result['status'] == 'success':
            print("✅ System initialized successfully")

            # Check component status
            all_enabled = True
            for component, enabled in result['components'].items():
                status = "✅ Enabled" if enabled else "❌ Disabled"
                print(f"  - {component}: {status}")
                if not enabled:
                    all_enabled = False

            return all_enabled
        else:
            print(f"❌ System initialization failed: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error during system check: {str(e)}")
        return False

def main() -> None:
    """Main entry point for installation script."""
    parser = argparse.ArgumentParser(
        description="Install LangGraph 101 and its dependencies"
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Install web interface dependencies"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Install development dependencies"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Install all dependencies (core, web, dev)"
    )
    parser.add_argument(
        "--upgrade", action="store_true",
        help="Upgrade existing packages"
    )
    parser.add_argument(
        "--no-setup", action="store_true",
        help="Skip environment setup"
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="Skip system check"
    )

    args = parser.parse_args()

    print("LangGraph 101 Installation")
    print("=========================\n")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Get package lists
    core_packages, web_packages, dev_packages = get_package_lists()

    # Determine which packages to install
    packages_to_install = core_packages.copy()

    if args.web or args.all:
        packages_to_install.extend(web_packages)

    if args.dev or args.all:
        packages_to_install.extend(dev_packages)

    # Install packages
    if install_packages(packages_to_install, args.upgrade):
        print("✅ Package installation complete")
    else:
        print("❌ Package installation failed")
        sys.exit(1)

    # Setup environment
    if not args.no_setup:
        if setup_environment():
            print("✅ Environment setup complete")
        else:
            print("❌ Environment setup failed")
            sys.exit(1)

    # Run system check
    if not args.no_check:
        if run_system_check():
            print("\n✅ All checks passed. LangGraph 101 is ready to use!")
            print("\nTo start the web interface:")
            print("  streamlit run streamlit_app.py")
            print("\nTo generate a performance dashboard:")
            print("  python generate_dashboard.py")
        else:
            print("\n⚠️  Some checks failed. Review the output above for details.")
            sys.exit(1)

if __name__ == "__main__":
    main()
