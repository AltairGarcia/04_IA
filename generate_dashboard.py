#!/usr/bin/env python
"""
Script to generate performance dashboard for LangGraph 101 project.
"""
import os
import argparse
import webbrowser
import importlib.util
from typing import Optional

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    required_packages = [
        ('matplotlib', 'Matplotlib is required for charting'),
        ('pandas', 'Pandas is required for data analysis'),
        ('numpy', 'NumPy is required for numerical operations')
    ]

    missing = []
    for package, message in required_packages:
        if importlib.util.find_spec(package) is None:
            missing.append(f"{package}: {message}")

    if missing:
        print("Missing required dependencies:")
        for msg in missing:
            print(f"  - {msg}")
        print("\nPlease install missing dependencies with:")
        print("pip install matplotlib pandas numpy")
        return False

    return True

def run_dashboard(output_dir: Optional[str] = None, open_browser: bool = True) -> None:
    """Run the performance dashboard generator.

    Args:
        output_dir: Optional directory to save dashboard output
        open_browser: Whether to open the dashboard in a browser
    """
    # Import the dashboard module
    from performance_dashboard import PerformanceDashboard

    print("Generating LangGraph 101 Performance Dashboard...")
    dashboard = PerformanceDashboard(output_dir=output_dir)
    outputs = dashboard.generate_dashboard()

    if not outputs:
        print("Error: Failed to generate dashboard.")
        return

    summary_path = outputs.get('summary')
    if summary_path and os.path.exists(summary_path):
        print(f"\nDashboard generated successfully at: {summary_path}")

        if open_browser:
            print("Opening dashboard in your web browser...")
            webbrowser.open(f"file://{os.path.abspath(summary_path)}")
    else:
        print("\nDashboard outputs generated but no summary file was created.")
        for chart_type, file_path in outputs.items():
            print(f"- {chart_type}: {file_path}")

def main() -> None:
    """Main entry point for the dashboard generator script."""
    parser = argparse.ArgumentParser(
        description="Generate a performance dashboard for LangGraph 101 project"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to save dashboard files (default: dashboard_output)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the dashboard in a browser"
    )

    args = parser.parse_args()

    if check_dependencies():
        run_dashboard(
            output_dir=args.output_dir,
            open_browser=not args.no_browser
        )

if __name__ == "__main__":
    main()
