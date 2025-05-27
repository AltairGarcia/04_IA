#!/usr/bin/env python3
"""
Script to directly examine API analytics data files.

This script provides a direct view of the API analytics data stored in the JSON files,
without requiring the full content_creation.py file to be functional.
"""
import os
import json
import sys
from datetime import datetime
from pprint import pprint

def main():
    """Examine API analytics data files."""
    analytics_dir = os.path.join(os.path.dirname(__file__), "analytics_data")

    if not os.path.exists(analytics_dir):
        print(f"Analytics directory not found: {analytics_dir}")
        return

    daily_usage_file = os.path.join(analytics_dir, "daily_usage.json")
    performance_file = os.path.join(analytics_dir, "api_performance.json")

    print("=" * 60)
    print("API ANALYTICS DATA EXAMINATION")
    print("=" * 60)

    # Check daily usage data
    if os.path.exists(daily_usage_file):
        try:
            with open(daily_usage_file, 'r') as f:
                daily_data = json.load(f)

            print("\nDAILY USAGE DATA:")
            print("-" * 50)

            for date, api_data in daily_data.items():
                print(f"\nDate: {date}")

                calls = sum(api_data[api].get("calls", 0) for api in api_data)
                errors = sum(api_data[api].get("errors", 0) for api in api_data)
                cost = sum(api_data[api].get("estimated_cost", 0.0) for api in api_data)

                print(f"Total Calls: {calls}, Errors: {errors}, Cost: ${cost:.2f}")
                print("API breakdown:")

                for api_name, metrics in api_data.items():
                    if metrics.get("calls", 0) > 0:
                        cost = metrics.get("estimated_cost", 0.0)
                        calls = metrics.get("calls", 0)
                        errors = metrics.get("errors", 0)

                        print(f"  - {api_name}: {calls} calls, {errors} errors, ${cost:.4f}")

                        # Show specific metrics based on API type
                        if "tokens" in metrics and metrics["tokens"] > 0:
                            print(f"    * {metrics['tokens']} tokens")
                        elif "seconds" in metrics and metrics["seconds"] > 0:
                            print(f"    * {metrics['seconds']:.1f} seconds")
                        elif "images" in metrics and metrics["images"] > 0:
                            print(f"    * {metrics['images']} images")
        except Exception as e:
            print(f"Error reading daily usage data: {str(e)}")
    else:
        print(f"\nDaily usage file not found: {daily_usage_file}")

    # Check performance data
    if os.path.exists(performance_file):
        try:
            with open(performance_file, 'r') as f:
                perf_data = json.load(f)

            print("\nPERFORMANCE DATA:")
            print("-" * 50)

            for api_name, metrics in perf_data.items():
                calls = metrics.get("calls", 0)
                if calls > 0:
                    latency = metrics.get("avg_latency", 0.0)
                    success_rate = metrics.get("success_rate", 1.0) * 100

                    print(f"{api_name}: {calls} calls, {latency:.2f}s avg latency, {success_rate:.1f}% success")

        except Exception as e:
            print(f"Error reading performance data: {str(e)}")
    else:
        print(f"\nPerformance file not found: {performance_file}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
