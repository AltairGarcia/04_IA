#!/usr/bin/env python3
"""
Fix analytics data to add missing estimated_cost fields for new APIs.
"""

import json
import os
from datetime import datetime

def fix_analytics_data():
    """Fix the analytics data to add missing estimated_cost fields."""
    analytics_dir = os.path.join(os.path.dirname(__file__), "analytics_data")
    daily_usage_file = os.path.join(analytics_dir, "daily_usage.json")

    print(f"Looking for file: {daily_usage_file}")

    if not os.path.exists(daily_usage_file):
        print("Daily usage file not found.")
        return

    print("Loading current data...")
    # Load current data
    with open(daily_usage_file, 'r') as f:
        daily_data = json.load(f)

    # APIs that should have estimated_cost
    cost_tracked_apis = ["gemini", "elevenlabs", "dalle", "stabilityai", "assemblyai", "deepgram", "newsapi", "openweather", "tavily"]

    updated = False

    print("Checking data...")
    # Fix each date's data
    for date_str, date_data in daily_data.items():
        print(f"Checking date: {date_str}")
        for api_name, api_data in date_data.items():
            print(f"  Checking API: {api_name}")
            if api_name in cost_tracked_apis and "estimated_cost" not in api_data:
                api_data["estimated_cost"] = 0.0
                print(f"Added estimated_cost field to {api_name} for {date_str}")
                updated = True
            elif api_name in cost_tracked_apis:
                print(f"  {api_name} already has estimated_cost")

    if updated:
        # Save fixed data
        with open(daily_usage_file, 'w') as f:
            json.dump(daily_data, f, indent=2)
        print("Analytics data fixed successfully!")
    else:
        print("No updates needed.")

if __name__ == "__main__":
    fix_analytics_data()
