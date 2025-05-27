"""
AI Error Log Analyzer using GeminiAPI
Usage: python ai_error_analyzer.py [N]
N = number of recent errors to analyze (default: 10)
"""
import os
import sys
import json
from tools import GeminiAPI
from analytics_dashboard import load_analytics_data, ERROR_TRACKING_FILE

def main():
    num_errors = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    # Load recent errors
    errors = load_analytics_data(ERROR_TRACKING_FILE)
    if not errors:
        print("No error data found.")
        sys.exit(0)
    # Sort and select most recent errors
    errors = sorted(errors, key=lambda e: e.get('timestamp', ''), reverse=True)[:num_errors]
    error_summary = "\n".join(f"[{e.get('timestamp')}] {e.get('category')}: {e.get('message')}" for e in errors)
    prompt = (
        "You are an expert AI debugging assistant. "
        "Analyze the following recent error log entries and provide a root cause analysis, "
        "possible solutions, and recommendations for developers.\n\n"
        f"Error log:\n{error_summary}"
    )
    gemini = GeminiAPI(api_key)
    print("Analyzing errors with Gemini...\n")
    try:
        analysis = gemini.generate_content(prompt)
        print(analysis)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
