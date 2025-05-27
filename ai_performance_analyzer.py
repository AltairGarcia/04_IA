"""
AI Performance Metrics Analyzer using GeminiAPI
Usage: python ai_performance_analyzer.py [N]
N = number of recent metrics to analyze (default: 20)
"""
import os
import sys
import json
from tools import GeminiAPI

PERF_METRICS_FILE = os.path.join("analytics_data", "performance_metrics.json")

def main():
    num_metrics = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    # Load recent performance metrics
    if not os.path.exists(PERF_METRICS_FILE):
        print("No performance metrics data found.")
        sys.exit(0)
    with open(PERF_METRICS_FILE, "r") as f:
        metrics = json.load(f)
    if not metrics:
        print("No performance metrics data found.")
        sys.exit(0)
    # Sort and select most recent metrics
    metrics = sorted(metrics, key=lambda m: m.get('timestamp', ''), reverse=True)[:num_metrics]
    metric_summary = "\n".join(f"[{m.get('timestamp')}] {m.get('component')} - {m.get('operation')}: {m.get('duration_ms')} ms" for m in metrics)
    prompt = (
        "You are an expert AI performance engineer. "
        "Analyze the following recent system performance metrics and provide a summary, "
        "identify any bottlenecks or anomalies, and suggest concrete optimization steps.\n\n"
        f"Performance metrics:\n{metric_summary}"
    )
    gemini = GeminiAPI(api_key)
    print("Analyzing performance metrics with Gemini...\n")
    try:
        analysis = gemini.generate_content(prompt)
        print(analysis)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
