"""
Unified AI Tools Launcher for LangGraph 101
Usage: python ai_tools_launcher.py
"""
import subprocess
import sys
import os

TOOLS = [
    ("AI Code Generation", "generate_code.py", "Enter your code prompt:"),
    ("AI Error Log Analyzer", "ai_error_analyzer.py", "Enter number of recent errors to analyze (default 10):"),
    ("AI Agent CLI (coding/testing)", "agent_cli.py", "Enter your natural language command:"),
    ("AI Performance Analyzer", "ai_performance_analyzer.py", "Enter number of recent metrics to analyze (default 20):"),
    ("AI Documentation Generator", "ai_doc_generator.py", "Enter the path to the code file to document:")
]

def main():
    while True:
        print("\n=== Unified AI Tools Launcher ===")
        for idx, (name, _, _) in enumerate(TOOLS, 1):
            print(f"{idx}. {name}")
        print("0. Exit")
        choice = input("Select a tool: ").strip()
        if choice == "0":
            print("Exiting.")
            break
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(TOOLS):
                print("Invalid choice.")
                continue
        except ValueError:
            print("Invalid input.")
            continue
        name, script, prompt = TOOLS[idx]
        arg = input(prompt + " ").strip()
        cmd = [sys.executable, script]
        if arg:
            cmd.append(arg)
        print(f"\nRunning: {name}\n{' '.join(cmd)}\n")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
