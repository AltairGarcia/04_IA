"""
AI Code Generation Script using GeminiAPI
Usage: python generate_code.py "Your prompt here"
"""
import sys
from tools import GeminiAPI
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_code.py 'Your prompt here'")
        sys.exit(1)
    prompt = sys.argv[1]
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    gemini = GeminiAPI(api_key)
    print("Generating code...\n")
    try:
        code = gemini.generate_content(prompt)
        print(code)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
