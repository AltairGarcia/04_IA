"""
AI Documentation Generator using GeminiAPI
Usage: python ai_doc_generator.py <file_to_document>
"""
import sys
import os
from tools import GeminiAPI

def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_doc_generator.py <file_to_document>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    prompt = (
        "You are an expert technical writer. "
        "Read the following code and generate high-quality documentation (docstring or markdown) "
        "explaining its purpose, usage, and key functions.\n\n"
        f"Code:\n{code}"
    )
    gemini = GeminiAPI(api_key)
    print("Generating documentation with Gemini...\n")
    try:
        doc = gemini.generate_content(prompt)
        print(doc)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
