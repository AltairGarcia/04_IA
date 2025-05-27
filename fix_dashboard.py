"""
Fix script for the content creation dashboard.

This script updates the content_dashboard.py file to add better error handling.
"""
import re

# Read the current dashboard file
dashboard_path = 'content_dashboard.py'
with open(dashboard_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add improved error handling for content creator initialization
pattern = r'(if "content_creator_initialized" not in st\.session_state:.*?)(\s+config = load_config\(\).*?st\.session_state\.content_creator_initialized = True)'
replacement = r'\1\n    try:\2\n    except Exception as e:\n        st.error(f"Failed to initialize content creator: {str(e)}")\n        st.info("Please check your .env file and make sure it contains the necessary API keys.")\n        return'

# Use re.DOTALL to match across multiple lines
modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the updated content
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(modified_content)

print("Successfully updated content_dashboard.py with improved error handling.")
