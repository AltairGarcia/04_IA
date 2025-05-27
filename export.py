"""
Export module for LangGraph 101 project.

This module handles exporting conversation history to different formats.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import html
import csv


def export_to_text(messages: List[Dict[str, Any]],
                   persona_name: str = "AI",
                   user_name: str = "User") -> str:
    """Export conversation to plain text format.

    Args:
        messages: List of message dictionaries.
        persona_name: Name of the AI persona.
        user_name: Name of the user.

    Returns:
        String containing the formatted conversation.
    """
    if not messages:
        return "No conversation to export."

    output = f"Conversation with {persona_name}\n"
    output += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += "=" * 50 + "\n\n"

    for msg in messages:
        timestamp = msg.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
        else:
            formatted_time = ""

        if msg["role"] == "user":
            prefix = f"{user_name} [{formatted_time}]: "
        else:
            prefix = f"{persona_name} [{formatted_time}]: "

        output += prefix + msg["content"] + "\n\n"

    return output


def export_to_html(messages: List[Dict[str, Any]],
                  persona_name: str = "AI",
                  user_name: str = "User") -> str:
    """Export conversation to HTML format.

    Args:
        messages: List of message dictionaries.
        persona_name: Name of the AI persona.
        user_name: Name of the user.

    Returns:
        String containing the HTML formatted conversation.
    """
    if not messages:
        return "<html><body><p>No conversation to export.</p></body></html>"

    # Create HTML header
    output = f"""<!DOCTYPE html>
<html>
<head>
    <title>Conversation with {html.escape(persona_name)}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .message {{ margin-bottom: 15px; padding: 10px; border-radius: 5px; }}
        .user {{ background-color: #f0f0f0; text-align: right; }}
        .ai {{ background-color: #e6f7ff; }}
        .timestamp {{ font-size: 0.8em; color: #666; margin-top: 5px; }}
        .persona-name {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Conversation with {html.escape(persona_name)}</h1>
        <p>Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

    # Add messages
    for msg in messages:
        timestamp = msg.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
        else:
            formatted_time = ""

        if msg["role"] == "user":
            output += f"""    <div class="message user">
        <div class="persona-name">{html.escape(user_name)}</div>
        <div>{html.escape(msg["content"]).replace("\\n", "<br>")}</div>
        <div class="timestamp">{formatted_time}</div>
    </div>
"""
        else:
            output += f"""    <div class="message ai">
        <div class="persona-name">{html.escape(persona_name)}</div>
        <div>{html.escape(msg["content"]).replace("\\n", "<br>")}</div>
        <div class="timestamp">{formatted_time}</div>
    </div>
"""

    # Close HTML tags
    output += """</body>
</html>"""

    return output


def export_to_json(messages: List[Dict[str, Any]],
                  persona_name: str = "AI",
                  user_name: str = "User") -> str:
    """Export conversation to JSON format.

    Args:
        messages: List of message dictionaries.
        persona_name: Name of the AI persona.
        user_name: Name of the user.

    Returns:
        String containing the JSON formatted conversation.
    """
    if not messages:
        return json.dumps({"messages": []})

    export_data = {
        "metadata": {
            "persona": persona_name,
            "user": user_name,
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages)
        },
        "messages": messages
    }

    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_to_csv(messages: List[Dict[str, Any]],
                 persona_name: str = "AI",
                 user_name: str = "User") -> str:
    """Export conversation to CSV format.

    Args:
        messages: List of message dictionaries.
        persona_name: Name of the AI persona.
        user_name: Name of the user.

    Returns:
        String containing the CSV formatted conversation.
    """
    if not messages:
        return "role,speaker,content,timestamp\n"

    import io
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["role", "speaker", "content", "timestamp"])

    # Write messages
    for msg in messages:
        if msg["role"] == "user":
            speaker = user_name
        else:
            speaker = persona_name

        writer.writerow([
            msg["role"],
            speaker,
            msg["content"],
            msg.get("timestamp", "")
        ])

    return output.getvalue()


def save_export(content: str, file_path: str) -> bool:
    """Save exported content to a file.

    Args:
        content: The content to save.
        file_path: The path to save the content to.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception:
        return False


def get_export_formats() -> Dict[str, Dict[str, Any]]:
    """Get available export formats.

    Returns:
        Dictionary mapping format names to functions and file extensions.
    """
    return {
        "text": {
            "function": export_to_text,
            "extension": ".txt",
            "description": "Texto plano (arquivo .txt)"
        },
        "html": {
            "function": export_to_html,
            "extension": ".html",
            "description": "Página HTML (arquivo .html)"
        },
        "json": {
            "function": export_to_json,
            "extension": ".json",
            "description": "Formato JSON (arquivo .json)"
        },
        "csv": {
            "function": export_to_csv,
            "extension": ".csv",
            "description": "Planilha CSV (arquivo .csv)"
        }
    }


def export_conversation(messages: List[Dict[str, Any]],
                       format_name: str = "text",
                       output_file: Optional[str] = None,
                       persona_name: str = "AI",
                       user_name: str = "User") -> tuple:
    """Export conversation to a specified format.

    Args:
        messages: List of message dictionaries.
        format_name: Format to export to (text, html, json, csv).
        output_file: Optional file path to save the export to.
        persona_name: Name of the AI persona.
        user_name: Name of the user.

    Returns:
        Tuple of (success, content or error message, file path if saved).
    """
    # Get available formats
    formats = get_export_formats()

    # Check if format is valid
    if format_name not in formats:
        valid_formats = ", ".join(formats.keys())
        return (False, f"Invalid format: {format_name}. Valid formats: {valid_formats}", None)

    # Get export function for this format
    export_function = formats[format_name]["function"]

    try:
        # Generate content
        content = export_function(messages, persona_name, user_name)

        # Save to file if output_file is provided
        if output_file:
            success = save_export(content, output_file)
            if not success:
                return (False, f"Failed to save to {output_file}", None)
            return (True, content, output_file)

        # Return content without saving
        return (True, content, None)
    except Exception as e:
        return (False, f"Export failed: {str(e)}", None)
