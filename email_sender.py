"""
Email Sender for LangGraph 101 project.

This module handles sending emails with conversation exports.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv


def send_email(recipient_email: str,
              subject: str,
              body: str,
              attachment_path: Optional[str] = None,
              attachment_name: Optional[str] = None) -> tuple:
    """Send an email with an optional attachment.

    Args:
        recipient_email: Email address to send to.
        subject: Email subject.
        body: Email body content.
        attachment_path: Optional path to a file to attach.
        attachment_name: Optional name for the attachment.

    Returns:
        Tuple of (success, message).
    """
    # Load environment variables
    load_dotenv(encoding='utf-16-le')

    # Get SMTP configuration from environment
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    sender_email = os.getenv("SENDER_EMAIL")

    # Check if SMTP configuration is available
    if not all([smtp_server, smtp_username, smtp_password, sender_email]):
        missing = []
        if not smtp_server: missing.append("SMTP_SERVER")
        if not smtp_username: missing.append("SMTP_USERNAME")
        if not smtp_password: missing.append("SMTP_PASSWORD")
        if not sender_email: missing.append("SENDER_EMAIL")

        return (False, f"Missing email configuration: {', '.join(missing)}")

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach body
        msg.attach(MIMEText(body, 'plain'))

        # Attach file if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as file:
                attachment = MIMEApplication(file.read())
                file_name = attachment_name or os.path.basename(attachment_path)
                attachment.add_header('Content-Disposition',
                                     f'attachment; filename="{file_name}"')
                msg.attach(attachment)

        # Connect to SMTP server and send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        return (True, "Email sent successfully")

    except Exception as e:
        return (False, f"Failed to send email: {str(e)}")


def email_conversation(messages: List[Dict[str, Any]],
                      recipient_email: str,
                      export_format: str = "html",
                      persona_name: str = "AI") -> tuple:
    """Email a conversation export.

    Args:
        messages: List of message dictionaries.
        recipient_email: Email address to send to.
        export_format: Format to export (text, html, json, csv).
        persona_name: Name of the AI persona.

    Returns:
        Tuple of (success, message).
    """
    # This is a placeholder that would use the export module
    # and the email sender function to export and send the conversation

    # In a real implementation, we would:
    # 1. Export the conversation to a temporary file
    # 2. Send the email with the file attached
    # 3. Delete the temporary file

    return (False, "Email sending not implemented yet")
