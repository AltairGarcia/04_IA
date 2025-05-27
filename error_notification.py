"""
Error Notification module for LangGraph 101 project.

This module provides automated error detection and notification capabilities
for the LangGraph project.
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from threading import Thread, Event
from dotenv import load_dotenv
from pathlib import Path
import socket

# Import local modules
from error_handling import ErrorCategory
from analytics_dashboard import load_analytics_data, ERROR_TRACKING_FILE, API_USAGE_FILE

# Use robust configuration loading
try:
    from config_robust import load_env_file_safely
    import os
    
    # Load environment variables safely
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    env_vars = load_env_file_safely(env_path)
    
    # Update os.environ with loaded variables
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            
except ImportError:
    # Fallback to dotenv if config_robust is not available
    from dotenv import load_dotenv
    load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NOTIFICATION_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "analytics_data", "notification_config.json")
DEFAULT_CHECK_INTERVAL = 3600  # Check every hour by default
MAX_ERRORS_THRESHOLD = int(os.getenv("ERROR_THRESHOLD", "10"))  # Alert if more than 10 errors in the time window
ERROR_WINDOW_HOURS = int(os.getenv("ERROR_WINDOW_HOURS", "24"))  # Look at errors in the last 24 hours
CRITICAL_ERROR_CATEGORIES = [
    ErrorCategory.AUTHENTICATION_ERROR.value,
    ErrorCategory.SERVER_API_ERROR.value,
    ErrorCategory.TIMEOUT_ERROR.value,
    ErrorCategory.MEMORY_ERROR.value  # Add memory errors as critical
]

# Email configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "").split(",") if os.getenv("ALERT_EMAIL_TO") else []
USE_TLS = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "t", "yes")

# Domain-specific error patterns to watch for (regex patterns)
DOMAIN_SPECIFIC_PATTERNS = [
    r"memory\s+corruption",  # Memory corruption issues
    r"api\s+rate\s+limit",  # Rate limiting issues
    r"token\s+limit\s+exceeded",  # Token limits for LLM APIs
    r"invalid\s+json\s+response",  # Broken LLM responses
    r"execution\s+timeout",  # Long-running operations timeout
]


class ErrorNotifier:
    """Error detection and notification system."""

    def __init__(self, email_config: Optional[Dict[str, Any]] = None):
        """Initialize the error notifier with email configuration.

        Args:
            email_config: Email configuration with SMTP server details
        """
        # Use environment variables as default if available
        if not email_config and all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, ALERT_EMAIL_FROM, ALERT_EMAIL_TO]):
            email_config = {
                'smtp_server': SMTP_SERVER,
                'smtp_port': SMTP_PORT,
                'username': SMTP_USERNAME,
                'password': SMTP_PASSWORD,
                'sender': ALERT_EMAIL_FROM,
                'recipients': ALERT_EMAIL_TO,
                'use_tls': USE_TLS
            }

        self.email_config = email_config or {}
        self.check_interval = DEFAULT_CHECK_INTERVAL
        self.stop_event = Event()
        self.monitor_thread = None

        # Notification state tracking
        self.last_notification_time = {}  # Track last notification by type
        self.notification_count = {}  # Count notifications by type

        # Load configuration if exists
        self._load_config()

    def _load_config(self) -> None:
        """Load notification configuration from file."""
        if os.path.exists(NOTIFICATION_CONFIG_FILE):
            try:
                with open(NOTIFICATION_CONFIG_FILE, 'r') as f:
                    config = json.load(f)

                    # Update configuration
                    self.email_config.update(config.get('email_config', {}))
                    self.check_interval = config.get('check_interval', DEFAULT_CHECK_INTERVAL)

                    logger.info("Loaded error notification configuration")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load notification configuration: {str(e)}")

    def _save_config(self) -> None:
        """Save current notification configuration to file."""
        os.makedirs(os.path.dirname(NOTIFICATION_CONFIG_FILE), exist_ok=True)

        try:
            config = {
                'email_config': self.email_config,
                'check_interval': self.check_interval,
                'last_updated': datetime.now().isoformat()
            }

            with open(NOTIFICATION_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info("Saved error notification configuration")
        except IOError as e:
            logger.error(f"Failed to save notification configuration: {str(e)}")

    def configure_email(self,
                       smtp_server: str,
                       smtp_port: int,
                       username: str,
                       password: str,
                       sender: str,
                       recipients: List[str],
                       use_tls: bool = True) -> None:
        """Configure email notification settings.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            sender: Sender email address
            recipients: List of recipient email addresses
            use_tls: Whether to use TLS encryption
        """
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,  # In a production environment, use a secure credential store
            'sender': sender,
            'recipients': recipients,
            'use_tls': use_tls
        }

        self._save_config()

    def set_check_interval(self, seconds: int) -> None:
        """Set the interval for checking errors.

        Args:
            seconds: Interval in seconds
        """
        self.check_interval = max(60, seconds)  # Minimum 1 minute
        self._save_config()

    def start_monitoring(self) -> None:
        """Start the error monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Error monitoring is already running")
            return

        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Started error monitoring thread")

    def stop_monitoring(self) -> None:
        """Stop the error monitoring thread."""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            logger.warning("Error monitoring is not running")
            return
        self.stop_event.set()
        self.monitor_thread.join(timeout=10.0)
        logger.info("Stopped error monitoring thread")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(f"Error monitoring started with {self.check_interval}s check interval")

        while not self.stop_event.is_set():
            try:
                # Check for errors
                self._check_for_errors()

                # Wait for next check or until stop event
                self.stop_event.wait(timeout=self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                # Wait a bit before retrying to avoid tight loops on persistent errors
                time.sleep(60)

    def _check_for_errors(self) -> None:
        """Check for errors that need notification."""
        # Load error data
        error_data = load_analytics_data(ERROR_TRACKING_FILE)

        if not error_data:
            return

        # Get current time
        now = datetime.now()
        time_window = timedelta(hours=ERROR_WINDOW_HOURS)
        threshold_time = now - time_window

        # Filter recent errors
        recent_errors = [
            error for error in error_data
            if datetime.fromisoformat(error['timestamp']) >= threshold_time
        ]

        # Critical errors
        critical_errors = [
            error for error in recent_errors
            if error['category'] in CRITICAL_ERROR_CATEGORIES
        ]

        # Check for alert conditions
        should_alert = False
        alert_message = ""
        notification_type = "general"  # Default notification type

        # 1. Too many errors overall
        if len(recent_errors) >= MAX_ERRORS_THRESHOLD:
            should_alert = True
            notification_type = "high_error_rate"
            alert_message += f"⚠️ High error rate detected: {len(recent_errors)} errors in the last {ERROR_WINDOW_HOURS} hours\n\n"

        # 2. Critical errors
        if critical_errors:
            should_alert = True
            notification_type = "critical_errors"
            alert_message += f"🚨 {len(critical_errors)} critical errors detected in the last {ERROR_WINDOW_HOURS} hours\n\n"

            # Group by category
            categories = {}
            for error in critical_errors:
                cat = error['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(error)

            for cat, errors in categories.items():
                alert_message += f"  - {cat}: {len(errors)} errors\n"
                # Show most recent error message
                most_recent = max(errors, key=lambda e: e['timestamp'])
                alert_message += f"    Latest: {most_recent['message']}\n"

                # System recommendations based on error category
                if cat == ErrorCategory.AUTHENTICATION_ERROR.value:
                    alert_message += f"    Recommendation: Check API key validity and permissions\n"
                elif cat == ErrorCategory.SERVER_API_ERROR.value:
                    alert_message += f"    Recommendation: Verify API endpoint status and request format\n"
                elif cat == ErrorCategory.TIMEOUT_ERROR.value:
                    alert_message += f"    Recommendation: Consider increasing timeout thresholds or optimizing requests\n"
                elif cat == ErrorCategory.MEMORY_ERROR.value:
                    alert_message += f"    Recommendation: Investigate memory leaks or increase resource allocation\n"

        # 3. API health check
        api_data = load_analytics_data(API_USAGE_FILE)
        if api_data:
            recent_api_calls = [
                call for call in api_data
                if datetime.fromisoformat(call['timestamp']) >= threshold_time
            ]

            if recent_api_calls:
                # Calculate error rate per API
                api_stats = {}
                for call in recent_api_calls:
                    api = call['api']
                    status = call['status']

                    if api not in api_stats:
                        api_stats[api] = {'total': 0, 'errors': 0}

                    api_stats[api]['total'] += 1
                    if status != 'success':
                        api_stats[api]['errors'] += 1

                # Check for high error rates
                for api, stats in api_stats.items():
                    if stats['total'] >= 5:  # Only check APIs with at least 5 calls
                        error_rate = (stats['errors'] / stats['total']) * 100
                        if error_rate >= 20:  # 20% or higher error rate
                            should_alert = True
                            notification_type = f"api_health_{api}"
                            alert_message += f"🔌 High error rate for {api} API: {error_rate:.1f}% ({stats['errors']}/{stats['total']})\n"
                            alert_message += f"  Recommendation: Check API status and consider implementing circuit breaker\n"

        # 4. Domain-specific error patterns
        import re
        domain_specific_detected = False
        domain_patterns_found = set()

        for error in recent_errors:
            error_msg = error.get('message', '')
            for pattern in DOMAIN_SPECIFIC_PATTERNS:
                if re.search(pattern, error_msg, re.IGNORECASE):
                    domain_specific_detected = True
                    domain_patterns_found.add(pattern)

        if domain_specific_detected:
            should_alert = True
            notification_type = "domain_specific"
            alert_message += f"🔍 Domain-specific error patterns detected:\n"
            for pattern in domain_patterns_found:
                alert_message += f"  - Pattern: {pattern}\n"

        # 5. Check system resources
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent

            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                should_alert = True
                notification_type = "resource_alert"
                alert_message += f"💻 System resource alert:\n"
                if cpu_percent > 90:
                    alert_message += f"  - High CPU usage: {cpu_percent}%\n"
                if memory_percent > 90:
                    alert_message += f"  - High memory usage: {memory_percent}%\n"
                if disk_percent > 90:
                    alert_message += f"  - High disk usage: {disk_percent}%\n"
        except ImportError:
            # psutil not available, skip this check
            logger.debug("psutil not available, skipping system resource checks")
            pass

        # Prevent notification flooding - don't send too many of the same type
        if should_alert:
            # Check if we've sent this type of notification recently
            can_send = self._can_send_notification(notification_type)
            if can_send:
                self._send_notification(f"Error Alert: {notification_type}", alert_message)
                self._record_notification_sent(notification_type)
            else:
                logger.info(f"Suppressing duplicate notification of type: {notification_type}")

    def _can_send_notification(self, notification_type: str) -> bool:
        """Check if we should send this type of notification based on frequency limits.

        Args:
            notification_type: The type of notification

        Returns:
            True if we can send this notification
        """
        now = datetime.now()

        # If never sent this type, we can send it
        if notification_type not in self.last_notification_time:
            return True

        # Check how long since last notification of this type
        last_sent = self.last_notification_time[notification_type]
        time_since_last = now - last_sent

        # Get count of how many we've sent today
        count_today = self.notification_count.get(notification_type, {}).get(now.date().isoformat(), 0)

        # Rules for sending:
        # 1. For critical notifications, allow sending every 30 minutes, max 10 per day
        # 2. For other notifications, allow sending every 2 hours, max 5 per day

        if notification_type.startswith("critical"):
            # Critical notifications: max 1 every 30 minutes, 10 per day
            if time_since_last < timedelta(minutes=30) or count_today >= 10:
                return False
        else:
            # Regular notifications: max 1 every 2 hours, 5 per day
            if time_since_last < timedelta(hours=2) or count_today >= 5:
                return False

        return True

    def _record_notification_sent(self, notification_type: str) -> None:
        """Record that we sent a notification of a particular type.

        Args:
            notification_type: The type of notification
        """
        now = datetime.now()
        today = now.date().isoformat()

        # Update last sent time
        self.last_notification_time[notification_type] = now
        # Update counts
        if notification_type not in self.notification_count:
            self.notification_count[notification_type] = {}

        if today not in self.notification_count[notification_type]:
            self.notification_count[notification_type][today] = 0

        self.notification_count[notification_type][today] += 1

    def _send_notification(self, subject: str, message: str) -> None:
        """Send an email notification with improved reliability.

        Args:
            subject: Email subject
            message: Email message body
        """
        if not self.email_config or not self.email_config.get('recipients'):
            logger.warning("Email not configured, can't send notification")
            # Log the message that would have been sent
            logger.info(f"Would have sent notification: {subject}\n{message}")

            # Save to error logs directory for reference
            try:
                log_dir = os.path.join(os.path.dirname(__file__), "error_logs")
                os.makedirs(log_dir, exist_ok=True)

                log_file = os.path.join(
                    log_dir,
                    f"notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )

                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Subject: {subject}\n\n{message}")

                logger.info(f"Notification saved to {log_file}")
            except Exception as e:
                logger.error(f"Failed to save notification log: {str(e)}")

            return

        # Format the message with HTML for better readability
        html_message = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ color: #cc0000; font-weight: bold; font-size: 18px; }}
                .content {{ margin: 15px 0; white-space: pre-line; }}
                .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px; }}
                .footer {{ font-size: 12px; color: #666; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">LangGraph 101 Error Alert</div>
            <div class="content">{message.replace('\n', '<br>')}</div>
            <div class="metadata">
                <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Host:</strong> {socket.gethostname()}<br>
                <strong>Environment:</strong> {os.getenv('ENVIRONMENT', 'development')}<br>
            </div>
            <div class="footer">
                <p>This is an automated message from the LangGraph 101 Error Notification System.</p>
                <p>To modify notification settings, visit the analytics dashboard or update the .env configuration.</p>
            </div>
        </body>
        </html>
        """

        # Create both plain text and HTML versions of the message
        plain_message = f"{message}\n\n"
        plain_message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        plain_message += f"Host: {socket.gethostname()}\n"
        plain_message += f"Environment: {os.getenv('ENVIRONMENT', 'development')}\n\n"
        plain_message += "This is an automated message from the LangGraph 101 Error Notification System."

        # Number of retry attempts
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Create message
                msg = MIMEMultipart('alternative')
                msg['From'] = self.email_config['sender']
                msg['To'] = ', '.join(self.email_config['recipients'])
                msg['Subject'] = f"LangGraph Alert: {subject}"

                # Attach both plain text and HTML versions
                msg.attach(MIMEText(plain_message, 'plain'))
                msg.attach(MIMEText(html_message, 'html'))

                # Connect to SMTP server
                server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                server.set_debuglevel(0)  # Set to 1 to see detailed SMTP conversation

                # Use longer timeout for unreliable networks
                server.timeout = 30  # 30 seconds timeout

                if self.email_config.get('use_tls', True):
                    server.starttls()

                # Login if credentials provided
                if self.email_config.get('username') and self.email_config.get('password'):
                    server.login(self.email_config['username'], self.email_config['password'])

                # Send email
                server.send_message(msg)
                server.quit()

                logger.info(f"Sent notification: {subject}")

                # Log successful email for analytics
                self._log_notification_sent(subject)

                return  # Success
            except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError,
                  smtplib.SMTPResponseException, ConnectionRefusedError) as e:
                # Network or server issues - retry
                logger.warning(f"SMTP error on attempt {attempt+1}/{max_retries}: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4 seconds
            except Exception as e:
                # Other errors - log and don't retry
                logger.error(f"Failed to send notification: {str(e)}")

                # Save failed notification to disk
                try:
                    log_dir = os.path.join(os.path.dirname(__file__), "error_logs")
                    os.makedirs(log_dir, exist_ok=True)

                    log_file = os.path.join(
                        log_dir,
                        f"failed_notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )

                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write(f"Subject: {subject}\n\nError: {str(e)}\n\n{message}")

                    logger.info(f"Failed notification saved to {log_file}")
                except Exception as e2:
                    logger.error(f"Failed to save notification log: {str(e2)}")

                return  # Give up after logging

        # If we get here, we've exhausted retries
        logger.error(f"Failed to send notification after {max_retries} attempts")

    def _log_notification_sent(self, subject: str) -> None:
        """Log that a notification was successfully sent.

        Args:
            subject: The notification subject
        """
        try:
            log_file = os.path.join(
                os.path.dirname(__file__),
                "analytics_data",
                "notification_history.json"
            )

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Load existing history
            history = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        history = json.load(f)
                except (json.JSONDecodeError, IOError):
                    # If file is corrupted, start with empty history
                    history = []

            # Add new entry
            history.append({
                'timestamp': datetime.now().isoformat(),
                'subject': subject,
                'recipients': len(self.email_config.get('recipients', [])),
                'host': socket.gethostname()
            })

            # Keep only last 100 entries
            history = history[-100:]

            # Save history
            with open(log_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to log notification: {str(e)}")

    def send_test_notification(self) -> bool:
        """Send a test notification to verify configuration.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._send_notification(
                "Test Notification",
                "This is a test notification from the LangGraph Error Notification System.\n"
                "If you received this message, the email configuration is working correctly."
            )
            return True
        except Exception as e:
            logger.error(f"Test notification failed: {str(e)}")
            return False


# Create a singleton instance
_notifier = None

def get_notifier() -> ErrorNotifier:
    """Get the singleton ErrorNotifier instance.

    Returns:
        ErrorNotifier instance
    """
    global _notifier
    if _notifier is None:
        _notifier = ErrorNotifier()
    return _notifier


def setup_error_monitoring(
    smtp_server: Optional[str] = None,
    smtp_port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    sender: Optional[str] = None,
    recipients: Optional[List[str]] = None,
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL,
    start: bool = True,
    use_env_config: bool = True
) -> None:
    """Set up and start error monitoring.

    This function can use values directly provided as parameters, or read from
    environment variables if use_env_config is True.

    Args:
        smtp_server: SMTP server hostname
        smtp_port: SMTP server port
        username: SMTP username
        password: SMTP password
        sender: Sender email address
        recipients: List of recipient email addresses
        check_interval_seconds: Interval between checks in seconds
        start: Whether to start monitoring immediately
        use_env_config: Whether to use environment variables if parameters are not provided
    """
    notifier = get_notifier()

    # Use provided values or environment variables
    if use_env_config:
        smtp_server = smtp_server or SMTP_SERVER
        smtp_port = smtp_port or SMTP_PORT
        username = username or SMTP_USERNAME
        password = password or SMTP_PASSWORD
        sender = sender or ALERT_EMAIL_FROM
        recipients = recipients or ALERT_EMAIL_TO

    # Only configure if we have the necessary values
    if all([smtp_server, sender, recipients]):
        notifier.configure_email(
            smtp_server=smtp_server,
            smtp_port=smtp_port or 587,
            username=username,
            password=password,
            sender=sender,
            recipients=recipients
        )

        notifier.set_check_interval(check_interval_seconds)

        if start:
            notifier.start_monitoring()
            logger.info("Error monitoring started")

            # Send a startup notification if this is a production environment
            if os.getenv('ENVIRONMENT', '').lower() == 'production':
                notifier._send_notification(
                    "System Startup",
                    f"The LangGraph 101 error monitoring system has been started on host {socket.gethostname()}.\n"
                    f"The system will check for errors every {check_interval_seconds/60:.1f} minutes."
                )
    else:
        logger.warning("Email configuration incomplete, error notifications will be logged but not sent")

        if start and not notifier.monitor_thread:
            notifier.start_monitoring()
            logger.info("Error monitoring started (logging mode only)")

def setup_error_monitoring_from_env(start: bool = True) -> None:
    """Set up error monitoring using environment variables.

    This is a convenient wrapper for setup_error_monitoring() that uses
    environment variables for all configuration.

    Args:
        start: Whether to start monitoring immediately
    """
    setup_error_monitoring(
        use_env_config=True,
        start=start
    )


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies are installed.

    Returns:
        Tuple of (all_dependencies_installed, missing_packages)
    """
    missing_packages = []

    # Check for required packages
    try:
        import dotenv
    except ImportError:
        missing_packages.append("python-dotenv")

    try:
        import psutil
    except ImportError:
        missing_packages.append("psutil")

    # Return results
    all_installed = len(missing_packages) == 0
    return all_installed, missing_packages


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)

    # Check dependencies
    all_installed, missing_packages = check_dependencies()
    if not all_installed:
        print("Missing dependencies detected. Please install:")
        print("pip install " + " ".join(missing_packages))
        print()

    # Check if we can read email config from environment
    if all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, ALERT_EMAIL_FROM]) and ALERT_EMAIL_TO:
        print(f"Email configuration found in environment variables")
        print(f"Sender: {ALERT_EMAIL_FROM}")
        print(f"Recipients: {', '.join(ALERT_EMAIL_TO)}")
        print()

        # Start monitoring with environment config
        setup_error_monitoring_from_env(start=False)

        # Test option
        test_option = input("Do you want to send a test email? (y/n): ")
        if test_option.lower() in ('y', 'yes'):
            notifier = get_notifier()
            result = notifier.send_test_notification()
            if result:
                print("Test notification sent successfully!")
            else:
                print("Failed to send test notification. Check logs for details.")
    else:
        print("Email configuration not found in environment variables.")
        print("Please configure email settings in the .env file:")
        print("""
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_username
SMTP_PASSWORD=your_password
ALERT_EMAIL_FROM=alerts@example.com
ALERT_EMAIL_TO=admin@example.com,another@example.com
        """)

    # Manual check
    print("Checking for errors...")
    notifier = get_notifier()
    notifier._check_for_errors()
    print("Done.")
