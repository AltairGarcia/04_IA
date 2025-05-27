"""
Centralized logging configuration for LangGraph 101.
Provides structured JSON logging and consistent formatting across modules.
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class JsonFormatter(logging.Formatter):
    """JSON log formatter that creates machine-readable structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Add extra attributes
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data)

def configure_logging(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    use_json: bool = True
) -> None:
    """Configure application-wide logging with both console and file handlers.

    Args:
        log_file: Path to log file (default: langgraph_system.log in project directory)
        console_level: Logging level for console output
        file_level: Logging level for file output
        use_json: Whether to use JSON structured logging format
    """
    # Default log file if not specified
    if not log_file:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        log_file = os.path.join(base_dir, "langgraph_system.log")

    # Create formatters
    if use_json:
        file_formatter = JsonFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s'
        )

    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

def get_contextual_logger(name: str, **context) -> logging.Logger:
    """Get a logger with additional context information attached.

    Args:
        name: Logger name
        **context: Additional context key-value pairs

    Returns:
        Logger with context information
    """
    logger = logging.getLogger(name)

    # Create a filter to add context to log records
    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.extra = context
            return True

    # Add the filter
    for handler in logger.handlers + logging.getLogger().handlers:
        handler.addFilter(ContextFilter())

    return logger
