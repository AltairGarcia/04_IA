"""
Error Integration module for LangGraph 101 project.

This module provides the integration between our error handling system and
various components of the application.
"""
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar
import time
import logging
import os
from datetime import datetime

from error_handling import ErrorHandler, ErrorCategory
from analytics_dashboard import AnalyticsTracker

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
R = TypeVar('R')


def api_call_decorator(
    api_name: str,
    performance_component: Optional[str] = None
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator for API calls to track performance and errors.

    Args:
        api_name: Name of the API being called
        performance_component: Component name for performance tracking

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            start_time = time.time()

            try:
                # Make the API call
                result = func(*args, **kwargs)

                # Log successful API call
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"API call to {api_name} completed in {duration_ms:.2f}ms")

                # Record analytics
                AnalyticsTracker.record_api_call(
                    api_name=api_name,
                    status="success",
                    duration_ms=duration_ms,
                    metadata={
                        "func": func.__name__,
                        "component": performance_component or func.__qualname__.split('.')[0]
                    }
                )

                # Record performance metric if component is specified
                if performance_component:
                    AnalyticsTracker.record_performance_metric(
                        component=performance_component,
                        operation="api_call",
                        duration_ms=duration_ms,
                        metadata={"api": api_name}
                    )

                return result

            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Categorize the error
                category, message = ErrorHandler.categorize_error(e)

                # Record error in analytics
                AnalyticsTracker.record_error(
                    error_message=message,
                    error_category=category,
                    source=f"{func.__qualname__}",
                    metadata={
                        "api": api_name,
                        "duration_ms": duration_ms,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )

                # Record failed API call
                AnalyticsTracker.record_api_call(
                    api_name=api_name,
                    status="error",
                    duration_ms=duration_ms,
                    metadata={
                        "error_type": category.value,
                        "error_message": message
                    }
                )

                # Log the error
                logger.error(f"API call to {api_name} failed after {duration_ms:.2f}ms: {message}")

                # Re-raise the exception
                raise

        return wrapper

    return decorator


def initialize_error_directory() -> None:
    """Initialize the directory for error logs."""
    error_dir = os.path.join(os.path.dirname(__file__), "error_logs")
    os.makedirs(error_dir, exist_ok=True)
    return error_dir


def log_error_to_file(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """Log an error to a file with detailed information.

    Args:
        error: The exception to log
        context: Additional context information

    Returns:
        Path to the error log file
    """
    error_dir = initialize_error_directory()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"error_{timestamp}.log"
    filepath = os.path.join(error_dir, filename)

    # Categorize the error
    category, message = ErrorHandler.categorize_error(error)

    # Get stack trace
    import traceback
    stack_trace = traceback.format_exc()

    # Write to file
    with open(filepath, "w") as f:
        f.write(f"Error Log: {timestamp}\n")
        f.write(f"Category: {category.value}\n")
        f.write(f"Message: {message}\n\n")

        if context:
            f.write("Context:\n")
            for key, value in context.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("Stack Trace:\n")
        f.write(stack_trace)

    logger.info(f"Error details logged to {filepath}")
    return filepath


class GlobalErrorHandler:
    """Global error handler for the application."""

    @staticmethod
    def handle_exception(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Handle exceptions from a function call.

        Args:
            func: Function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function call or error response
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            error_filepath = log_error_to_file(
                error=e,
                context={
                    "function": func.__qualname__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Format standardized error response
            error_response = ErrorHandler.format_error_response(
                e,
                context={"error_log": error_filepath}
            )

            return error_response
