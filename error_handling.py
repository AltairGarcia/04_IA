"""
Error Handling module for LangGraph 101 project.

This module provides centralized error handling and resilience capabilities
for the LangGraph project, including retry mechanisms, fallbacks, and
standardized error reporting.
"""
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import requests
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
R = TypeVar('R')

# Custom Application Exceptions
class AppBaseException(Exception):
    """Base class for custom application exceptions."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class ConfigurationError(AppBaseException):
    """Exception raised for errors in configuration."""
    pass

class ApiKeyError(AppBaseException): # Could also inherit from a more general AuthenticationError
    """Exception raised for API key related errors."""
    pass

class ResourceNotFoundError(AppBaseException):
    """Exception raised when a resource is not found."""
    pass

class InvalidInputError(AppBaseException):
    """Exception raised for invalid input data."""
    pass

class NetworkConnectivityError(AppBaseException):
    """Exception raised for network connectivity issues not covered by requests.RequestException."""
    pass

class ServiceUnavailableError(AppBaseException):
    """Exception raised when an external service is unavailable."""
    pass


class ErrorCategory(Enum):
    """Categories of errors for better handling and reporting."""
    CLIENT_API_ERROR = "client_api_error" # 4xx errors (excluding auth, validation, resource not found)
    SERVER_API_ERROR = "server_api_error" # 5xx errors
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_NOT_FOUND_ERROR = "resource_not_found_error" # More specific than RESOURCE_ERROR
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    CONFIGURATION_ERROR = "configuration_error" # New category
    INTERNAL_ERROR = "internal_error" # For unexpected errors within the application itself
    SERVICE_UNAVAILABLE_ERROR = "service_unavailable_error" # For external service issues
    UNKNOWN_ERROR = "unknown_error"
    # Deprecating RESOURCE_ERROR in favor of RESOURCE_NOT_FOUND_ERROR or more specific ones
    # Deprecating API_ERROR in favor of CLIENT_API_ERROR and SERVER_API_ERROR


class ErrorHandler:
    """Central error handler for standardized error management."""

    @staticmethod
    def categorize_error(error: Exception) -> Tuple[ErrorCategory, str]:
        """Categorize an exception into a standard error category.

        Args:
            error: The exception to categorize

        Returns:
            Tuple of (ErrorCategory, error_message)
        """
        error_message = str(error)

        # Prioritize custom application exceptions
        if isinstance(error, ApiKeyError):
            return ErrorCategory.AUTHENTICATION_ERROR, f"API key error: {error_message}"
        elif isinstance(error, ResourceNotFoundError):
            return ErrorCategory.RESOURCE_NOT_FOUND_ERROR, f"Resource not found: {error_message}"
        elif isinstance(error, InvalidInputError):
            return ErrorCategory.VALIDATION_ERROR, f"Invalid input: {error_message}"
        elif isinstance(error, ConfigurationError):
            return ErrorCategory.CONFIGURATION_ERROR, f"Configuration error: {error_message}"
        elif isinstance(error, NetworkConnectivityError):
            return ErrorCategory.NETWORK_ERROR, f"Network connectivity issue: {error_message}"
        elif isinstance(error, ServiceUnavailableError):
            return ErrorCategory.SERVICE_UNAVAILABLE_ERROR, f"Service unavailable: {error_message}"
        # Keep AppBaseException check general for other custom errors that might not have specific categories yet        elif isinstance(error, AppBaseException):
            # Try to find a more specific category based on message if possible, or default to internal error
            if "memory" in error_message.lower(): # Example if some AppBaseException might relate to memory
                return ErrorCategory.MEMORY_ERROR, f"Memory related error: {error_message}"
            return ErrorCategory.INTERNAL_ERROR, f"Application error: {error_message}"

        # Handle requests library exceptions
        elif isinstance(error, requests.RequestException):
            if isinstance(error, requests.ConnectionError):
                return ErrorCategory.NETWORK_ERROR, f"Network connection error: {error_message}"
            elif isinstance(error, requests.Timeout):
                return ErrorCategory.TIMEOUT_ERROR, f"Request timeout error: {error_message}"
            elif isinstance(error, requests.HTTPError):
                # Try to get status code, fallback to message if not available
                status_code = getattr(getattr(error, 'response', None), 'status_code', None)
                if status_code is None:
                    # Fallback: guess from message
                    if "401" in error_message or "403" in error_message:
                        return ErrorCategory.AUTHENTICATION_ERROR, f"Authentication error: {error_message}"
                    elif "404" in error_message:
                        return ErrorCategory.RESOURCE_NOT_FOUND_ERROR, f"Resource not found: {error_message}"
                    elif "500" in error_message or "server error" in error_message.lower():
                        return ErrorCategory.SERVER_API_ERROR, f"Server API error: {error_message}"
                    elif "400" in error_message or "422" in error_message:
                        return ErrorCategory.VALIDATION_ERROR, f"Validation/Client error: {error_message}"
                    elif "4" in error_message and "error" in error_message.lower():
                        return ErrorCategory.CLIENT_API_ERROR, f"Client API error: {error_message}"
                    else:
                        return ErrorCategory.UNKNOWN_ERROR, f"Unknown HTTP error: {error_message}"
                else:
                    if status_code in (401, 403):
                        return ErrorCategory.AUTHENTICATION_ERROR, f"Authentication error (HTTP {status_code}): {error_message}"
                    elif status_code == 404:
                        return ErrorCategory.RESOURCE_NOT_FOUND_ERROR, f"Resource not found (HTTP 404): {error_message}"
                    elif status_code in (400, 422) or (status_code >= 400 and status_code < 500 and "validation" in error_message.lower()):
                        return ErrorCategory.VALIDATION_ERROR, f"Validation/Client error (HTTP {status_code}): {error_message}"
                    elif status_code >= 500:
                        return ErrorCategory.SERVER_API_ERROR, f"Server API error (HTTP {status_code}): {error_message}"
                    elif status_code >= 400 and status_code < 500:
                        return ErrorCategory.CLIENT_API_ERROR, f"Client API error (HTTP {status_code}): {error_message}"
                    else:
                        return ErrorCategory.UNKNOWN_ERROR, f"Unknown HTTP error: {error_message}"
            elif "invalid api key" in error_message.lower() or "api key error" in error_message.lower():
                return ErrorCategory.AUTHENTICATION_ERROR, f"API key error: {error_message}"
            else: # Other requests.RequestException like TooManyRedirects
                return ErrorCategory.NETWORK_ERROR, f"Network request error: {error_message}"        # General Python exceptions (less specific, attempt to categorize based on type or message)
        elif isinstance(error, MemoryError): # Direct check for MemoryError
            return ErrorCategory.MEMORY_ERROR, f"Memory error: {error_message}"
        elif isinstance(error, TimeoutError): # Python's built-in TimeoutError
            return ErrorCategory.TIMEOUT_ERROR, f"Timeout error: {error_message}"
        elif isinstance(error, ConnectionError): # Python's built-in ConnectionError 
            return ErrorCategory.NETWORK_ERROR, f"Connection error: {error_message}"
        elif isinstance(error, ValueError): # Broader, could be refined if common patterns emerge
            return ErrorCategory.VALIDATION_ERROR, f"Value error (potential validation issue): {error_message}"
        elif isinstance(error, TypeError):
            return ErrorCategory.INTERNAL_ERROR, f"Type error (potential internal logic issue): {error_message}"

        # Fallback string checks (use sparingly, prefer typed exceptions)
        elif "timeout" in error_message.lower(): # General timeout string check
            return ErrorCategory.TIMEOUT_ERROR, f"Timeout detected: {error_message}"
        elif "memory" in error_message.lower() or "capacity" in error_message.lower():
            return ErrorCategory.MEMORY_ERROR, f"Resource (memory/capacity) error: {error_message}"
        elif "validation" in error_message.lower() or "invalid" in error_message.lower():
            return ErrorCategory.VALIDATION_ERROR, f"Validation issue: {error_message}"
        # Final fallback for any Exception
        logger.error(f"Unknown error encountered: {error_message}", exc_info=True)
        return ErrorCategory.UNKNOWN_ERROR, f"Server error: {error_message}" if "server error" in error_message.lower() else f"Unknown error: {error_message}"

    @staticmethod
    def format_error_response(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format a standardized error response.

        Args:
            error: The exception that occurred
            context: Optional context about the error (passed directly to this function)

        Returns:
            Dictionary with error details
        """
        category, message = ErrorHandler.categorize_error(error)

        # Initialize final_context with the context passed as an argument to this function
        final_context = context.copy() if context else {}

        # If the error is an AppBaseException and has its own context, merge it.
        # This strategy gives precedence to keys in the exception's context if there's an overlap.
        if isinstance(error, AppBaseException) and error.context:
            for key, value in error.context.items():
                final_context[key] = value # Exception's context can override argument's context

        error_details = {
            "success": False,
            "error": message,
            "error_type": category.value,
            "timestamp": time.time(),
        }

        if final_context: # Use the merged context
            error_details["context"] = final_context

        # Log the error with stack trace for debugging, using the final merged context
        log_payload = {
            "event_type": "application_error", # A specific field to identify this type of log
            "error_message": message,
            # final_context will be included if it's not empty
        }
        if final_context:
            log_payload["context"] = final_context

        # The JsonFormatter will automatically pick up exc_info for the stack trace
        # We pass our structured payload via the 'extra' parameter
        # The main message for the log record can be a summary.
        logger.error(
            f"Formatted application error: {category.value}",
            exc_info=error, # Pass the original exception for traceback
            extra=log_payload # Pass our custom fields
        )

        return error_details

    @staticmethod
    def with_retry(max_retries: int = 3,
                   delay: float = 1.0,
                   backoff_factor: float = 2.0,
                   retry_errors: List[ErrorCategory] = None,
                   retry_exceptions: List[type] = None) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """Decorator for retrying operations that may fail.
        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff_factor: Multiplier applied to delay between retries
            retry_errors: List of error categories to retry on
            retry_exceptions: List of exception types to retry on (optional)
        Returns:
            Decorator function
        """
        if retry_errors is None:
            retry_errors = [
                ErrorCategory.NETWORK_ERROR,
                ErrorCategory.TIMEOUT_ERROR,
                ErrorCategory.SERVER_API_ERROR
            ]
        if retry_exceptions is None:
            retry_exceptions = []
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> R:
                retries = 0
                current_delay = delay
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        # If retry_exceptions is set, only retry for those exception types
                        if retry_exceptions and not any(isinstance(e, exc) for exc in retry_exceptions):
                            raise
                        category, _ = ErrorHandler.categorize_error(e)
                        retries += 1
                        if retries > max_retries or (category not in retry_errors and not retry_exceptions):
                            raise
                        logger.warning(
                            f"Retry {retries}/{max_retries} for {func.__name__} after {current_delay}s due to {category.value}: {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            return wrapper
        return decorator

    @staticmethod
    def with_fallback(fallback_function: Callable[..., R]) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """Decorator for providing a fallback when an operation fails.

        Args:
            fallback_function: Function to call if the main function fails

        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> R:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Function {func.__name__} failed, using fallback. Error: {str(e)}")
                    return fallback_function(*args, **kwargs)

            return wrapper

        return decorator


# Utility functions for common error handling patterns

def safe_request(url: str, method: str = "get", timeout: float = 10.0, **kwargs: Any) -> requests.Response:
    """Make a request with standardized error handling and timeout.

    Args:
        url: URL to request
        method: HTTP method (get, post, etc.)
        timeout: Request timeout in seconds. Defaults to 10.0.
        **kwargs: Additional arguments for requests

    Returns:
        Response object

    Raises:
        requests.RequestException: If the request fails after retries
    """
    @ErrorHandler.with_retry(max_retries=3, delay=1.0) # Retries include TIMEOUT_ERROR by default
    def _make_request() -> requests.Response:
        # Prioritize timeout from kwargs if provided, otherwise use the default from the function signature
        request_kwargs = kwargs.copy()
        if 'timeout' not in request_kwargs:
            request_kwargs['timeout'] = timeout

        response = getattr(requests, method.lower())(url, **request_kwargs)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        return response

    return _make_request()


def graceful_degradation(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Decorator for graceful degradation of services.

    Args:
        func: Function that returns a dictionary result

    Returns:
        Wrapper function that handles errors
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Create error response
            error_response = ErrorHandler.format_error_response(
                e, context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
            )

            # Include function-specific information if available
            if hasattr(func, "__qualname__"):
                parts = func.__qualname__.split('.')
                if len(parts) > 1 and hasattr(args[0], parts[0]):
                    error_response["service"] = parts[0]

            return error_response

    return wrapper
