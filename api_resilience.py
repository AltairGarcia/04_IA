"""
API resilience module for wrapping external API calls with circuit breakers and retry logic.
"""
import time
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from functools import wraps

# Import resilience components
from resilient_operations import CircuitBreaker, with_retry, FallbackHandler

# Get logger
from logging_config import get_contextual_logger
logger = get_contextual_logger(__name__, module="api_resilience", component_type="external_api")

# Type for generic function return
T = TypeVar('T')

# Circuit breakers for different APIs
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(api_name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for the specified API.

    Args:
        api_name: Name of the API (e.g., "gemini", "openai", "elevenlabs")

    Returns:
        CircuitBreaker instance for the API
    """
    if api_name not in _circuit_breakers:
        _circuit_breakers[api_name] = CircuitBreaker(
            failure_threshold=5,  # Open after 5 consecutive errors
            reset_timeout=60.0,   # Wait 60s before attempting half-open
            half_open_timeout=30.0  # Test one request every 30s in half-open
        )
    return _circuit_breakers[api_name]


def with_circuit_breaker(
    api_name: str,
    fallback_value: Optional[Any] = None,
    fallback_func: Optional[Callable[..., Any]] = None
) -> Callable:
    """Decorator to apply circuit breaker pattern to API calls.

    Args:
        api_name: Name of the API being called
        fallback_value: Value to return if circuit is open
        fallback_func: Function to call if circuit is open

    Returns:
        Decorated function with circuit breaker logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            circuit = get_circuit_breaker(api_name)

            if not circuit.allow_request():
                logger.warning(f"Circuit breaker open for {api_name}, request rejected")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return cast(T, fallback_value)

            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure()
                logger.error(f"API call to {api_name} failed: {str(e)}")
                raise

        return wrapper
    return decorator


def resilient_api_call(
    api_name: str,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    fallback_value: Optional[Any] = None,
    fallback_func: Optional[Callable[..., Any]] = None
) -> Callable:
    """Combined decorator for API resilience - applies circuit breaker and retry.

    Args:
        api_name: Name of the API being called
        max_retries: Maximum number of retries for transient failures
        initial_backoff: Initial backoff time in seconds
        fallback_value: Value to return if all attempts fail
        fallback_func: Function to call if all attempts fail

    Returns:
        Function decorated with circuit breaker and retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply circuit breaker first (outermost)
        circuit_protected = with_circuit_breaker(
            api_name=api_name,
            fallback_value=fallback_value,
            fallback_func=fallback_func
        )(func)

        # Then apply retry logic (innermost, only if circuit allows)
        retry_protected = with_retry(
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            jitter=True,
            on_retry_callback=lambda e, a, b: logger.info(
                f"Retrying {api_name} call after error: {str(e)}"
            )
        )(circuit_protected)

        return retry_protected

    return decorator
