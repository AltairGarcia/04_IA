"""
Resilient operations module providing retry mechanisms and circuit breakers.
"""
import time
import logging
import functools
import random
from typing import Callable, Any, Dict, Optional, TypeVar, cast

# Type variable for generic function return type
T = TypeVar('T')

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker implementation to prevent repeated failed calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting reset after opening
            half_open_timeout: Time in seconds to wait in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout

        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = 0.0
        self.last_success_time = 0.0

    def record_success(self) -> None:
        """Record a successful operation."""
        self.failures = 0
        self.last_success_time = time.time()
        if self.state == "HALF-OPEN":
            logger.info("Circuit breaker reset to CLOSED after successful call")
            self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == "CLOSED" and self.failures >= self.failure_threshold:
            logger.warning(f"Circuit OPEN after {self.failures} failures")
            self.state = "OPEN"
        elif self.state == "HALF-OPEN":
            logger.warning("Circuit reopened after test call failure")
            self.state = "OPEN"

    def allow_request(self) -> bool:
        """Check if a request should be allowed based on circuit state."""
        now = time.time()

        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if now - self.last_failure_time > self.reset_timeout:
                logger.info("Circuit state changed to HALF-OPEN for testing")
                self.state = "HALF-OPEN"
                return True
            return False
        elif self.state == "HALF-OPEN":
            if now - self.last_success_time < self.half_open_timeout:
                return False
            return True
        return True


def with_retry(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry_callback: Optional[Callable[[Exception, int, float], None]] = None
) -> Callable:
    """Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_factor: Multiplier for backoff after each retry
        jitter: Whether to add randomness to backoff
        retryable_exceptions: Exceptions that should trigger retry
        on_retry_callback: Optional callback to call on each retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    if attempt > 0:
                        # Calculate backoff time
                        backoff = min(
                            initial_backoff * (backoff_factor ** (attempt - 1)),
                            max_backoff
                        )

                        # Add jitter if enabled (±25%)
                        if jitter:
                            backoff = backoff * (1 + random.uniform(-0.25, 0.25))

                        logger.warning(
                            f"Retry attempt {attempt}/{max_retries} after {backoff:.2f}s"
                        )

                        # Call retry callback if provided
                        if on_retry_callback and last_exception:
                            on_retry_callback(last_exception, attempt, backoff)

                        time.sleep(backoff)

                    # Call the original function
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}"
                    )

                    # If this was the last attempt, re-raise
                    if attempt == max_retries:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise

            # This should never be reached due to the re-raise above
            raise RuntimeError("Unexpected error in retry logic")

        return wrapper

    return decorator


class FallbackHandler:
    """Handler for providing fallback mechanisms when operations fail."""

    @staticmethod
    def with_fallback(fallback_func: Callable[..., T], *fallback_args: Any, **fallback_kwargs: Any) -> Callable:
        """Decorator to provide a fallback when a function fails.

        Args:
            fallback_func: Function to call if the primary function fails
            *fallback_args: Arguments to pass to the fallback function
            **fallback_kwargs: Keyword arguments to pass to the fallback function

        Returns:
            Decorated function with fallback logic
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Function {func.__name__} failed, using fallback: {str(e)}")
                    return fallback_func(*fallback_args, **fallback_kwargs)

            return wrapper

        return decorator
