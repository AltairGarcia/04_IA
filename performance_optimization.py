"""
Performance Optimization module for LangGraph 101 project.

This module provides performance optimization features for the LangGraph project,
including caching, adaptive timeouts, and load management.
"""
import functools
import time
import logging
import threading
import queue
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, TypeVar, Tuple, Union, cast
import os
import json
import hashlib
from datetime import datetime, timedelta
import collections
try:
    import msgpack  # More efficient serialization
except ImportError:
    msgpack = None  # Fallback if not installed

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for function signatures
T = TypeVar('T')
R = TypeVar('R')

# Constants
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "performance_cache")
DEFAULT_CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds
MAX_QUEUE_SIZE = 100  # Maximum tasks in queue
DEFAULT_MAX_MEMORY_ITEMS = 1000  # Maximum number of items to store in memory

# Serialization formats
SERIALIZE_JSON = "json"
SERIALIZE_MSGPACK = "msgpack"
DEFAULT_SERIALIZATION = SERIALIZE_MSGPACK if msgpack else SERIALIZE_JSON


class PerformanceCache:
    """Caching system for expensive operations."""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        default_expiry: int = DEFAULT_CACHE_EXPIRY,
        max_memory_items: int = DEFAULT_MAX_MEMORY_ITEMS,
        serialization_format: str = DEFAULT_SERIALIZATION
    ):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache storage
            default_expiry: Default cache expiry time in seconds
            max_memory_items: Maximum number of items to store in memory (LRU policy)
            serialization_format: Format to use for serialization (json or msgpack)
        """
        self.cache_dir = cache_dir
        self.default_expiry = default_expiry
        self.max_memory_items = max_memory_items
        self.serialization_format = serialization_format

        # Use OrderedDict to implement LRU cache
        self.memory_cache = collections.OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "file_hits": 0,
            "memory_size": 0,
            "file_size": 0
        }

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Initialized performance cache at {cache_dir} with serialization: {serialization_format}")

        # Run initial cleanup of expired items
        self.cleanup_expired()

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments.

        Args:
            func_name: Function name
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Cache key string
        """
        # Convert arguments to a canonical string representation
        args_str = str(args) + str(sorted(kwargs.items()))

        # Generate a hash
        key = hashlib.md5(f"{func_name}:{args_str}".encode()).hexdigest()

        return key

    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key.

        Args:
            key: Cache key

        Returns:
            File path for the cache entry
        """
        return os.path.join(self.cache_dir, f"cache_{key}.json")

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data using the configured serialization format.

        Args:
            data: The data to serialize

        Returns:
            Serialized data as bytes
        """
        if self.serialization_format == SERIALIZE_MSGPACK and msgpack:
            try:
                return msgpack.packb(data, use_bin_type=True)
            except Exception as e:
                logger.warning(f"Error serializing with msgpack: {str(e)}, falling back to JSON")
                # Fall back to JSON on error

        # Default to JSON
        return json.dumps(data).encode('utf-8')

    def _deserialize_data(self, data_bytes: bytes) -> Any:
        """Deserialize data using the configured serialization format.

        Args:
            data_bytes: The serialized data as bytes

        Returns:
            Deserialized data
        """
        if self.serialization_format == SERIALIZE_MSGPACK and msgpack:
            try:
                return msgpack.unpackb(data_bytes, raw=False)
            except Exception as e:
                logger.warning(f"Error deserializing with msgpack: {str(e)}, falling back to JSON")
                # Fall back to JSON on error

        # Default to JSON
        return json.loads(data_bytes.decode('utf-8'))

    def get(self, func_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get an item from cache if it exists and is not expired.

        Args:
            func_name: Function name
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Cached result or None if not found or expired
        """
        key = self._generate_key(func_name, args, kwargs)

        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if time.time() < expiry:
                    logger.debug(f"Memory cache hit for {func_name}")
                    self.stats["memory_hits"] += 1
                    return value
                else:
                    # Expired, remove from memory cache
                    del self.memory_cache[key]

            # Check file cache
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        cache_data = json.load(f)

                    # Check expiry
                    if cache_data["expiry"] > time.time():
                        # Add to memory cache for faster access next time
                        self.memory_cache[key] = (cache_data["value"], cache_data["expiry"])
                        logger.debug(f"File cache hit for {func_name}")
                        self.stats["file_hits"] += 1
                        return cache_data["value"]
                    else:
                        # Expired, remove the file
                        os.remove(cache_path)
                        logger.debug(f"Expired cache entry removed for {func_name}")
                except Exception as e:
                    logger.warning(f"Error reading cache: {str(e)}")
                    # If there's an error, treat it as a cache miss

        # Cache miss
        self.stats["misses"] += 1
        return None

    def set(self, func_name: str, args: tuple, kwargs: dict, value: Any, expiry: Optional[int] = None) -> None:
        """Store an item in cache.

        Args:
            func_name: Function name
            args: Function positional arguments
            kwargs: Function keyword arguments
            value: Value to cache
            expiry: Expiry time in seconds (from now)
        """
        if expiry is None:
            expiry = self.default_expiry

        expiry_time = time.time() + expiry
        key = self._generate_key(func_name, args, kwargs)

        with self.lock:
            # Store in memory cache
            self.memory_cache[key] = (value, expiry_time)

            # Enforce memory limit (LRU eviction)
            self._enforce_memory_limit()

            # Store in file cache
            cache_path = self._get_cache_path(key)
            try:
                cache_data = {
                    "func_name": func_name,
                    "args_hash": hashlib.md5(str(args).encode()).hexdigest(),
                    "kwargs_hash": hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest(),
                    "value": value,
                    "expiry": expiry_time,
                    "created": time.time()
                }

                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f)

                logger.debug(f"Cached result for {func_name} (expires in {expiry}s)")
            except Exception as e:
                logger.warning(f"Error writing to cache: {str(e)}")

    def clear(self, func_name: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            func_name: If provided, only clear entries for this function

        Returns:
            Number of entries cleared
        """
        count = 0

        with self.lock:
            # Clear memory cache
            if func_name:
                # Clear specific function entries
                keys_to_remove = [
                    key for key, (_, _) in self.memory_cache.items()
                    if key.startswith(func_name)
                ]

                for key in keys_to_remove:
                    del self.memory_cache[key]
                    count += 1
            else:
                # Clear all entries
                count = len(self.memory_cache)
                self.memory_cache.clear()

            # Clear file cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith("cache_") and filename.endswith(".json"):
                        if func_name is None:
                            # If no function name specified, delete all
                            os.remove(os.path.join(self.cache_dir, filename))
                            count += 1
                        else:
                            # Check if file matches function name
                            try:
                                with open(os.path.join(self.cache_dir, filename), 'r') as f:
                                    cache_data = json.load(f)

                                if cache_data["func_name"] == func_name:
                                    os.remove(os.path.join(self.cache_dir, filename))
                                    count += 1
                            except:
                                # If there's an error reading the file, skip it
                                pass
            except Exception as e:
                logger.error(f"Error clearing file cache: {str(e)}")

        logger.info(f"Cleared {count} cache entries{' for ' + func_name if func_name else ''}")
        return count

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        current_time = time.time()

        with self.lock:
            # Clear expired memory cache entries
            keys_to_remove = [
                key for key, (_, expiry) in self.memory_cache.items()
                if expiry <= current_time
            ]

            for key in keys_to_remove:
                del self.memory_cache[key]
                count += 1

            # Clear expired file cache entries
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith("cache_") and filename.endswith(".json"):
                        file_path = os.path.join(self.cache_dir, filename)
                        try:
                            with open(file_path, 'r') as f:
                                cache_data = json.load(f)

                            if cache_data["expiry"] <= current_time:
                                os.remove(file_path)
                                count += 1
                        except:
                            # If there's an error reading the file, skip it
                            pass
            except Exception as e:
                logger.error(f"Error cleaning up expired cache: {str(e)}")

        logger.info(f"Removed {count} expired cache entries")
        return count

    def _update_lru_cache(self, key: str):
        """Update the LRU cache order.

        Args:
            key: The cache key to update
        """
        with self.lock:
            if key in self.memory_cache:
                # Move to the end (most recently used)
                value = self.memory_cache.pop(key)
                self.memory_cache[key] = value

    def _enforce_memory_limit(self):
        """Remove least recently used items when cache exceeds the size limit."""
        with self.lock:
            while len(self.memory_cache) > self.max_memory_items:
                # Remove the first item (least recently used)
                self.memory_cache.popitem(last=False)

    async def async_get(self, func_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Async version of the get method.

        Args:
            func_name: Function name
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Cached result or None if not found or expired
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get(func_name, args, kwargs))

    async def async_set(self, func_name: str, args: tuple, kwargs: dict, value: Any,
                      expiry: Optional[int] = None) -> None:
        """Async version of the set method.

        Args:
            func_name: Function name
            args: Function positional arguments
            kwargs: Function keyword arguments
            value: Value to cache
            expiry: Expiry time in seconds (from now)
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.set(func_name, args, kwargs, value, expiry)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            # Update memory size
            self.stats["memory_size"] = len(self.memory_cache)

            # Count file cache size
            try:
                self.stats["file_size"] = len([f for f in os.listdir(self.cache_dir)
                                            if f.startswith("cache_") and f.endswith(".json")])
            except Exception:
                self.stats["file_size"] = -1  # Error counting files

            return self.stats.copy()  # Return a copy to prevent modification



def cached(expiry: Optional[int] = None) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator to cache function results.

    Args:
        expiry: Cache expiry time in seconds

    Returns:
        Decorated function
    """
    # Get or create the cache
    cache = PerformanceCache()

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            func_name = func.__qualname__

            # Skip cache for certain argument combinations if needed
            skip_cache = kwargs.pop("skip_cache", False)
            if skip_cache:
                return func(*args, **kwargs)

            # Try to get from cache
            result = cache.get(func_name, args, kwargs)
            if result is not None:
                return result

            # Execute the function
            result = func(*args, **kwargs)

            # Store in cache (if result is cacheable)
            try:
                # Check if the result is JSON-serializable
                json.dumps(result)
                # Cache the result
                cache.set(func_name, args, kwargs, result, expiry)
            except (TypeError, OverflowError):
                # If the result can't be serialized, don't cache it
                logger.warning(f"Result of {func_name} is not cacheable (not JSON-serializable)")

            return result

        # Add a method to clear this function's cache
        def clear_cache() -> int:
            return cache.clear(func.__qualname__)

        wrapper.clear_cache = clear_cache  # type: ignore

        return wrapper

    return decorator


def async_cached(expiry: Optional[int] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to cache async function results.

    Args:
        expiry: Cache expiry time in seconds

    Returns:
        Decorated async function
    """
    # Get or create the cache
    cache = PerformanceCache()

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__qualname__

            # Skip cache for certain argument combinations if needed
            skip_cache = kwargs.pop("skip_cache", False)
            if skip_cache:
                return await func(*args, **kwargs)

            # Try to get from cache
            result = await cache.async_get(func_name, args, kwargs)
            if result is not None:
                return result

            # Execute the function
            result = await func(*args, **kwargs)

            # Store in cache (if result is cacheable)
            try:
                # Check if the result is JSON-serializable
                json.dumps(result)
                # Cache the result
                await cache.async_set(func_name, args, kwargs, result, expiry)
            except (TypeError, OverflowError):
                # If the result can't be serialized, don't cache it
                logger.warning(f"Result of {func_name} is not cacheable (not JSON-serializable)")

            return result

        # Add a method to clear this function's cache
        async def clear_cache() -> int:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: cache.clear(func.__qualname__)
            )

        wrapper.clear_cache = clear_cache

        return wrapper

    return decorator


class AdaptiveTimeout:
    """Adaptive timeout management for API calls."""

    def __init__(self):
        """Initialize the adaptive timeout manager."""
        self.timeouts: Dict[str, Dict[str, float]] = {}
        self.history: Dict[str, List[Tuple[float, bool]]] = {}  # (duration, success)
        self.lock = threading.RLock()

        # Default timeouts
        self.default_initial_timeout = 10.0  # 10 seconds
        self.default_min_timeout = 5.0  # 5 seconds
        self.default_max_timeout = 60.0  # 60 seconds

        # History window
        self.history_window = 20  # Consider last 20 calls

    def get_timeout(self, service_name: str, operation: str = "default") -> float:
        """Get the current timeout for a service.

        Args:
            service_name: Name of the service
            operation: Type of operation

        Returns:
            Timeout in seconds
        """
        with self.lock:
            if service_name not in self.timeouts:
                self.timeouts[service_name] = {"default": self.default_initial_timeout}

            if operation not in self.timeouts[service_name]:
                self.timeouts[service_name][operation] = self.timeouts[service_name]["default"]

            return self.timeouts[service_name][operation]

    def record_call(self, service_name: str, operation: str, duration: float, success: bool) -> None:
        """Record a call to the service.

        Args:
            service_name: Name of the service
            operation: Type of operation
            duration: Call duration in seconds
            success: Whether the call succeeded
        """
        key = f"{service_name}:{operation}"

        with self.lock:
            # Initialize history if needed
            if key not in self.history:
                self.history[key] = []

            # Add the call to history
            self.history[key].append((duration, success))

            # Keep only the most recent calls
            if len(self.history[key]) > self.history_window:
                self.history[key] = self.history[key][-self.history_window:]

            # Update the timeout based on history
            self._update_timeout(service_name, operation)

    def _update_timeout(self, service_name: str, operation: str) -> None:
        """Update the timeout based on call history.

        Args:
            service_name: Name of the service
            operation: Type of operation
        """
        key = f"{service_name}:{operation}"

        if key not in self.history or not self.history[key]:
            return

        # Calculate success rate and average duration
        history = self.history[key]
        success_count = sum(1 for _, success in history if success)
        success_rate = success_count / len(history)

        # Calculate average duration of successful calls
        successful_durations = [d for d, s in history if s]
        avg_duration = sum(successful_durations) / len(successful_durations) if successful_durations else self.default_initial_timeout

        # Calculate 95th percentile duration
        if successful_durations:
            successful_durations.sort()
            p95_index = int(len(successful_durations) * 0.95)
            p95_duration = successful_durations[p95_index] if p95_index < len(successful_durations) else successful_durations[-1]
        else:
            p95_duration = self.default_initial_timeout

        # Adjust timeout based on success rate and duration
        current_timeout = self.get_timeout(service_name, operation)

        if success_rate < 0.7:
            # Low success rate, increase timeout significantly
            new_timeout = current_timeout * 1.5
        elif success_rate < 0.9:
            # Moderate success rate, increase timeout slightly
            new_timeout = current_timeout * 1.2
        elif len(history) >= 5:
            # Good success rate with enough history, align timeout with actual performance
            new_timeout = p95_duration * 1.5  # 50% buffer over 95th percentile
        else:
            # Not enough history or very good success rate, keep current timeout
            new_timeout = current_timeout

        # Ensure the timeout is within min/max bounds
        new_timeout = max(self.default_min_timeout, min(new_timeout, self.default_max_timeout))

        # Update the timeout
        if service_name not in self.timeouts:
            self.timeouts[service_name] = {}

        self.timeouts[service_name][operation] = new_timeout

        logger.debug(f"Updated timeout for {service_name}:{operation} to {new_timeout:.2f}s " +
                    f"(success rate: {success_rate:.2f}, avg duration: {avg_duration:.2f}s)")


class TaskQueue:
    """Task queue for managing concurrent operations."""

    def __init__(self, max_workers: int = 4, max_queue_size: int = MAX_QUEUE_SIZE):
        """Initialize the task queue.

        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum size of the task queue
        """
        self._max_workers = max_workers
        self._running = False
        self._tasks = queue.Queue(max_queue_size)
        self._workers = []
        self._executor = None
        self._lock = threading.RLock()

        logger.info(f"Initialized task queue with {max_workers} workers")

    def start(self) -> None:
        """Start processing tasks."""
        with self._lock:
            if self._running:
                logger.warning("Task queue is already running")
                return

            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
            self._running = True

            # Start worker threads
            for i in range(self._max_workers):
                worker = threading.Thread(target=self._worker_loop, daemon=True)
                worker.start()
                self._workers.append(worker)

            logger.info(f"Started {self._max_workers} worker threads")

    def stop(self) -> None:
        """Stop processing tasks."""
        with self._lock:
            if not self._running:
                logger.warning("Task queue is not running")
                return

            self._running = False

            # Add sentinel values to stop workers
            for _ in range(self._max_workers):
                self._tasks.put(None)

            # Wait for workers to finish
            for worker in self._workers:
                worker.join(timeout=2.0)

            # Clear worker list
            self._workers = []

            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None

            logger.info("Stopped task queue")

    def add_task(self, task: Callable, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Add a task to the queue.

        Args:
            task: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            A Future object representing the task result.
        """
        if not self._running or not self._executor:
            logger.warning("Cannot add task: Task queue is not running")
            raise RuntimeError("Task queue is not running")

        # For async functions, wrap them to ensure they're properly awaited
        if asyncio.iscoroutinefunction(task):
            def wrapper(*args, **kwargs):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(task(*args, **kwargs))
                finally:
                    loop.close()
            return self._executor.submit(wrapper, *args, **kwargs)
        else:
            # For regular functions, submit them directly
            return self._executor.submit(task, *args, **kwargs)

    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while self._running:
            try:
                # Get a task from the queue
                task_item = self._tasks.get(block=True, timeout=1.0)

                if task_item is None:
                    # Sentinel value, exit the loop
                    break

                task, args, kwargs = task_item

                try:
                    # Execute the task
                    task(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error executing task: {str(e)}")
                finally:
                    # Mark task as done
                    self._tasks.task_done()

            except queue.Empty:
                # Timeout on queue get, continue the loop
                pass
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                # Short sleep to avoid tight loop on persistent errors
                time.sleep(0.1)

# Singleton instance
_task_queue = None

def get_task_queue(max_workers: int = 4) -> TaskQueue:
    """Get the singleton instance of TaskQueue."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(max_workers=max_workers)
    return _task_queue


# Create singleton instances
_cache_instance = None
_timeout_manager = None

def get_cache() -> PerformanceCache:
    """Get the singleton cache instance.

    Returns:
        PerformanceCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PerformanceCache()
    return _cache_instance


def get_timeout_manager() -> AdaptiveTimeout:
    """Get the singleton timeout manager instance.

    Returns:
        AdaptiveTimeout instance
    """
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = AdaptiveTimeout()
    return _timeout_manager


def adaptive_timeout(service_name: str, operation: str = "default") -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator to apply adaptive timeouts to function calls.

    Args:
        service_name: Name of the service
        operation: Type of operation

    Returns:
        Decorated function
    """
    timeout_manager = get_timeout_manager()

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Get the current timeout
            timeout = timeout_manager.get_timeout(service_name, operation)

            # Start timing
            start_time = time.time()
            success = False

            try:
                # Add timeout to kwargs if not already present
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = timeout

                # Execute the function
                result = func(*args, **kwargs)
                success = True
                return result
            finally:
                # Record the call duration and success
                duration = time.time() - start_time
                timeout_manager.record_call(service_name, operation, duration, success)

        return wrapper

    return decorator


# Cleanup and maintenance functions
def run_maintenance_tasks() -> None:
    """Run maintenance tasks for the performance optimization system."""
    # Clean up expired cache entries
    cache = get_cache()
    count = cache.cleanup_expired()
    logger.info(f"Maintenance: Removed {count} expired cache entries")


# Initialize the performance optimization systems
def initialize_performance_optimization() -> None:
    """Initialize the performance optimization systems."""
    # Create the cache directory
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

    # Initialize the cache
    get_cache()

    # Initialize the timeout manager
    get_timeout_manager()

    # Initialize and start the task queue
    task_queue = get_task_queue()
    task_queue.start()

    # Schedule regular maintenance tasks
    def scheduled_maintenance() -> None:
        while True:
            try:
                run_maintenance_tasks()
            except Exception as e:
                logger.error(f"Error during maintenance: {str(e)}")

            # Run maintenance daily
            time.sleep(24 * 60 * 60)

    # Start the maintenance thread
    maintenance_thread = threading.Thread(target=scheduled_maintenance, daemon=True)
    maintenance_thread.start()

    logger.info("Performance optimization systems initialized")


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)

    # Initialize the performance optimization systems
    initialize_performance_optimization()

    # Test the cache
    @cached(expiry=60)
    def example_function(a: int, b: int) -> int:
        print(f"Computing {a} + {b}")
        return a + b

    # First call should compute the result
    result1 = example_function(5, 7)
    print(f"Result 1: {result1}")

    # Second call should use cached result
    result2 = example_function(5, 7)
    print(f"Result 2: {result2}")

    # Different arguments should compute a new result
    result3 = example_function(10, 20)
    print(f"Result 3: {result3}")

    # Test the adaptive timeout
    @adaptive_timeout("example-service", "add")
    def example_api_call(a: int, b: int, timeout: float = 10.0) -> int:
        print(f"API call with timeout {timeout}s")
        time.sleep(0.2)  # Simulate API call
        return a + b

    # Make some API calls
    for _ in range(5):
        result = example_api_call(1, 2)
        print(f"API result: {result}")

    # Show the adaptive timeout
    timeout_manager = get_timeout_manager()
    print(f"Adapted timeout: {timeout_manager.get_timeout('example-service', 'add')}s")

    print("Performance optimization test complete")
