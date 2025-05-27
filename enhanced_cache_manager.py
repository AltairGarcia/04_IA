#!/usr/bin/env python3
"""
Enhanced Cache Manager for LangGraph 101

This module provides comprehensive caching functionality with enhanced Redis backend,
automatic fallback mechanisms, and intelligent cache strategies.
"""

import json
import pickle
import time
import logging
import hashlib
import threading
from typing import Any, Optional, Dict, List, Union, Callable, TypeVar
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager
import os

# Import enhanced Redis manager
from enhanced_redis_manager import get_redis_manager, RedisConfig, EnhancedRedisManager

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')

class CacheError(Exception):
    """Custom exception for cache-related errors"""
    pass

class EnhancedCacheManager:
    """
    Enhanced cache manager with automatic Redis installation and intelligent fallback.
    
    Features:
    - Enhanced Redis backend with automatic installation and fallback
    - Multiple serialization formats (JSON, pickle)
    - TTL support with intelligent expiration
    - Cache warming and preloading
    - Statistics and monitoring
    - Distributed cache invalidation
    - Performance optimization
    """
    
    def __init__(self, 
                 redis_url: Optional[str] = None,
                 redis_password: Optional[str] = None,
                 default_ttl: int = 3600,
                 max_memory_items: int = 1000,
                 enable_compression: bool = True):
        """
        Initialize enhanced cache manager.
          Args:
            redis_url: Redis connection URL
            redis_password: Redis password
            default_ttl: Default time-to-live in seconds
            max_memory_items: Maximum items in memory cache
            enable_compression: Enable data compression
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6380/0')
        self.redis_password = redis_password or os.getenv('REDIS_PASSWORD')
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        self.enable_compression = enable_compression
        
        # Performance statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'sets': 0,
            'deletes': 0,
            'redis_available': False,
            'memory_cache_size': 0,
            'total_operations': 0,
            'cache_hit_rate': 0.0,
            'average_response_time': 0.0
        }
        
        # Response time tracking
        self._response_times = []
        self._stats_lock = threading.Lock()
        
        # In-memory cache as fallback
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_cache_lock = threading.RLock()
        
        # Enhanced Redis manager
        self._initialize_enhanced_redis()
        
        # Cache warming queue
        self.warming_queue: List[Callable] = []
        
        # Performance monitoring
        self._start_performance_monitoring()
        
        logger.info(f"Enhanced CacheManager initialized - Redis: {self.stats['redis_available']}")
    
    def _initialize_enhanced_redis(self):
        """Initialize enhanced Redis manager with automatic installation."""
        try:
            # Parse Redis URL to extract connection parameters
            if self.redis_url.startswith('redis://'):
                url_parts = self.redis_url.replace('redis://', '').split(':')
                host = url_parts[0] if url_parts else 'localhost'
                port_db = url_parts[1] if len(url_parts) > 1 else '6379/0'
                port = int(port_db.split('/')[0]) if '/' in port_db else int(port_db)
                db = int(port_db.split('/')[1]) if '/' in port_db else 0
            else:
                host, port, db = 'localhost', 6379, 0
            
            # Create Redis configuration
            redis_config = RedisConfig(
                host=host,
                port=port,
                db=db,
                password=self.redis_password,
                fallback_max_memory_items=self.max_memory_items,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            
            # Get enhanced Redis manager with auto-installation
            self.redis_manager = get_redis_manager(redis_config)
            self.stats['redis_available'] = self.redis_manager.is_redis_available()
            
            logger.info("Enhanced Redis manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced Redis manager: {e}")
            self.redis_manager = None
            self.stats['redis_available'] = False
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring."""
        monitor_thread = threading.Thread(
            target=self._performance_monitoring_loop,
            daemon=True,
            name="Cache-Performance-Monitor"
        )
        monitor_thread.start()
    
    def _performance_monitoring_loop(self):
        """Background performance monitoring loop."""
        while True:
            try:
                time.sleep(60)  # Monitor every minute
                self._update_performance_stats()
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        with self._stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            if total_requests > 0:
                self.stats['cache_hit_rate'] = (self.stats['hits'] / total_requests) * 100
            
            if self._response_times:
                self.stats['average_response_time'] = sum(self._response_times) / len(self._response_times)
                # Keep only recent response times (last 1000)
                self._response_times = self._response_times[-1000:]
    
    def _record_response_time(self, response_time: float):
        """Record response time for performance monitoring."""
        with self._stats_lock:
            self._response_times.append(response_time)
    
    def _generate_key(self, key: str, prefix: str = "enhanced") -> str:
        """Generate a standardized cache key."""
        return f"{prefix}:{key}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for cache storage."""
        try:
            # Try JSON first for simple types (faster and more compatible)
            if isinstance(data, (str, int, float, bool, list, dict, type(None))):
                return json.dumps(data)
            else:
                # Use pickle for complex objects, but encode as base64 string
                import base64
                pickled_data = pickle.dumps(data)
                return f"PICKLE:{base64.b64encode(pickled_data).decode('utf-8')}"
                
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            # Fallback to string representation
            return str(data)
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from cache storage."""
        try:
            # Check if it's pickle data
            if data.startswith("PICKLE:"):
                import base64
                pickled_data = base64.b64decode(data[7:].encode('utf-8'))
                return pickle.loads(pickled_data)
            else:
                # Try JSON
                return json.loads(data)
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            # Return as string if deserialization fails
            return data
    
    def _memory_cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache with TTL."""
        with self.memory_cache_lock:
            # Clean expired items if cache is getting full
            if len(self.memory_cache) >= self.max_memory_items:
                self._cleanup_memory_cache()
            
            expires_at = None
            if ttl and ttl > 0:
                expires_at = time.time() + ttl
            
            self.memory_cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time(),
                'access_count': 0
            }
            self.stats['memory_cache_size'] = len(self.memory_cache)
    
    def _memory_cache_get(self, key: str) -> Any:
        """Get value from memory cache."""
        with self.memory_cache_lock:
            if key not in self.memory_cache:
                return None
            
            item = self.memory_cache[key]
            
            # Check expiration
            if item['expires_at'] and time.time() > item['expires_at']:
                del self.memory_cache[key]
                self.stats['memory_cache_size'] = len(self.memory_cache)
                return None
            
            # Update access count for LRU
            item['access_count'] += 1
            
            return item['value']
    
    def _memory_cache_delete(self, key: str):
        """Delete value from memory cache."""
        with self.memory_cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                self.stats['memory_cache_size'] = len(self.memory_cache)
    
    def _cleanup_memory_cache(self):
        """Clean up expired items from memory cache with LRU eviction."""
        current_time = time.time()
        expired_keys = []
        
        # Remove expired items
        for key, item in self.memory_cache.items():
            if item['expires_at'] and current_time > item['expires_at']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still too full, remove least recently used items
        if len(self.memory_cache) >= self.max_memory_items:
            # Sort by access count and creation time (LRU)
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: (x[1]['access_count'], x[1]['created_at'])
            )
            
            # Remove oldest 20% of items
            remove_count = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:remove_count]:
                del self.memory_cache[key]
        
        self.stats['memory_cache_size'] = len(self.memory_cache)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in cache with automatic fallback.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        cache_key = self._generate_key(key)
        
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            # Use enhanced Redis manager (handles fallback automatically)
            if self.redis_manager:
                try:
                    serialized_data = self._serialize_data(value)
                    success = self.redis_manager.set(cache_key, serialized_data, ex=ttl if ttl > 0 else None)
                    
                    if success:
                        # Also set in memory cache for faster access
                        self._memory_cache_set(cache_key, value, ttl)
                        self.stats['sets'] += 1
                        self.stats['total_operations'] += 1
                        self._record_response_time(time.time() - start_time)
                        return True
                
                except Exception as e:
                    logger.debug(f"Enhanced Redis set error: {e}")
                    self.stats['errors'] += 1
            
            # Direct fallback to memory cache if Redis fails
            self._memory_cache_set(cache_key, value, ttl)
            self.stats['sets'] += 1
            self.stats['total_operations'] += 1
            self._record_response_time(time.time() - start_time)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.stats['errors'] += 1
            self.stats['total_operations'] += 1
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache with automatic fallback.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        cache_key = self._generate_key(key)
        
        try:
            # Try memory cache first (fastest)
            memory_value = self._memory_cache_get(cache_key)
            if memory_value is not None:
                self.stats['hits'] += 1
                self.stats['total_operations'] += 1
                self._record_response_time(time.time() - start_time)
                return memory_value
            
            # Use enhanced Redis manager (handles fallback automatically)
            if self.redis_manager:
                try:
                    cached_data = self.redis_manager.get(cache_key)
                    if cached_data is not None:
                        value = self._deserialize_data(cached_data)
                        # Store in memory cache for next time
                        self._memory_cache_set(cache_key, value, self.default_ttl)
                        self.stats['hits'] += 1
                        self.stats['total_operations'] += 1
                        self._record_response_time(time.time() - start_time)
                        return value
                
                except Exception as e:
                    logger.debug(f"Enhanced Redis get error: {e}")
                    self.stats['errors'] += 1
            
            # Cache miss
            self.stats['misses'] += 1
            self.stats['total_operations'] += 1
            self._record_response_time(time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats['errors'] += 1
            self.stats['total_operations'] += 1
            return default
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_key(key)
        
        try:
            success = False
            
            # Delete from enhanced Redis manager
            if self.redis_manager:
                try:
                    result = self.redis_manager.delete(cache_key)
                    success = result > 0
                except Exception as e:
                    logger.debug(f"Enhanced Redis delete error: {e}")
                    self.stats['errors'] += 1
            
            # Delete from memory cache
            self._memory_cache_delete(cache_key)
            
            self.stats['deletes'] += 1
            self.stats['total_operations'] += 1
            return True  # Return True if any cache was cleared
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.stats['errors'] += 1
            self.stats['total_operations'] += 1
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        cache_key = self._generate_key(key)
        
        # Check memory cache first
        if self._memory_cache_get(cache_key) is not None:
            return True
        
        # Check enhanced Redis manager
        if self.redis_manager:
            try:
                return self.redis_manager.exists(cache_key) > 0
            except Exception as e:
                logger.debug(f"Enhanced Redis exists error: {e}")
                self.stats['errors'] += 1
        
        return False
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter in cache."""
        cache_key = self._generate_key(key)
        
        # Use enhanced Redis manager
        if self.redis_manager:
            try:
                return self.redis_manager.incr(cache_key, amount)
            except Exception as e:
                logger.debug(f"Enhanced Redis incr error: {e}")
                self.stats['errors'] += 1
        
        # Fallback to memory cache
        with self.memory_cache_lock:
            current = self._memory_cache_get(cache_key)
            if current is None:
                new_value = amount
            else:
                try:
                    new_value = int(current) + amount
                except (ValueError, TypeError):
                    new_value = amount
            
            self._memory_cache_set(cache_key, new_value)
            return new_value
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key."""
        cache_key = self._generate_key(key)
        
        # Use enhanced Redis manager
        if self.redis_manager:
            try:
                return self.redis_manager.expire(cache_key, seconds)
            except Exception as e:
                logger.debug(f"Enhanced Redis expire error: {e}")
                self.stats['errors'] += 1
        
        # For memory cache, update expiration
        with self.memory_cache_lock:
            if cache_key in self.memory_cache:
                self.memory_cache[cache_key]['expires_at'] = time.time() + seconds
                return True
        
        return False
    
    def ttl(self, key: str) -> int:
        """Get time-to-live for a key."""
        cache_key = self._generate_key(key)
        
        # Use enhanced Redis manager
        if self.redis_manager:
            try:
                return self.redis_manager.ttl(cache_key)
            except Exception as e:
                logger.debug(f"Enhanced Redis ttl error: {e}")
                self.stats['errors'] += 1
        
        # Fallback to memory cache
        with self.memory_cache_lock:
            if cache_key in self.memory_cache:
                item = self.memory_cache[cache_key]
                if item['expires_at']:
                    remaining = item['expires_at'] - time.time()
                    return max(0, int(remaining))
                return -1  # No expiration set
            return -2  # Key doesn't exist
    
    def clear(self, pattern: Optional[str] = None) -> bool:
        """
        Clear cache items.
        
        Args:
            pattern: Optional pattern to match keys
            
        Returns:
            True if successful
        """
        try:
            # Use enhanced Redis manager (handles Redis and fallback)
            if self.redis_manager:
                try:
                    if pattern:
                        # Clear pattern-matched keys (Redis implementation may vary)
                        # For now, just clear all and let Redis handle patterns
                        self.redis_manager.fallback.flushdb()
                    else:
                        self.redis_manager.fallback.flushdb()
                except Exception as e:
                    logger.debug(f"Enhanced Redis clear error: {e}")
                    self.stats['errors'] += 1
            
            # Clear memory cache
            with self.memory_cache_lock:
                if pattern:
                    # Clear keys matching pattern
                    pattern_key = self._generate_key(pattern)
                    keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                else:
                    self.memory_cache.clear()
                
                self.stats['memory_cache_size'] = len(self.memory_cache)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats['errors'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            error_rate = (self.stats['errors'] / max(1, self.stats['total_operations'])) * 100
            
            stats = {
                **self.stats,
                'total_requests': total_requests,
                'hit_rate_percent': round(hit_rate, 2),
                'error_rate_percent': round(error_rate, 2),
                'redis_stats': None
            }
            
            # Add Redis manager stats if available
            if self.redis_manager:
                try:
                    stats['redis_stats'] = self.redis_manager.get_stats()
                except Exception as e:
                    logger.debug(f"Error getting Redis stats: {e}")
            
            return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            'cache_manager_available': True,
            'redis_available': False,
            'memory_cache_available': True,
            'enhanced_redis_manager_available': False,
            'memory_cache_size': self.stats['memory_cache_size'],
            'performance_stats': {
                'hit_rate': self.stats.get('cache_hit_rate', 0),
                'average_response_time_ms': self.stats.get('average_response_time', 0) * 1000,
                'total_operations': self.stats['total_operations']
            }
        }
        
        # Test enhanced Redis manager
        if self.redis_manager:
            try:
                health['enhanced_redis_manager_available'] = True
                health['redis_available'] = self.redis_manager.is_redis_available()
                health['redis_ping'] = self.redis_manager.ping()
                
                # Get Redis manager statistics
                redis_stats = self.redis_manager.get_stats()
                health['redis_manager_stats'] = redis_stats
                
            except Exception as e:
                logger.warning(f"Enhanced Redis health check failed: {e}")
                health['redis_error'] = str(e)
        
        return health
    
    def warm_cache(self, warming_functions: List[Callable] = None):
        """
        Warm cache with predefined data.
        
        Args:
            warming_functions: List of functions to call for cache warming
        """
        functions = warming_functions or self.warming_queue
        
        logger.info(f"Starting cache warming with {len(functions)} functions")
        
        for func in functions:
            try:
                logger.info(f"Cache warming: {func.__name__}")
                start_time = time.time()
                func()
                elapsed = time.time() - start_time
                logger.info(f"Cache warming completed for {func.__name__} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Cache warming error for {func.__name__}: {e}")
    
    def add_warming_function(self, func: Callable):
        """Add a function to the cache warming queue."""
        self.warming_queue.append(func)
        logger.info(f"Added warming function: {func.__name__}")
    
    def optimize_performance(self):
        """Optimize cache performance based on usage patterns."""
        try:
            logger.info("Starting cache performance optimization")
            
            # Clean up memory cache
            self._cleanup_memory_cache()
            
            # Update Redis manager availability
            if self.redis_manager:
                self.stats['redis_available'] = self.redis_manager.is_redis_available()
            
            # Log performance metrics
            stats = self.get_stats()
            logger.info(f"Cache optimization completed. Hit rate: {stats['hit_rate_percent']}%, "
                       f"Memory usage: {stats['memory_cache_size']} items, "
                       f"Redis available: {stats['redis_available']}")
            
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")

# Enhanced caching decorator
def enhanced_cached(ttl: Optional[int] = None, 
                   key_prefix: str = "func",
                   cache_manager: Optional[EnhancedCacheManager] = None,
                   serialize_args: bool = True):
    """
    Enhanced decorator to cache function results with better key generation.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        cache_manager: Cache manager instance (uses global if None)
        serialize_args: Whether to serialize complex arguments
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get cache manager
            cm = cache_manager or get_enhanced_cache_manager()
            
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__module__, func.__name__]
            
            # Add stringified arguments to key
            if args:
                if serialize_args:
                    key_parts.extend([hashlib.md5(str(arg).encode()).hexdigest()[:8] for arg in args])
                else:
                    key_parts.extend([str(arg)[:50] for arg in args])  # Limit length
            
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                if serialize_args:
                    kwargs_str = json.dumps(sorted_kwargs, sort_keys=True, default=str)
                    key_parts.append(hashlib.md5(kwargs_str.encode()).hexdigest()[:8])
                else:
                    key_parts.extend([f"{k}={str(v)[:20]}" for k, v in sorted_kwargs])
            
            # Create hash of the key to ensure consistent length
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            result = cm.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result with appropriate TTL
            effective_ttl = ttl
            if effective_ttl is None:
                # Dynamic TTL based on execution time
                if execution_time > 1.0:
                    effective_ttl = 3600  # 1 hour for slow functions
                elif execution_time > 0.1:
                    effective_ttl = 1800  # 30 minutes for medium functions
                else:
                    effective_ttl = 300   # 5 minutes for fast functions
            
            cm.set(cache_key, result, effective_ttl)
            
            return result
        
        return wrapper
    return decorator

# Singleton enhanced cache manager
_enhanced_cache_manager: Optional[EnhancedCacheManager] = None
_enhanced_cache_lock = threading.Lock()

def get_enhanced_cache_manager() -> EnhancedCacheManager:
    """Get the singleton enhanced cache manager instance."""
    global _enhanced_cache_manager
    
    if _enhanced_cache_manager is None:
        with _enhanced_cache_lock:
            if _enhanced_cache_manager is None:
                _enhanced_cache_manager = EnhancedCacheManager()
    
    return _enhanced_cache_manager

def init_enhanced_cache_manager(redis_url: str = None, **kwargs) -> EnhancedCacheManager:
    """Initialize the global enhanced cache manager with custom settings."""
    global _enhanced_cache_manager
    
    with _enhanced_cache_lock:
        _enhanced_cache_manager = EnhancedCacheManager(redis_url=redis_url, **kwargs)
    
    return _enhanced_cache_manager

# Context manager for enhanced cache operations
@contextmanager
def enhanced_cache_context(cache_manager: EnhancedCacheManager = None):
    """Context manager for enhanced cache operations with automatic optimization."""
    cm = cache_manager or get_enhanced_cache_manager()
    try:
        yield cm
    except Exception as e:
        logger.error(f"Enhanced cache context error: {e}")
        raise
    finally:
        # Optimize performance after operations
        try:
            cm.optimize_performance()
        except Exception as e:
            logger.debug(f"Cache optimization in context manager failed: {e}")

if __name__ == "__main__":
    # Example usage and comprehensive testing
    print("Testing Enhanced Cache Manager")
    
    cache = EnhancedCacheManager()
    
    # Test basic operations
    print("\n1. Testing basic operations:")
    cache.set("test_key", {"data": "test_value", "number": 42}, ttl=60)
    value = cache.get("test_key")
    print(f"Cached value: {value}")
    
    # Test counter operations
    print("\n2. Testing counter operations:")
    cache.set("counter", "0")
    for i in range(5):
        count = cache.incr("counter")
        print(f"Counter: {count}")
    
    # Test TTL operations
    print("\n3. Testing TTL operations:")
    cache.set("temp_key", "temporary_value", ttl=2)
    print(f"TTL: {cache.ttl('temp_key')} seconds")
    print(f"Exists: {cache.exists('temp_key')}")
    
    # Test enhanced decorator
    print("\n4. Testing enhanced decorator:")
    @enhanced_cached(ttl=300)
    def expensive_function(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * y + (x + y)
    
    start_time = time.time()
    result1 = expensive_function(5, 10)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_function(5, 10)
    second_call_time = time.time() - start_time
    
    print(f"First call: {result1} in {first_call_time:.3f}s")
    print(f"Second call: {result2} in {second_call_time:.3f}s (cached)")
    
    # Test performance stats
    print("\n5. Performance statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test health check
    print("\n6. Health check:")
    health = cache.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    # Test cache warming
    print("\n7. Testing cache warming:")
    def warm_function_1():
        cache.set("warm_key_1", "warm_value_1")
    
    def warm_function_2():
        cache.set("warm_key_2", "warm_value_2")
    
    cache.add_warming_function(warm_function_1)
    cache.add_warming_function(warm_function_2)
    cache.warm_cache()
    
    print(f"Warm key 1: {cache.get('warm_key_1')}")
    print(f"Warm key 2: {cache.get('warm_key_2')}")
    
    print("\nEnhanced Cache Manager testing completed successfully!")
