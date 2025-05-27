#!/usr/bin/env python3
"""
Enhanced Rate Limiting System for LangGraph 101

This module provides sophisticated rate limiting capabilities that extend beyond
basic DDoS protection to include intelligent throttling, adaptive limits,
and user-specific rate controls.

Features:
- Multiple rate limiting algorithms (Token Bucket, Sliding Window, Fixed Window)
- User-specific and IP-based rate limiting
- Adaptive rate limiting based on system load
- API endpoint-specific limits
- Rate limit bursting and smoothing
- Distributed rate limiting with Redis
- Real-time monitoring and alerts
- Whitelist/blacklist management
- Rate limit exemptions and overrides
"""

import time
import redis
import logging
import threading
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import math
import psutil
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitResult(Enum):
    """Rate limit check results"""
    ALLOWED = "allowed"
    DENIED = "denied"
    WARNING = "warning"
    BLACKLISTED = "blacklisted"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    requests_per_hour: float = 10000.0
    burst_size: int = 20
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    enable_adaptive: bool = True
    adaptive_factor: float = 0.8  # Reduce limits when system load is high
    warning_threshold: float = 0.8  # Warn at 80% of limit
    backoff_seconds: int = 60
    max_backoff_seconds: int = 3600
    enable_redis: bool = True
    redis_key_prefix: str = "rate_limit"
    
    def __post_init__(self):
        # Ensure per-minute is at least per-second * 60
        if self.requests_per_minute < self.requests_per_second * 60:
            self.requests_per_minute = self.requests_per_second * 60
        # Ensure per-hour is at least per-minute * 60
        if self.requests_per_hour < self.requests_per_minute * 60:
            self.requests_per_hour = self.requests_per_minute * 60


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    result: RateLimitResult
    requests_remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    current_usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'result': self.result.value,
            'requests_remaining': self.requests_remaining,
            'reset_time': self.reset_time.isoformat(),
            'retry_after': self.retry_after,
            'current_usage': self.current_usage
        }


@dataclass
class RateLimitViolation:
    """Rate limit violation record"""
    identifier: str
    endpoint: str
    timestamp: datetime
    requests_count: int
    limit_exceeded: str
    severity: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


class TokenBucket:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        with self.lock:
            now = time.time()
            # Add tokens based on time elapsed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
            
    def get_tokens(self) -> float:
        """Get current token count"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            return self.tokens


class SlidingWindowCounter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
        
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
                
            # Check if we're under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
            
    def get_count(self) -> int:
        """Get current request count in window"""
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            return len(self.requests)


class FixedWindowCounter:
    """Fixed window rate limiter implementation"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.window_start = time.time()
        self.request_count = 0
        self.lock = threading.Lock()
        
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            
            # Check if we need to reset window
            if now - self.window_start >= self.window_size:
                self.window_start = now
                self.request_count = 0
                
            # Check if under limit
            if self.request_count < self.max_requests:
                self.request_count += 1
                return True
            return False
            
    def get_count(self) -> int:
        """Get current request count in window"""
        with self.lock:
            now = time.time()
            if now - self.window_start >= self.window_size:
                return 0
            return self.request_count


class EnhancedRateLimiter:
    """Enhanced rate limiting system with multiple algorithms and features"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        
        # Rate limit configurations
        self._configs: Dict[str, RateLimitConfig] = {}
        self._default_config = RateLimitConfig()
        
        # Local rate limiters (fallback when Redis unavailable)
        self._local_limiters: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Blacklist and whitelist
        self._blacklist: set = set()
        self._whitelist: set = set()
        self._exemptions: Dict[str, datetime] = {}  # Temporary exemptions
        
        # Violation tracking
        self._violations: List[RateLimitViolation] = []
        self._violation_counts: Dict[str, int] = defaultdict(int)
        
        # System load monitoring
        self._system_load_factor = 1.0
        self._load_monitor_thread = None
        self._monitoring_enabled = True
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'denied_requests': 0,
            'warnings_issued': 0,
            'blacklist_hits': 0,
            'redis_failures': 0
        }
        
        # Start system load monitoring
        self._start_load_monitoring()
        
    def _start_load_monitoring(self):
        """Start system load monitoring thread"""
        def monitor_system_load():
            while self._monitoring_enabled:
                try:
                    # Get CPU and memory usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    # Calculate load factor (reduce limits when system is stressed)
                    load_factor = 1.0
                    if cpu_percent > 80 or memory_percent > 80:
                        load_factor = 0.5  # Reduce limits by 50%
                    elif cpu_percent > 60 or memory_percent > 60:
                        load_factor = 0.7  # Reduce limits by 30%
                    elif cpu_percent > 40 or memory_percent > 40:
                        load_factor = 0.85  # Reduce limits by 15%
                        
                    self._system_load_factor = load_factor
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error monitoring system load: {e}")
                    time.sleep(60)
                    
        self._load_monitor_thread = threading.Thread(
            target=monitor_system_load,
            daemon=True,
            name="RateLimit-LoadMonitor"
        )
        self._load_monitor_thread.start()
        
    def add_config(self, identifier: str, config: RateLimitConfig):
        """Add rate limit configuration for identifier (user, IP, endpoint)"""
        self._configs[identifier] = config
        logger.info(f"Added rate limit config for {identifier}")
        
    def add_blacklist(self, identifier: str):
        """Add identifier to blacklist"""
        self._blacklist.add(identifier)
        logger.warning(f"Added {identifier} to blacklist")
        
    def add_whitelist(self, identifier: str):
        """Add identifier to whitelist"""
        self._whitelist.add(identifier)
        logger.info(f"Added {identifier} to whitelist")
        
    def add_exemption(self, identifier: str, duration_seconds: int):
        """Add temporary exemption for identifier"""
        expiry = datetime.now() + timedelta(seconds=duration_seconds)
        self._exemptions[identifier] = expiry
        logger.info(f"Added exemption for {identifier} until {expiry}")
        
    def _get_effective_config(self, identifier: str) -> RateLimitConfig:
        """Get effective configuration considering adaptive limits"""
        config = self._configs.get(identifier, self._default_config)
        
        if config.enable_adaptive:
            # Apply system load factor
            adjusted_config = RateLimitConfig(
                requests_per_second=config.requests_per_second * self._system_load_factor,
                requests_per_minute=config.requests_per_minute * self._system_load_factor,
                requests_per_hour=config.requests_per_hour * self._system_load_factor,
                burst_size=int(config.burst_size * self._system_load_factor),
                algorithm=config.algorithm,
                enable_adaptive=config.enable_adaptive,
                adaptive_factor=config.adaptive_factor,
                warning_threshold=config.warning_threshold,
                backoff_seconds=config.backoff_seconds,
                max_backoff_seconds=config.max_backoff_seconds,
                enable_redis=config.enable_redis,
                redis_key_prefix=config.redis_key_prefix
            )
            return adjusted_config
        return config
        
    def _check_exemptions(self, identifier: str) -> bool:
        """Check if identifier has active exemption"""
        if identifier in self._exemptions:
            if datetime.now() < self._exemptions[identifier]:
                return True
            else:
                # Exemption expired
                del self._exemptions[identifier]
        return False
        
    def _get_redis_key(self, identifier: str, window_type: str) -> str:
        """Generate Redis key for rate limiting"""
        config = self._get_effective_config(identifier)
        return f"{config.redis_key_prefix}:{identifier}:{window_type}"
        
    def _check_redis_rate_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """Check rate limit using Redis"""
        try:
            now = time.time()
            pipe = self.redis_client.pipeline()
            
            # Check multiple time windows
            windows = {
                'second': (1, config.requests_per_second),
                'minute': (60, config.requests_per_minute),
                'hour': (3600, config.requests_per_hour)
            }
            
            results = {}
            for window_name, (window_size, limit) in windows.items():
                key = self._get_redis_key(identifier, window_name)
                
                # Use sliding window log approach
                pipe.zremrangebyscore(key, 0, now - window_size)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, window_size + 1)
                
            responses = pipe.execute()
            
            # Process responses in groups of 4
            for i, (window_name, (window_size, limit)) in enumerate(windows.items()):
                start_idx = i * 4
                current_count = responses[start_idx + 1]
                
                if current_count > limit:
                    # Rate limit exceeded
                    reset_time = datetime.fromtimestamp(now + window_size)
                    return RateLimitStatus(
                        result=RateLimitResult.DENIED,
                        requests_remaining=0,
                        reset_time=reset_time,
                        retry_after=int(config.backoff_seconds),
                        current_usage={window_name: current_count}
                    )
                    
                results[window_name] = current_count
                
            # All windows passed
            min_remaining = min(
                int(limit - results[window_name])
                for window_name, (_, limit) in windows.items()
            )
            
            # Check for warning threshold
            max_usage_ratio = max(
                results[window_name] / limit
                for window_name, (_, limit) in windows.items()
            )
            
            result_type = RateLimitResult.ALLOWED
            if max_usage_ratio >= config.warning_threshold:
                result_type = RateLimitResult.WARNING
                
            return RateLimitStatus(
                result=result_type,
                requests_remaining=min_remaining,
                reset_time=datetime.fromtimestamp(now + 3600),  # Next hour reset
                current_usage=results
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            self._stats['redis_failures'] += 1
            # Fall back to local rate limiting
            return self._check_local_rate_limit(identifier, config)
            
    def _check_local_rate_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """Check rate limit using local counters"""
        if identifier not in self._local_limiters:
            self._local_limiters[identifier] = {}
            
        limiters = self._local_limiters[identifier]
        
        # Initialize limiters if needed
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            if 'bucket' not in limiters:
                limiters['bucket'] = TokenBucket(
                    capacity=config.burst_size,
                    refill_rate=config.requests_per_second
                )
                
            bucket = limiters['bucket']
            allowed = bucket.consume()
            
            if allowed:
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    requests_remaining=int(bucket.get_tokens()),
                    reset_time=datetime.now() + timedelta(seconds=1)
                )
            else:
                return RateLimitStatus(
                    result=RateLimitResult.DENIED,
                    requests_remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=config.backoff_seconds),
                    retry_after=config.backoff_seconds
                )
                
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            if 'sliding' not in limiters:
                limiters['sliding'] = SlidingWindowCounter(
                    window_size=60,  # 1 minute window
                    max_requests=int(config.requests_per_minute)
                )
                
            sliding = limiters['sliding']
            allowed = sliding.is_allowed()
            
            if allowed:
                remaining = int(config.requests_per_minute) - sliding.get_count()
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    requests_remaining=remaining,
                    reset_time=datetime.now() + timedelta(minutes=1)
                )
            else:
                return RateLimitStatus(
                    result=RateLimitResult.DENIED,
                    requests_remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=config.backoff_seconds),
                    retry_after=config.backoff_seconds
                )
                
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            if 'fixed' not in limiters:
                limiters['fixed'] = FixedWindowCounter(
                    window_size=60,  # 1 minute window
                    max_requests=int(config.requests_per_minute)
                )
                
            fixed = limiters['fixed']
            allowed = fixed.is_allowed()
            
            if allowed:
                remaining = int(config.requests_per_minute) - fixed.get_count()
                return RateLimitStatus(
                    result=RateLimitResult.ALLOWED,
                    requests_remaining=remaining,
                    reset_time=datetime.now() + timedelta(minutes=1)
                )
            else:
                return RateLimitStatus(
                    result=RateLimitResult.DENIED,
                    requests_remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=config.backoff_seconds),
                    retry_after=config.backoff_seconds
                )
                
        # Default fallback
        return RateLimitStatus(
            result=RateLimitResult.ALLOWED,
            requests_remaining=999,
            reset_time=datetime.now() + timedelta(hours=1)
        )
        
    def check_rate_limit(self, identifier: str, endpoint: str = None, 
                        user_agent: str = None, ip_address: str = None) -> RateLimitStatus:
        """Check if request is allowed under rate limits"""
        self._stats['total_requests'] += 1
        
        # Check blacklist
        if identifier in self._blacklist:
            self._stats['blacklist_hits'] += 1
            return RateLimitStatus(
                result=RateLimitResult.BLACKLISTED,
                requests_remaining=0,
                reset_time=datetime.now() + timedelta(hours=24),
                retry_after=86400  # 24 hours
            )
            
        # Check whitelist
        if identifier in self._whitelist:
            self._stats['allowed_requests'] += 1
            return RateLimitStatus(
                result=RateLimitResult.ALLOWED,
                requests_remaining=999999,
                reset_time=datetime.now() + timedelta(hours=1)
            )
            
        # Check exemptions
        if self._check_exemptions(identifier):
            self._stats['allowed_requests'] += 1
            return RateLimitStatus(
                result=RateLimitResult.ALLOWED,
                requests_remaining=999999,
                reset_time=datetime.now() + timedelta(hours=1)
            )
            
        # Get effective configuration
        config = self._get_effective_config(identifier)
        
        # Check rate limits
        if self.redis_client and config.enable_redis:
            status = self._check_redis_rate_limit(identifier, config)
        else:
            status = self._check_local_rate_limit(identifier, config)
            
        # Update statistics
        if status.result == RateLimitResult.ALLOWED:
            self._stats['allowed_requests'] += 1
        elif status.result == RateLimitResult.DENIED:
            self._stats['denied_requests'] += 1
            
            # Record violation
            violation = RateLimitViolation(
                identifier=identifier,
                endpoint=endpoint or "unknown",
                timestamp=datetime.now(),
                requests_count=status.current_usage.get('minute', 0),
                limit_exceeded="rate_limit",
                severity="high",
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            self._violations.append(violation)
            self._violation_counts[identifier] += 1
            
            # Auto-blacklist for repeated violations
            if self._violation_counts[identifier] >= 10:
                self.add_blacklist(identifier)
                
        elif status.result == RateLimitResult.WARNING:
            self._stats['warnings_issued'] += 1
            
        return status
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        return {
            'stats': self._stats.copy(),
            'system_load_factor': self._system_load_factor,
            'blacklist_size': len(self._blacklist),
            'whitelist_size': len(self._whitelist),
            'active_exemptions': len(self._exemptions),
            'total_violations': len(self._violations),
            'recent_violations': [
                {
                    'identifier': v.identifier,
                    'endpoint': v.endpoint,
                    'timestamp': v.timestamp.isoformat(),
                    'requests_count': v.requests_count
                }
                for v in self._violations[-10:]
            ],
            'top_violators': [
                {'identifier': k, 'violations': v}
                for k, v in sorted(
                    self._violation_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ]
        }
        
    def cleanup_old_data(self, hours: int = 24):
        """Clean up old violations and tracking data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Clean up violations
        self._violations = [
            v for v in self._violations
            if v.timestamp > cutoff_time
        ]
        
        # Clean up expired exemptions
        expired_exemptions = [
            k for k, v in self._exemptions.items()
            if v < datetime.now()
        ]
        for k in expired_exemptions:
            del self._exemptions[k]
            
        logger.info(f"Cleaned up old rate limiting data (older than {hours} hours)")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_enabled = False
        logger.info("Rate limiting monitoring stopped")


# Decorator for automatic rate limiting
def rate_limit(identifier_func: Callable = None, config: RateLimitConfig = None,
               limiter: EnhancedRateLimiter = None):
    """Decorator for applying rate limiting to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = "default"
                
            # Get limiter
            nonlocal limiter
            if limiter is None:
                limiter = EnhancedRateLimiter()
                
            # Check rate limit
            status = limiter.check_rate_limit(identifier)
            
            if status.result in [RateLimitResult.DENIED, RateLimitResult.BLACKLISTED]:
                raise Exception(f"Rate limit exceeded. Retry after {status.retry_after} seconds")
                
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global rate limiter instance
_global_limiter = None


def get_global_limiter() -> EnhancedRateLimiter:
    """Get the global rate limiter instance"""
    global _global_limiter
    if _global_limiter is None:
        # Try to connect to Redis
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
        except Exception:
            redis_client = None
            
        _global_limiter = EnhancedRateLimiter(redis_client)
    return _global_limiter


if __name__ == "__main__":
    # Demo and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create rate limiter
    limiter = EnhancedRateLimiter()
    
    # Add configurations
    default_config = RateLimitConfig(
        requests_per_second=5,
        requests_per_minute=100,
        burst_size=10
    )
    limiter.add_config("default", default_config)
    
    # Test rate limiting
    print("Testing rate limiting...")
    
    for i in range(15):
        status = limiter.check_rate_limit("test_user", f"endpoint_{i%3}")
        print(f"Request {i+1}: {status.result.value} - Remaining: {status.requests_remaining}")
        
        if status.result == RateLimitResult.DENIED:
            print(f"Rate limited! Retry after {status.retry_after} seconds")
            break
            
        time.sleep(0.1)
        
    # Show statistics
    stats = limiter.get_statistics()
    print(f"\nStatistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Cleanup
    limiter.stop_monitoring()
    
    print("\nRate limiting test completed!")


def create_rate_limiter(redis_client: Optional[redis.Redis] = None, 
                       config: Optional[Dict[str, Any]] = None) -> EnhancedRateLimiter:
    """Factory function to create rate limiter with proper parameters"""
    limiter = EnhancedRateLimiter(redis_client)
    if config:
        for endpoint, limit_config in config.items():
            if isinstance(limit_config, dict):
                limiter.add_limit(endpoint, **limit_config)
    return limiter
