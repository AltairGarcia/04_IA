#!/usr/bin/env python3
"""
Redis Fallback Handler for LangGraph 101

This module provides fallback functionality when Redis is not available,
using in-memory storage and file-based persistence as alternatives.
"""

import json
import os
import time
import threading
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RedisFeatures:
    """In-memory implementation of Redis features"""
    
    def __init__(self, persistence_file: str = "redis_fallback.json"):
        self._data = {}
        self._expiration = {}
        self._persistence_file = persistence_file
        self._lock = threading.RLock()
        self._load_from_file()
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a key-value pair with optional expiration"""
        with self._lock:
            self._data[key] = json.dumps(value) if not isinstance(value, str) else value
            
            if ex:
                self._expiration[key] = datetime.now() + timedelta(seconds=ex)
            elif key in self._expiration:
                del self._expiration[key]
            
            self._save_to_file()
            return True
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        with self._lock:
            if key in self._data:
                # Check if expired
                if key in self._expiration and datetime.now() > self._expiration[key]:
                    del self._data[key]
                    del self._expiration[key]
                    self._save_to_file()
                    return None
                
                return self._data[key]
            return None
    
    def delete(self, *keys) -> int:
        """Delete keys"""
        with self._lock:
            deleted = 0
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    deleted += 1
                if key in self._expiration:
                    del self._expiration[key]
            
            if deleted > 0:
                self._save_to_file()
            return deleted
    
    def exists(self, *keys) -> int:
        """Check if keys exist"""
        with self._lock:
            count = 0
            for key in keys:
                if key in self._data:
                    # Check if expired
                    if key in self._expiration and datetime.now() > self._expiration[key]:
                        del self._data[key]
                        del self._expiration[key]
                    else:
                        count += 1
            return count
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value"""
        with self._lock:
            current = self.get(key)
            if current is None:
                new_value = amount
            else:
                try:
                    new_value = int(current) + amount
                except ValueError:
                    raise ValueError("Value is not an integer")
            
            self.set(key, str(new_value))
            return new_value
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key"""
        with self._lock:
            if key in self._data:
                self._expiration[key] = datetime.now() + timedelta(seconds=seconds)
                self._save_to_file()
                return True
            return False
    
    def ttl(self, key: str) -> int:
        """Get time to live for a key"""
        with self._lock:
            if key in self._expiration:
                remaining = self._expiration[key] - datetime.now()
                return max(0, int(remaining.total_seconds()))
            return -1 if key in self._data else -2
    
    def ping(self) -> bool:
        """Check if the fallback system is working"""
        return True
    
    def _cleanup_expired(self):
        """Background thread to clean up expired keys"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                with self._lock:
                    now = datetime.now()
                    expired_keys = [
                        key for key, exp_time in self._expiration.items()
                        if now > exp_time
                    ]
                    
                    for key in expired_keys:
                        if key in self._data:
                            del self._data[key]
                        del self._expiration[key]
                    
                    if expired_keys:
                        self._save_to_file()
                        logger.debug(f"Cleaned up {len(expired_keys)} expired keys")
                        
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def _save_to_file(self):
        """Save data to file for persistence"""
        try:
            data_to_save = {
                'data': self._data,
                'expiration': {
                    k: v.isoformat() for k, v in self._expiration.items()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self._persistence_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving to persistence file: {e}")
    
    def _load_from_file(self):
        """Load data from file"""
        try:
            if os.path.exists(self._persistence_file):
                with open(self._persistence_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._data = data.get('data', {})
                
                # Convert expiration times back to datetime objects
                exp_data = data.get('expiration', {})
                self._expiration = {
                    k: datetime.fromisoformat(v) for k, v in exp_data.items()
                }
                
                logger.info(f"Loaded {len(self._data)} keys from persistence file")
                
        except Exception as e:
            logger.error(f"Error loading from persistence file: {e}")
            self._data = {}
            self._expiration = {}


class RedisFallbackManager:
    """Manager for Redis fallback functionality"""
    
    def __init__(self):
        self.redis_features = RedisFeatures()
        self.redis_available = False
        self._check_redis_availability()
    
    def _check_redis_availability(self):
        """Check if Redis is available"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6380, socket_timeout=1)
            r.ping()
            self.redis_available = True
            logger.info("Redis server is available")
        except Exception:
            self.redis_available = False
            logger.warning("Redis server not available, using fallback")
    
    def get_client(self):
        """Get Redis client or fallback"""
        if self.redis_available:
            try:
                import redis
                return redis.Redis(host='localhost', port=6380)
            except Exception:
                self.redis_available = False
        
        return self.redis_features
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available"""
        return self.redis_available


# Global fallback manager instance
redis_fallback = RedisFallbackManager()
