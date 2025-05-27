#!/usr/bin/env python3
"""
Enhanced Redis Manager with Automatic Installation and Fallback

This module provides comprehensive Redis management including:
- Automatic Redis server installation and startup
- Enhanced fallback mechanisms
- Performance monitoring and optimization
- Production-ready configuration
"""

import sys
import os
import subprocess
import time
import logging
import json
import threading
import platform
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    """Redis configuration with fallback options"""
    host: str = "localhost"
    port: int = 6380  # Updated to use our Docker Redis instance
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Fallback settings
    fallback_enabled: bool = True
    fallback_persistence_file: str = "redis_fallback.json"
    fallback_max_memory_items: int = 10000
    fallback_cleanup_interval: int = 300  # 5 minutes

@dataclass
class RedisStats:
    """Redis performance statistics"""
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    fallback_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_operations: int = 0
    last_successful_ping: Optional[datetime] = None
    uptime_start: datetime = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate connection success rate"""
        if self.connection_attempts == 0:
            return 0.0
        return self.successful_connections / self.connection_attempts

class EnhancedRedisFallback:
    """Enhanced in-memory Redis fallback with advanced features"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._data = {}
        self._expiration = {}
        self._access_times = {}  # For LRU eviction
        self._lock = threading.RLock()
        self._stats = RedisStats()
        
        # Load existing data
        self._load_from_file()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Cleanup expired keys
        cleanup_thread = threading.Thread(
            target=self._cleanup_expired_loop, 
            daemon=True, 
            name="Redis-Fallback-Cleanup"
        )
        cleanup_thread.start()
        
        # Memory management
        memory_thread = threading.Thread(
            target=self._memory_management_loop, 
            daemon=True, 
            name="Redis-Fallback-Memory"
        )
        memory_thread.start()
        
        # Statistics reporting
        stats_thread = threading.Thread(
            target=self._stats_reporting_loop, 
            daemon=True, 
            name="Redis-Fallback-Stats"
        )
        stats_thread.start()
    
    def set(self, key: str, value: Any, ex: Optional[int] = None, 
            px: Optional[int] = None, nx: bool = False, xx: bool = False) -> bool:
        """Enhanced set operation with Redis-compatible options"""
        with self._lock:
            self._stats.total_operations += 1
            
            # Check nx/xx conditions
            exists = key in self._data
            if nx and exists:
                return False
            if xx and not exists:
                return False
            
            # Serialize value
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)
            
            # Set value
            self._data[key] = value
            self._access_times[key] = datetime.now()
            
            # Handle expiration
            if ex is not None:
                self._expiration[key] = datetime.now() + timedelta(seconds=ex)
            elif px is not None:
                self._expiration[key] = datetime.now() + timedelta(milliseconds=px)
            elif key in self._expiration:
                del self._expiration[key]
            
            # Memory management
            self._enforce_memory_limit()
            
            # Persist changes
            self._save_to_file_async()
            
            return True
    
    def get(self, key: str) -> Optional[str]:
        """Enhanced get operation with access tracking"""
        with self._lock:
            self._stats.total_operations += 1
            
            if key not in self._data:
                self._stats.cache_misses += 1
                return None
            
            # Check expiration
            if self._is_expired(key):
                self._remove_key(key)
                self._stats.cache_misses += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = datetime.now()
            self._stats.cache_hits += 1
            
            return self._data[key]
    
    def delete(self, *keys) -> int:
        """Delete multiple keys"""
        with self._lock:
            deleted = 0
            for key in keys:
                if self._remove_key(key):
                    deleted += 1
            
            if deleted > 0:
                self._save_to_file_async()
            
            return deleted
    
    def exists(self, *keys) -> int:
        """Check existence of multiple keys"""
        with self._lock:
            count = 0
            for key in keys:
                if key in self._data and not self._is_expired(key):
                    count += 1
            return count
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter with atomic operation"""
        with self._lock:
            current = self.get(key)
            if current is None:
                new_value = amount
            else:
                try:
                    new_value = int(current) + amount
                except ValueError:
                    raise ValueError(f"Value at key '{key}' is not an integer")
            
            self.set(key, str(new_value))
            return new_value
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for existing key"""
        with self._lock:
            if key in self._data and not self._is_expired(key):
                self._expiration[key] = datetime.now() + timedelta(seconds=seconds)
                self._save_to_file_async()
                return True
            return False
    
    def ttl(self, key: str) -> int:
        """Get time-to-live for key"""
        with self._lock:
            if key not in self._data:
                return -2  # Key doesn't exist
            
            if key not in self._expiration:
                return -1  # Key exists but has no expiration
            
            remaining = self._expiration[key] - datetime.now()
            return max(0, int(remaining.total_seconds()))
    
    def ping(self) -> bool:
        """Health check for fallback system"""
        return True
    
    def flushdb(self) -> bool:
        """Clear all data in current database"""
        with self._lock:
            self._data.clear()
            self._expiration.clear()
            self._access_times.clear()
            self._save_to_file_async()
            return True
    
    def info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        with self._lock:
            memory_usage = sys.getsizeof(self._data) + sys.getsizeof(self._expiration)
            
            return {
                'redis_version': 'fallback-1.0.0',
                'used_memory': memory_usage,
                'used_memory_human': self._format_bytes(memory_usage),
                'connected_clients': 1,
                'total_commands_processed': self._stats.total_operations,
                'keyspace_hits': self._stats.cache_hits,
                'keyspace_misses': self._stats.cache_misses,
                'keys': len(self._data),
                'expires': len(self._expiration),
                'uptime_in_seconds': int((datetime.now() - self._stats.uptime_start).total_seconds()),
                'fallback_mode': True
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired"""
        if key not in self._expiration:
            return False
        return datetime.now() > self._expiration[key]
    
    def _remove_key(self, key: str) -> bool:
        """Remove key and its metadata"""
        removed = False
        if key in self._data:
            del self._data[key]
            removed = True
        if key in self._expiration:
            del self._expiration[key]
        if key in self._access_times:
            del self._access_times[key]
        return removed
    
    def _enforce_memory_limit(self):
        """Enforce memory limits using LRU eviction"""
        max_items = self.config.fallback_max_memory_items
        
        if len(self._data) <= max_items:
            return
        
        # Sort by access time (oldest first)
        items_by_access = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest items
        items_to_remove = len(self._data) - max_items
        for key, _ in items_by_access[:items_to_remove]:
            self._remove_key(key)
        
        logger.info(f"Evicted {items_to_remove} items due to memory limit")
    
    def _cleanup_expired_loop(self):
        """Background thread for cleaning expired keys"""
        while True:
            try:
                time.sleep(self.config.fallback_cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def _cleanup_expired(self):
        """Remove all expired keys"""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, exp_time in self._expiration.items()
                if now > exp_time
            ]
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self._save_to_file_async()
                logger.debug(f"Cleaned up {len(expired_keys)} expired keys")
    
    def _memory_management_loop(self):
        """Background memory management"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                with self._lock:
                    self._enforce_memory_limit()
            except Exception as e:
                logger.error(f"Error in memory management: {e}")
    
    def _stats_reporting_loop(self):
        """Background statistics reporting"""
        while True:
            try:
                time.sleep(300)  # Report every 5 minutes
                self._log_statistics()
            except Exception as e:
                logger.error(f"Error in stats reporting: {e}")
    
    def _log_statistics(self):
        """Log performance statistics"""
        info = self.info()
        logger.info(f"Redis Fallback Stats: {info['keys']} keys, "
                   f"{info['used_memory_human']} memory, "
                   f"{self._stats.cache_hits} hits, "
                   f"{self._stats.cache_misses} misses")
    
    def _save_to_file_async(self):
        """Asynchronous file saving"""
        threading.Thread(target=self._save_to_file, daemon=True).start()
    
    def _save_to_file(self):
        """Save data to persistence file"""
        try:
            data_to_save = {
                'data': self._data,
                'expiration': {
                    k: v.isoformat() for k, v in self._expiration.items()
                },
                'access_times': {
                    k: v.isoformat() for k, v in self._access_times.items()
                },
                'stats': asdict(self._stats),
                'timestamp': datetime.now().isoformat()
            }
            
            # Write to temporary file first
            temp_file = self.config.fallback_persistence_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            # Atomic rename
            os.replace(temp_file, self.config.fallback_persistence_file)
            
        except Exception as e:
            logger.error(f"Error saving fallback data: {e}")
    
    def _load_from_file(self):
        """Load data from persistence file"""
        try:
            if not os.path.exists(self.config.fallback_persistence_file):
                return
            
            with open(self.config.fallback_persistence_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._data = data.get('data', {})
            
            # Convert ISO strings back to datetime objects
            exp_data = data.get('expiration', {})
            self._expiration = {
                k: datetime.fromisoformat(v) for k, v in exp_data.items()
            }
            
            access_data = data.get('access_times', {})
            self._access_times = {
                k: datetime.fromisoformat(v) for k, v in access_data.items()
            }
            
            logger.info(f"Loaded {len(self._data)} keys from fallback storage")
            
        except Exception as e:
            logger.error(f"Error loading fallback data: {e}")
            self._data = {}
            self._expiration = {}
            self._access_times = {}
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f}TB"

class RedisInstaller:
    """Automatic Redis server installation and management"""
    
    @staticmethod
    def detect_platform() -> str:
        """Detect the operating system platform"""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        elif system == "darwin":
            return "macos"
        else:
            return "unknown"
    
    @staticmethod
    def check_docker_available() -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(['docker', '--version'], 
                         capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def install_redis_docker() -> Tuple[bool, str]:
        """Install Redis using Docker"""
        try:
            # Pull Redis image
            subprocess.run(['docker', 'pull', 'redis:latest'], 
                         capture_output=True, check=True, timeout=120)
            
            # Stop existing container if running
            subprocess.run(['docker', 'stop', 'redis-langgraph'], 
                         capture_output=True, timeout=10)
            subprocess.run(['docker', 'rm', 'redis-langgraph'], 
                         capture_output=True, timeout=10)
            
            # Start new Redis container
            cmd = [
                'docker', 'run', '-d',
                '--name', 'redis-langgraph',
                '-p', '6379:6379',
                '--restart', 'unless-stopped',
                'redis:latest',
                'redis-server',
                '--appendonly', 'yes',
                '--maxmemory', '256mb',
                '--maxmemory-policy', 'allkeys-lru'
            ]
            
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            
            # Wait for Redis to start
            time.sleep(3)
            
            return True, "Redis installed and started via Docker"
            
        except subprocess.CalledProcessError as e:
            return False, f"Docker installation failed: {e.stderr.decode()}"
        except Exception as e:
            return False, f"Docker installation error: {str(e)}"
    
    @staticmethod
    def install_redis_windows() -> Tuple[bool, str]:
        """Install Redis on Windows using Memurai"""
        try:
            # Check if Chocolatey is available
            subprocess.run(['choco', '--version'], 
                         capture_output=True, check=True, timeout=5)
            
            # Install Memurai (Redis for Windows)
            result = subprocess.run(
                ['choco', 'install', 'memurai-developer', '-y'],
                capture_output=True, timeout=300
            )
            
            if result.returncode == 0:
                # Start Memurai service
                subprocess.run(['net', 'start', 'memurai'], 
                             capture_output=True, timeout=30)
                return True, "Memurai (Redis for Windows) installed and started"
            else:
                return False, "Chocolatey installation failed"
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "Chocolatey not available for Redis installation"
        except Exception as e:
            return False, f"Windows installation error: {str(e)}"
    
    @staticmethod
    def install_redis_linux() -> Tuple[bool, str]:
        """Install Redis on Linux"""
        try:
            # Try apt (Ubuntu/Debian)
            subprocess.run(['sudo', 'apt', 'update'], 
                         capture_output=True, check=True, timeout=60)
            subprocess.run(['sudo', 'apt', 'install', '-y', 'redis-server'], 
                         capture_output=True, check=True, timeout=180)
            subprocess.run(['sudo', 'systemctl', 'start', 'redis-server'], 
                         capture_output=True, check=True, timeout=30)
            subprocess.run(['sudo', 'systemctl', 'enable', 'redis-server'], 
                         capture_output=True, check=True, timeout=30)
            
            return True, "Redis installed and started via apt"
            
        except subprocess.CalledProcessError:
            try:
                # Try yum (CentOS/RHEL)
                subprocess.run(['sudo', 'yum', 'install', '-y', 'redis'], 
                             capture_output=True, check=True, timeout=180)
                subprocess.run(['sudo', 'systemctl', 'start', 'redis'], 
                             capture_output=True, check=True, timeout=30)
                subprocess.run(['sudo', 'systemctl', 'enable', 'redis'], 
                             capture_output=True, check=True, timeout=30)
                
                return True, "Redis installed and started via yum"
                
            except subprocess.CalledProcessError as e:
                return False, f"Linux package manager installation failed: {e}"
        except Exception as e:
            return False, f"Linux installation error: {str(e)}"

class EnhancedRedisManager:
    """Enhanced Redis manager with auto-installation and intelligent fallback"""
    
    def __init__(self, config: RedisConfig = None):
        self.config = config or RedisConfig()
        self.redis_client = None
        self.fallback = EnhancedRedisFallback(self.config)
        self.stats = RedisStats()
        self._connection_lock = threading.Lock()
        self._last_health_check = datetime.now()
        
        # Initialize connection
        self._initialize_redis()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def _initialize_redis(self) -> bool:
        """Initialize Redis connection with automatic installation"""
        try:
            import redis
            
            # Try to connect to existing Redis
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.stats.successful_connections += 1
            self.stats.last_successful_ping = datetime.now()
            logger.info("Redis connection established successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.stats.failed_connections += 1
            
            # Attempt automatic installation
            if self._attempt_auto_installation():
                # Retry connection after installation
                return self._initialize_redis()
            
            # Fall back to in-memory storage
            logger.info("Using enhanced fallback storage")
            self.redis_client = None
            return False
        
        finally:
            self.stats.connection_attempts += 1
    
    def _attempt_auto_installation(self) -> bool:
        """Attempt automatic Redis installation"""
        logger.info("Attempting automatic Redis installation...")
        
        platform = RedisInstaller.detect_platform()
        
        # Try Docker first (cross-platform)
        if RedisInstaller.check_docker_available():
            success, message = RedisInstaller.install_redis_docker()
            if success:
                logger.info(f"Redis installation successful: {message}")
                time.sleep(5)  # Wait for startup
                return True
            else:
                logger.warning(f"Docker installation failed: {message}")
        
        # Platform-specific installation
        if platform == "windows":
            success, message = RedisInstaller.install_redis_windows()
        elif platform == "linux":
            success, message = RedisInstaller.install_redis_linux()
        else:
            success, message = False, f"Automatic installation not supported for {platform}"
        
        if success:
            logger.info(f"Redis installation successful: {message}")
            time.sleep(5)  # Wait for startup
            return True
        else:
            logger.warning(f"Platform installation failed: {message}")
            return False
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name="Redis-Health-Monitor"
        )
        health_thread.start()
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    def _perform_health_check(self):
        """Perform Redis health check"""
        if self.redis_client is None:
            return
        
        try:
            self.redis_client.ping()
            self.stats.last_successful_ping = datetime.now()
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self.stats.failed_connections += 1
            
            # Try to reconnect
            with self._connection_lock:
                self._initialize_redis()
    
    def set(self, key: str, value: Any, **kwargs) -> bool:
        """Set key-value with automatic fallback"""
        try:
            if self.redis_client:
                result = self.redis_client.set(key, value, **kwargs)
                if result:
                    return True
        except Exception as e:
            logger.debug(f"Redis set failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.set(key, value, **kwargs)
    
    def get(self, key: str) -> Optional[str]:
        """Get value with automatic fallback"""
        try:
            if self.redis_client:
                result = self.redis_client.get(key)
                if result is not None:
                    return result
        except Exception as e:
            logger.debug(f"Redis get failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.get(key)
    
    def delete(self, *keys) -> int:
        """Delete keys with automatic fallback"""
        try:
            if self.redis_client:
                return self.redis_client.delete(*keys)
        except Exception as e:
            logger.debug(f"Redis delete failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.delete(*keys)
    
    def exists(self, *keys) -> int:
        """Check key existence with automatic fallback"""
        try:
            if self.redis_client:
                return self.redis_client.exists(*keys)
        except Exception as e:
            logger.debug(f"Redis exists failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.exists(*keys)
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment with automatic fallback"""
        try:
            if self.redis_client:
                return self.redis_client.incr(key, amount)
        except Exception as e:
            logger.debug(f"Redis incr failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.incr(key, amount)
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration with automatic fallback"""
        try:
            if self.redis_client:
                return self.redis_client.expire(key, seconds)
        except Exception as e:
            logger.debug(f"Redis expire failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.expire(key, seconds)
    
    def ttl(self, key: str) -> int:
        """Get TTL with automatic fallback"""
        try:
            if self.redis_client:
                return self.redis_client.ttl(key)
        except Exception as e:
            logger.debug(f"Redis ttl failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.ttl(key)
    
    def ping(self) -> bool:
        """Ping with automatic fallback"""
        try:
            if self.redis_client:
                return self.redis_client.ping()
        except Exception as e:
            logger.debug(f"Redis ping failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback
        return self.fallback.ping()
    
    def info(self, section: str = None) -> Dict[str, Any]:
        """Get info with enhanced fallback data"""
        try:
            if self.redis_client:
                redis_info = self.redis_client.info(section)
                redis_info['fallback_mode'] = False
                redis_info['fallback_operations'] = self.stats.fallback_operations
                return redis_info
        except Exception as e:
            logger.debug(f"Redis info failed, using fallback: {e}")
            self.stats.fallback_operations += 1
        
        # Use fallback info
        fallback_info = self.fallback.info()
        fallback_info.update({
            'connection_success_rate': self.stats.get_success_rate(),
            'total_connection_attempts': self.stats.connection_attempts,
            'fallback_operations': self.stats.fallback_operations
        })
        return fallback_info
    
    def is_redis_available(self) -> bool:
        """Check if Redis server is available"""
        return self.redis_client is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'redis_available': self.is_redis_available(),
            'connection_stats': asdict(self.stats),
            'fallback_active': not self.is_redis_available(),
            'info': self.info()
        }

# Global enhanced Redis manager instance
enhanced_redis_manager = None

def get_redis_manager(config: RedisConfig = None) -> EnhancedRedisManager:
    """Get global Redis manager instance"""
    global enhanced_redis_manager
    if enhanced_redis_manager is None:
        enhanced_redis_manager = EnhancedRedisManager(config)
    return enhanced_redis_manager

def get_redis_fallback():
    """Get Redis fallback manager (lazy-loaded for backward compatibility)"""
    return get_redis_manager()

# Lazy-loaded backward compatibility - no auto-initialization on import
redis_fallback = None
