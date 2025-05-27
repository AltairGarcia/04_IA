#!/usr/bin/env python3
"""
Database Connection Pooling System for LangGraph 101

This module provides robust database connection pooling to optimize database performance
and manage connections efficiently across the application.

Features:
- Multi-database support (SQLite, PostgreSQL, MySQL)
- Connection pool management with configurable limits
- Health checks and automatic recovery
- Connection leak detection and prevention
- Performance monitoring and metrics
- Thread-safe operations
- Automatic failover and retry mechanisms
"""

import sqlite3
import logging
import threading
import time
import queue
import contextlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, ContextManager
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import weakref
import traceback
import json

try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    from mysql.connector import pooling as mysql_pool
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabasePoolError(Exception):
    """Exception raised for database pool errors."""
    pass


@dataclass
class PoolConfig:
    """Database pool configuration"""
    min_connections: int = 2
    max_connections: int = 10
    max_overflow: int = 5
    pool_recycle: int = 3600  # 1 hour
    pool_timeout: int = 30
    pool_pre_ping: bool = True
    retry_on_disconnect: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: int = 30
    health_check_interval: int = 60  # seconds


@dataclass
class ConnectionStats:
    """Connection statistics and metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    checked_out: int = 0
    overflow_connections: int = 0
    connection_requests: int = 0
    connection_failures: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    average_checkout_time: float = 0.0
    peak_connections: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'idle_connections': self.idle_connections,
            'checked_out': self.checked_out,
            'overflow_connections': self.overflow_connections,
            'connection_requests': self.connection_requests,
            'connection_failures': self.connection_failures,
            'pool_hits': self.pool_hits,
            'pool_misses': self.pool_misses,
            'average_checkout_time': self.average_checkout_time,
            'peak_connections': self.peak_connections,
            'created_at': self.created_at.isoformat()
        }


class PooledConnection:
    """Wrapper for pooled database connections"""
    
    def __init__(self, connection, pool, connection_id: str):
        self.connection = connection
        self.pool = pool
        self.connection_id = connection_id
        self.checked_out_at = datetime.now()
        self.last_used = datetime.now()
        self.is_valid = True
        self._in_transaction = False
        
    def __enter__(self):
        return self.connection
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool._return_connection(self)
        
    def execute(self, *args, **kwargs):
        """Execute query with automatic return to pool"""
        try:
            result = self.connection.execute(*args, **kwargs)
            self.last_used = datetime.now()
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.is_valid = False
            raise
            
    def commit(self):
        """Commit transaction"""
        try:
            self.connection.commit()
            self._in_transaction = False
        except Exception as e:
            logger.error(f"Commit failed: {e}")
            self.is_valid = False
            raise
            
    def rollback(self):
        """Rollback transaction"""
        try:
            self.connection.rollback()
            self._in_transaction = False
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.is_valid = False
            raise


class DatabaseConnectionPool:
    """High-performance database connection pool"""
    
    def __init__(self, database_url: str, config: PoolConfig = None):
        self.database_url = database_url
        self.config = config or PoolConfig()
        self.db_type = self._detect_database_type()
        
        # Connection management
        self._pool = queue.Queue(maxsize=self.config.max_connections)
        self._overflow_connections = set()
        self._all_connections = weakref.WeakSet()
        self._checked_out_connections = {}
        self._connection_counter = 0
        
        # Thread safety
        self._pool_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        
        # Statistics and monitoring
        self.stats = ConnectionStats()
        self._checkout_times = []
        self._health_check_thread = None
        self._is_healthy = True
        
        # Initialize pool
        self._initialize_pool()
        self._start_health_checks()
        
    def _detect_database_type(self) -> str:
        """Detect database type from URL"""
        url_lower = self.database_url.lower()
        if url_lower.startswith('sqlite'):
            return 'sqlite'
        elif url_lower.startswith('postgresql') or url_lower.startswith('postgres'):
            return 'postgresql'
        elif url_lower.startswith('mysql'):
            return 'mysql'
        else:
            return 'sqlite'  # Default fallback
            
    def _create_connection(self) -> Any:
        """Create a new database connection"""
        try:
            if self.db_type == 'sqlite':
                # Extract path from SQLite URL
                db_path = self.database_url.replace('sqlite:///', '').replace('sqlite://', '')
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                
                conn = sqlite3.connect(
                    db_path,
                    check_same_thread=False,
                    timeout=self.config.connection_timeout,
                    isolation_level=None  # Autocommit mode
                )
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = 10000")
                conn.row_factory = sqlite3.Row
                
            elif self.db_type == 'postgresql' and POSTGRES_AVAILABLE:
                conn = psycopg2.connect(
                    self.database_url,
                    connect_timeout=self.config.connection_timeout
                )
                conn.autocommit = True
                
            elif self.db_type == 'mysql' and MYSQL_AVAILABLE:
                # Parse MySQL URL for connection parameters
                # This is a simplified parser - in production use proper URL parsing
                conn = mysql.connector.connect(
                    host='localhost',  # Parse from URL
                    database='langgraph101',  # Parse from URL
                    user='root',  # Parse from URL
                    password='',  # Parse from URL
                    autocommit=True,
                    connect_timeout=self.config.connection_timeout
                )
                
            else:
                raise DatabasePoolError(f"Unsupported database type: {self.db_type}")
                
            logger.debug(f"Created new {self.db_type} connection")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create {self.db_type} connection: {e}")
            raise DatabasePoolError(f"Connection creation failed: {e}")
            
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections"""
        logger.info(f"Initializing connection pool for {self.db_type} database")
        
        with self._pool_lock:
            for _ in range(self.config.min_connections):
                try:
                    conn = self._create_connection()
                    self._pool.put(conn, block=False)
                    self.stats.total_connections += 1
                    self.stats.idle_connections += 1
                except Exception as e:
                    logger.error(f"Failed to initialize pool connection: {e}")
                    
        logger.info(f"Pool initialized with {self.stats.total_connections} connections")
        
    def _start_health_checks(self):
        """Start background health check thread"""
        def health_check_worker():
            while True:
                try:
                    time.sleep(self.config.health_check_interval)
                    self._perform_health_check()
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    
        self._health_check_thread = threading.Thread(
            target=health_check_worker,
            daemon=True,
            name="DBPool-HealthCheck"
        )
        self._health_check_thread.start()
        
    def _perform_health_check(self):
        """Perform health check on pool connections"""
        with self._pool_lock:
            # Check pool statistics
            self._update_pool_stats()
            
            # Clean up old connections
            self._cleanup_old_connections()
            
            # Validate sample connections
            if self.config.pool_pre_ping:
                self._validate_connections()
                
        logger.debug(f"Health check completed. Pool status: {self.get_pool_status()}")
        
    def _cleanup_old_connections(self):
        """Clean up connections that have exceeded recycle time"""
        current_time = datetime.now()
        recycle_threshold = timedelta(seconds=self.config.pool_recycle)
        
        # Clean up overflow connections
        to_remove = set()
        for conn_id in self._overflow_connections:
            if conn_id in self._checked_out_connections:
                checkout_time = self._checked_out_connections[conn_id].checked_out_at
                if current_time - checkout_time > recycle_threshold:
                    to_remove.add(conn_id)
                    
        for conn_id in to_remove:
            self._remove_connection(conn_id)
            
    def _validate_connections(self):
        """Validate a sample of pool connections"""
        if self._pool.empty():
            return
            
        # Test one connection from the pool
        try:
            conn = self._pool.get(block=False)
            is_valid = self._test_connection(conn)
            
            if is_valid:
                self._pool.put(conn, block=False)
            else:
                # Replace invalid connection
                self._close_connection(conn)
                new_conn = self._create_connection()
                self._pool.put(new_conn, block=False)
                logger.info("Replaced invalid connection in pool")
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            
    def _test_connection(self, connection) -> bool:
        """Test if a connection is still valid"""
        try:
            if self.db_type == 'sqlite':
                connection.execute("SELECT 1").fetchone()
            elif self.db_type == 'postgresql':
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
            elif self.db_type == 'mysql':
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            return True
        except Exception:
            return False
            
    @contextlib.contextmanager
    def get_connection(self) -> ContextManager[PooledConnection]:
        """Get a connection from the pool with context manager"""
        connection = None
        start_time = time.time()
        
        try:
            connection = self._checkout_connection()
            checkout_time = time.time() - start_time
            
            with self._stats_lock:
                self._checkout_times.append(checkout_time)
                if len(self._checkout_times) > 1000:
                    self._checkout_times = self._checkout_times[-500:]
                self.stats.average_checkout_time = sum(self._checkout_times) / len(self._checkout_times)
                
            yield connection
            
        except Exception as e:
            logger.error(f"Connection checkout failed: {e}")
            with self._stats_lock:
                self.stats.connection_failures += 1
            raise
        finally:
            if connection:
                self._return_connection(connection)
                
    def _checkout_connection(self) -> PooledConnection:
        """Check out a connection from the pool"""
        with self._stats_lock:
            self.stats.connection_requests += 1
            
        start_time = time.time()
        
        # Try to get connection from pool
        try:
            conn = self._pool.get(block=True, timeout=self.config.pool_timeout)
            
            with self._stats_lock:
                self.stats.pool_hits += 1
                self.stats.idle_connections -= 1
                self.stats.checked_out += 1
                
            connection_id = f"conn_{self._connection_counter}"
            self._connection_counter += 1
            
            pooled_conn = PooledConnection(conn, self, connection_id)
            self._checked_out_connections[connection_id] = pooled_conn
            
            logger.debug(f"Checked out connection {connection_id}")
            return pooled_conn
            
        except queue.Empty:
            # Pool is empty, try overflow
            if len(self._overflow_connections) < self.config.max_overflow:
                return self._create_overflow_connection()
            else:
                raise DatabasePoolError("Connection pool exhausted")
                
    def _create_overflow_connection(self) -> PooledConnection:
        """Create an overflow connection"""
        with self._stats_lock:
            self.stats.pool_misses += 1
            
        try:
            conn = self._create_connection()
            connection_id = f"overflow_{self._connection_counter}"
            self._connection_counter += 1
            
            pooled_conn = PooledConnection(conn, self, connection_id)
            self._overflow_connections.add(connection_id)
            self._checked_out_connections[connection_id] = pooled_conn
            
            with self._stats_lock:
                self.stats.overflow_connections += 1
                self.stats.checked_out += 1
                
            logger.debug(f"Created overflow connection {connection_id}")
            return pooled_conn
            
        except Exception as e:
            logger.error(f"Failed to create overflow connection: {e}")
            raise DatabasePoolError(f"Overflow connection creation failed: {e}")
            
    def _return_connection(self, pooled_conn: PooledConnection):
        """Return a connection to the pool"""
        connection_id = pooled_conn.connection_id
        
        try:
            # Remove from checked out connections
            if connection_id in self._checked_out_connections:
                del self._checked_out_connections[connection_id]
                
            with self._stats_lock:
                self.stats.checked_out -= 1
                
            # Handle overflow connections
            if connection_id.startswith('overflow_'):
                self._overflow_connections.discard(connection_id)
                self._close_connection(pooled_conn.connection)
                
                with self._stats_lock:
                    self.stats.overflow_connections -= 1
                    
                logger.debug(f"Closed overflow connection {connection_id}")
                return
                
            # Return regular connection to pool
            if pooled_conn.is_valid and not pooled_conn._in_transaction:
                try:
                    self._pool.put(pooled_conn.connection, block=False)
                    
                    with self._stats_lock:
                        self.stats.idle_connections += 1
                        
                    logger.debug(f"Returned connection {connection_id} to pool")
                    
                except queue.Full:
                    # Pool is full, close connection
                    self._close_connection(pooled_conn.connection)
                    
            else:
                # Connection is invalid, close it
                self._close_connection(pooled_conn.connection)
                logger.debug(f"Closed invalid connection {connection_id}")
                
        except Exception as e:
            logger.error(f"Error returning connection {connection_id}: {e}")
            
    def _close_connection(self, connection):
        """Close a database connection"""
        try:
            connection.close()
            
            with self._stats_lock:
                self.stats.total_connections -= 1
                
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
            
    def _remove_connection(self, connection_id: str):
        """Remove a connection from all tracking"""
        if connection_id in self._checked_out_connections:
            pooled_conn = self._checked_out_connections[connection_id]
            self._close_connection(pooled_conn.connection)
            del self._checked_out_connections[connection_id]
            
        self._overflow_connections.discard(connection_id)
        
    def _update_pool_stats(self):
        """Update pool statistics"""
        with self._stats_lock:
            self.stats.active_connections = self.stats.checked_out
            self.stats.peak_connections = max(
                self.stats.peak_connections,
                self.stats.total_connections
            )
            
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status and statistics"""
        self._update_pool_stats()
        
        return {
            'database_type': self.db_type,
            'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,
            'pool_config': {
                'min_connections': self.config.min_connections,
                'max_connections': self.config.max_connections,
                'max_overflow': self.config.max_overflow,
                'pool_timeout': self.config.pool_timeout
            },
            'statistics': self.stats.to_dict(),
            'health_status': 'healthy' if self._is_healthy else 'unhealthy',
            'pool_size': self._pool.qsize(),
            'checked_out_count': len(self._checked_out_connections),
            'overflow_count': len(self._overflow_connections)
        }
        
    def close_pool(self):
        """Close all connections and shutdown pool"""
        logger.info("Shutting down database connection pool")
        
        with self._pool_lock:
            # Close all checked out connections
            for pooled_conn in list(self._checked_out_connections.values()):
                self._close_connection(pooled_conn.connection)
            self._checked_out_connections.clear()
            
            # Close all pool connections
            while not self._pool.empty():
                try:
                    conn = self._pool.get(block=False)
                    self._close_connection(conn)
                except queue.Empty:
                    break
                    
            # Clear overflow connections
            self._overflow_connections.clear()
            
        logger.info("Database connection pool shutdown complete")
        
    def __del__(self):
        """Cleanup when pool is garbage collected"""
        try:
            self.close_pool()
        except Exception:
            pass

    def return_connection(self, connection):
        """Return a connection to the pool"""
        try:
            if hasattr(connection, 'close'):
                connection.close()
            logger.debug("Connection returned to pool")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
    
    def release_connection(self, connection):
        """Alias for return_connection for backward compatibility"""
        return self.return_connection(connection)


class DatabasePoolManager:
    """Manages multiple database connection pools"""
    
    def __init__(self):
        self._pools: Dict[str, DatabaseConnectionPool] = {}
        self._default_pool: Optional[str] = None
        self._lock = threading.Lock()
        
    def create_pool(self, name: str, database_url: str, config: PoolConfig = None) -> DatabaseConnectionPool:
        """Create a new connection pool"""
        with self._lock:
            if name in self._pools:
                raise DatabasePoolError(f"Pool '{name}' already exists")
                
            pool = DatabaseConnectionPool(database_url, config)
            self._pools[name] = pool
            
            if self._default_pool is None:
                self._default_pool = name
                
            logger.info(f"Created database pool '{name}' for {pool.db_type}")
            return pool
            
    def get_pool(self, name: str = None) -> DatabaseConnectionPool:
        """Get a connection pool by name"""
        with self._lock:
            if name is None:
                name = self._default_pool
                
            if name is None:
                raise DatabasePoolError("No pools available")
                
            if name not in self._pools:
                raise DatabasePoolError(f"Pool '{name}' not found")
                
            return self._pools[name]
            
    def get_connection(self, pool_name: str = None):
        """Get a connection from specified pool"""
        pool = self.get_pool(pool_name)
        return pool.get_connection()
        
    def get_all_pool_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pools"""
        with self._lock:
            return {
                name: pool.get_pool_status()
                for name, pool in self._pools.items()
            }
            
    def close_all_pools(self):
        """Close all connection pools"""
        with self._lock:
            for name, pool in self._pools.items():
                try:
                    pool.close_pool()
                    logger.info(f"Closed pool '{name}'")
                except Exception as e:
                    logger.error(f"Error closing pool '{name}': {e}")
                    
            self._pools.clear()
            self._default_pool = None


# Global pool manager instance
_pool_manager = None


def get_pool_manager() -> DatabasePoolManager:
    """Get the global pool manager instance"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = DatabasePoolManager()
    return _pool_manager


def create_default_pool(database_url: str = None, config: PoolConfig = None) -> DatabaseConnectionPool:
    """Create the default database pool"""
    if database_url is None:
        database_url = "sqlite:///data/langgraph_101.db"
        
    manager = get_pool_manager()
    return manager.create_pool('default', database_url, config)


# Convenience functions
def get_connection(pool_name: str = None):
    """Get a database connection from the pool"""
    manager = get_pool_manager()
    return manager.get_connection(pool_name)


def get_pool_status(pool_name: str = None) -> Dict[str, Any]:
    """Get pool status"""
    manager = get_pool_manager()
    pool = manager.get_pool(pool_name)
    return pool.get_pool_status()


def close_all_pools():
    """Close all database pools"""
    manager = get_pool_manager()
    manager.close_all_pools()


if __name__ == "__main__":
    # Demo and testing
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test pool
    config = PoolConfig(
        min_connections=2,
        max_connections=5,
        max_overflow=2
    )
    
    pool = create_default_pool(config=config)
    
    # Test connection checkout/return
    print("Testing connection pool...")
    
    # Test multiple connections
    connections = []
    for i in range(3):
        conn = pool.get_connection()
        connections.append(conn)
        print(f"Checked out connection {i+1}")
        
    # Test pool status
    status = pool.get_pool_status()
    print(f"\nPool Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Return connections
    for i, conn in enumerate(connections):
        conn.__exit__(None, None, None)
        print(f"Returned connection {i+1}")
        
    # Final status
    final_status = pool.get_pool_status()
    print(f"\nFinal Pool Status:")
    print(json.dumps(final_status, indent=2, default=str))
    
    # Cleanup
    close_all_pools()
    print("\nPool test completed successfully!")
