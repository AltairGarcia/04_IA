#!/usr/bin/env python3
"""
Unified Monitoring System for LangGraph 101 Project

This module consolidates all monitoring functionality into a single, efficient system
that prevents memory leaks and duplicate monitoring instances.

IMPROVEMENTS IMPLEMENTED:
1. Single monitoring instance using proper singleton pattern
2. Efficient memory management with automatic cleanup
3. Unified metrics collection and storage
4. Thread-safe operations with proper cleanup
5. Database optimization and connection pooling
6. Configurable monitoring intervals and thresholds

Author: GitHub Copilot
Date: 2025-05-27
"""

import os
import gc
import sys
import time
import psutil
import threading
import logging
import sqlite3
import weakref
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tracemalloc
import uuid

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('unified_monitoring.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Comprehensive system metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_threads: int
    gc_objects: int
    process_memory_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_available_gb': self.memory_available_gb,
            'disk_percent': self.disk_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'active_threads': self.active_threads,
            'gc_objects': self.gc_objects,
            'process_memory_mb': self.process_memory_mb
        }


@dataclass
class Alert:
    """System alert representation."""
    id: str
    timestamp: datetime
    severity: str  # info, warning, error, critical
    source: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class ThreadSafeSingleton(type):
    """Enhanced thread-safe singleton metaclass."""
    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def reset_instance(cls, target_class):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if target_class in cls._instances:
                instance = cls._instances[target_class]
                if hasattr(instance, 'stop_monitoring'):
                    instance.stop_monitoring()
                del cls._instances[target_class]


class DatabaseManager:
    """Efficient database connection manager with pooling."""
    
    def __init__(self, db_path: str, max_connections: int = 3):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = deque(maxlen=max_connections)
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Metrics table with better indexing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_gb REAL,
                    memory_available_gb REAL,
                    disk_percent REAL,
                    disk_free_gb REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    active_threads INTEGER,
                    gc_objects INTEGER,
                    process_memory_mb REAL
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON system_alerts(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON system_alerts(severity)")
            
            conn.commit()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection from pool."""
        with self._lock:
            if self._connections:
                return self._connections.popleft()
            else:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                return conn
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool."""
        with self._lock:
            if len(self._connections) < self.max_connections:
                self._connections.append(conn)
            else:
                conn.close()
    
    def __enter__(self):
        self.conn = self.get_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'conn'):
            self.return_connection(self.conn)


class UnifiedMonitoringSystem(metaclass=ThreadSafeSingleton):
    """
    Unified monitoring system that consolidates all monitoring functionality.
    
    Features:
    - Single monitoring instance (singleton)
    - Efficient memory management
    - Thread-safe operations
    - Database connection pooling
    - Automatic cleanup
    - Configurable thresholds
    """
    
    def __init__(self, db_path: str = "unified_monitoring.db"):
        """Initialize unified monitoring system."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        
        # Monitoring state
        self.active = False
        self.monitoring_thread = None
        self.cleanup_thread = None
        
        # Data storage (limited size for memory efficiency)
        self.metrics_buffer = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alerts_buffer = deque(maxlen=100)    # Keep last 100 alerts
        
        # Thresholds (configurable)
        self.thresholds = {
            'memory_warning': 80.0,     # Warning at 80%
            'memory_critical': 90.0,    # Critical at 90%
            'cpu_warning': 80.0,        # CPU warning
            'cpu_critical': 95.0,       # CPU critical
            'disk_warning': 85.0,       # Disk warning
            'disk_critical': 95.0,      # Disk critical
            'thread_warning': 50,       # Too many threads
            'gc_objects_warning': 50000 # Too many GC objects
        }
        
        # Monitoring intervals
        self.intervals = {
            'metrics_collection': 30,   # Collect metrics every 30s
            'cleanup_cycle': 300,       # Cleanup every 5 minutes
            'database_optimization': 3600  # Optimize DB every hour
        }
        
        # Registered components
        self.registered_components = weakref.WeakSet()
        self.conversation_registry = weakref.WeakValueDictionary()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="UnifiedMonitor")
        
        # Performance tracking
        self.last_metrics = None
        self.performance_history = deque(maxlen=100)
        
        logger.info("Unified Monitoring System initialized (singleton)")
    
    def start_monitoring(self):
        """Start unified monitoring system."""
        if self.active:
            logger.warning("Monitoring already active")
            return
        
        self.active = True
        
        # Start main monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="UnifiedMonitor-Main"
        )
        self.monitoring_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="UnifiedMonitor-Cleanup"
        )
        self.cleanup_thread.start()
        
        logger.info("Unified monitoring system started")
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        self.active = False
        
        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
          # Shutdown thread pool
        try:
            # Try with timeout parameter (Python 3.9+)
            self.thread_pool.shutdown(wait=True, timeout=10)
        except TypeError:
            # Fallback for older Python versions
            self.thread_pool.shutdown(wait=True)
        
        logger.info("Unified monitoring system stopped")
    
    def register_component(self, component: Any, component_type: str = "unknown"):
        """Register a component for monitoring."""
        self.registered_components.add(component)
        logger.info(f"Registered component: {component_type}")
    
    def register_conversation(self, conversation_id: str, instance: Any):
        """Register a conversation instance."""
        self.conversation_registry[conversation_id] = instance
        logger.debug(f"Registered conversation: {conversation_id}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Thread and GC metrics
            active_threads = threading.active_count()
            gc_objects = len(gc.get_objects())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024**3),
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                active_threads=active_threads,
                gc_objects=gc_objects,
                process_memory_mb=process_memory
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def store_metrics(self, metrics: SystemMetrics):
        """Store metrics in database."""
        try:
            with self.db_manager as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, memory_used_gb,
                        memory_available_gb, disk_percent, disk_free_gb,
                        network_bytes_sent, network_bytes_recv, active_threads,
                        gc_objects, process_memory_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_used_gb,
                    metrics.memory_available_gb,
                    metrics.disk_percent,
                    metrics.disk_free_gb,
                    metrics.network_bytes_sent,
                    metrics.network_bytes_recv,
                    metrics.active_threads,
                    metrics.gc_objects,
                    metrics.process_memory_mb
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and create alerts."""
        alerts = []
        
        # Memory checks
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            alerts.append(self._create_alert(
                'critical', 'memory_monitor',
                f"Critical memory usage: {metrics.memory_percent:.1f}%",
                {'memory_percent': metrics.memory_percent, 'available_gb': metrics.memory_available_gb}
            ))
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            alerts.append(self._create_alert(
                'warning', 'memory_monitor',
                f"High memory usage: {metrics.memory_percent:.1f}%",
                {'memory_percent': metrics.memory_percent, 'available_gb': metrics.memory_available_gb}
            ))
        
        # CPU checks
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append(self._create_alert(
                'critical', 'cpu_monitor',
                f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                {'cpu_percent': metrics.cpu_percent}
            ))
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append(self._create_alert(
                'warning', 'cpu_monitor',
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                {'cpu_percent': metrics.cpu_percent}
            ))
        
        # Disk checks
        if metrics.disk_percent >= self.thresholds['disk_critical']:
            alerts.append(self._create_alert(
                'critical', 'disk_monitor',
                f"Critical disk usage: {metrics.disk_percent:.1f}%",
                {'disk_percent': metrics.disk_percent, 'free_gb': metrics.disk_free_gb}
            ))
        elif metrics.disk_percent >= self.thresholds['disk_warning']:
            alerts.append(self._create_alert(
                'warning', 'disk_monitor',
                f"High disk usage: {metrics.disk_percent:.1f}%",
                {'disk_percent': metrics.disk_percent, 'free_gb': metrics.disk_free_gb}
            ))
        
        # Thread count check
        if metrics.active_threads >= self.thresholds['thread_warning']:
            alerts.append(self._create_alert(
                'warning', 'thread_monitor',
                f"High thread count: {metrics.active_threads}",
                {'active_threads': metrics.active_threads}
            ))
        
        # GC objects check
        if metrics.gc_objects >= self.thresholds['gc_objects_warning']:
            alerts.append(self._create_alert(
                'warning', 'gc_monitor',
                f"High GC object count: {metrics.gc_objects:,}",
                {'gc_objects': metrics.gc_objects}
            ))
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
    
    def _create_alert(self, severity: str, source: str, message: str, details: Dict) -> Alert:
        """Create a new alert."""
        alert_id = f"{source}_{severity}_{int(time.time())}"
        return Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            source=source,
            message=message,
            details=details
        )
    
    def _store_alert(self, alert: Alert):
        """Store alert in database."""
        try:
            self.alerts_buffer.append(alert)
            
            with self.db_manager as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO system_alerts (
                        id, timestamp, severity, source, message, details, resolved, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.severity,
                    alert.source,
                    alert.message,
                    json.dumps(alert.details),
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_cleanup = time.time()
        last_db_optimization = time.time()
        
        while self.active:
            try:
                # Collect metrics
                metrics = self.collect_system_metrics()
                if metrics:
                    self.metrics_buffer.append(metrics)
                    self.last_metrics = metrics
                    
                    # Store in database (async)
                    self.thread_pool.submit(self.store_metrics, metrics)
                    
                    # Check thresholds
                    self.check_thresholds(metrics)
                    
                    # Trigger emergency cleanup if needed
                    if metrics.memory_percent >= self.thresholds['memory_critical']:
                        logger.error(f"EMERGENCY: Memory at {metrics.memory_percent:.1f}%")
                        self.thread_pool.submit(self._emergency_cleanup)
                
                # Periodic cleanup
                current_time = time.time()
                if current_time - last_cleanup >= self.intervals['cleanup_cycle']:
                    self.thread_pool.submit(self._periodic_cleanup)
                    last_cleanup = current_time
                
                # Database optimization
                if current_time - last_db_optimization >= self.intervals['database_optimization']:
                    self.thread_pool.submit(self._optimize_database)
                    last_db_optimization = current_time
                
                time.sleep(self.intervals['metrics_collection'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.active:
            try:
                # Regular garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"GC collected {collected} objects")
                
                # Clean conversation registry
                self._cleanup_conversations()
                
                time.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(120)
    
    def _periodic_cleanup(self):
        """Periodic cleanup operations."""
        try:
            logger.info("Running periodic cleanup")
            
            # Cleanup old metrics from memory
            if len(self.metrics_buffer) > 500:
                for _ in range(200):
                    if self.metrics_buffer:
                        self.metrics_buffer.popleft()
            
            # Cleanup old alerts from memory
            if len(self.alerts_buffer) > 50:
                for _ in range(20):
                    if self.alerts_buffer:
                        self.alerts_buffer.popleft()
            
            # Database cleanup
            with self.db_manager as conn:
                cursor = conn.cursor()
                
                # Keep only last 10,000 metrics
                cursor.execute("""
                    DELETE FROM system_metrics 
                    WHERE id NOT IN (
                        SELECT id FROM system_metrics 
                        ORDER BY timestamp DESC LIMIT 10000
                    )
                """)
                
                # Keep only last 1,000 alerts
                cursor.execute("""
                    DELETE FROM system_alerts 
                    WHERE id NOT IN (
                        SELECT id FROM system_alerts 
                        ORDER BY timestamp DESC LIMIT 1000
                    )
                """)
                
                conn.commit()
                
            logger.info("Periodic cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup for critical memory situations."""
        try:
            logger.error("EMERGENCY CLEANUP ACTIVATED")
            
            # Force multiple GC cycles
            for i in range(3):
                collected = gc.collect()
                logger.info(f"Emergency GC cycle {i+1}: collected {collected} objects")
            
            # Clear buffers
            self.metrics_buffer.clear()
            self.alerts_buffer.clear()
            self.performance_history.clear()
            
            # Clear conversation registry
            self.conversation_registry.clear()
            
            # Emergency database cleanup
            try:
                with self.db_manager as conn:
                    cursor = conn.cursor()
                    # Keep only last 1000 records
                    cursor.execute("""
                        DELETE FROM system_metrics 
                        WHERE id NOT IN (
                            SELECT id FROM system_metrics 
                            ORDER BY timestamp DESC LIMIT 1000
                        )
                    """)
                    cursor.execute("VACUUM")
                    conn.commit()
            except Exception as db_error:
                logger.error(f"Emergency database cleanup failed: {db_error}")
            
            logger.error("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in emergency cleanup: {e}")
    
    def _cleanup_conversations(self):
        """Clean up old conversation instances."""
        # Conversations are automatically cleaned up by WeakValueDictionary
        # when they're no longer referenced elsewhere
        count = len(self.conversation_registry)
        if count > 100:
            logger.warning(f"High conversation count: {count}")
    
    def _optimize_database(self):
        """Optimize database performance."""
        try:
            logger.info("Running database optimization")
            
            with self.db_manager as conn:
                cursor = conn.cursor()
                
                # Update statistics
                cursor.execute("ANALYZE")
                
                # Vacuum if needed (only if significant deletions occurred)
                cursor.execute("PRAGMA auto_vacuum = INCREMENTAL")
                cursor.execute("PRAGMA incremental_vacuum")
                
                conn.commit()
                
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.last_metrics:
            return {"error": "No metrics available"}
        
        metrics = self.last_metrics
        recent_alerts = list(self.alerts_buffer)[-5:]  # Last 5 alerts
        
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_available_gb": metrics.memory_available_gb,
            "disk_percent": metrics.disk_percent,
            "disk_free_gb": metrics.disk_free_gb,
            "active_threads": metrics.active_threads,
            "gc_objects": metrics.gc_objects,
            "process_memory_mb": metrics.process_memory_mb,
            "registered_components": len(self.registered_components),
            "conversations": len(self.conversation_registry),
            "recent_alerts": [
                {
                    "severity": alert.severity,
                    "source": alert.source,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ],
            "status": (
                "CRITICAL" if metrics.memory_percent >= self.thresholds['memory_critical'] else
                "WARNING" if metrics.memory_percent >= self.thresholds['memory_warning'] else
                "OK"
            )
        }
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics.to_dict()
            for metrics in self.metrics_buffer
            if metrics.timestamp >= cutoff_time
        ]
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update monitoring thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated thresholds: {new_thresholds}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'active') and self.active:
            self.stop_monitoring()


# Global instance
_unified_monitor = None

def get_unified_monitor() -> UnifiedMonitoringSystem:
    """Get the global unified monitoring instance."""
    global _unified_monitor
    if _unified_monitor is None:
        _unified_monitor = UnifiedMonitoringSystem()
    return _unified_monitor

def start_unified_monitoring():
    """Start the unified monitoring system."""
    monitor = get_unified_monitor()
    monitor.start_monitoring()

def stop_unified_monitoring():
    """Stop the unified monitoring system."""
    monitor = get_unified_monitor()
    monitor.stop_monitoring()

def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    monitor = get_unified_monitor()
    return monitor.get_current_status()

def register_component(component: Any, component_type: str = "unknown"):
    """Register a component for monitoring."""
    monitor = get_unified_monitor()
    monitor.register_component(component, component_type)

def register_conversation(conversation_id: str, instance: Any):
    """Register a conversation instance."""
    monitor = get_unified_monitor()
    monitor.register_conversation(conversation_id, instance)


if __name__ == "__main__":
    # Test the unified monitoring system
    print("Testing Unified Monitoring System...")
    
    # Start monitoring
    start_unified_monitoring()
    
    try:
        # Let it run for a few seconds
        time.sleep(10)
        
        # Get status
        status = get_system_status()
        print("\nSystem Status:")
        for key, value in status.items():
            if key != "recent_alerts":
                print(f"  {key}: {value}")
        
        if "recent_alerts" in status and status["recent_alerts"]:
            print("\nRecent Alerts:")
            for alert in status["recent_alerts"]:
                print(f"  [{alert['severity']}] {alert['source']}: {alert['message']}")
        
        # Test component registration
        register_component("test_component", "test")
        register_conversation("test_conv_1", {"test": "data"})
        
        print("\nTest completed successfully!")
        
    finally:
        stop_unified_monitoring()
        print("Unified monitoring system stopped")
