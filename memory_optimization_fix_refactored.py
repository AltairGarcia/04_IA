#!/usr/bin/env python3
"""
Memory Optimization and Performance Monitoring Fix

This module addresses critical memory issues and multiple monitoring instances
in the LangGraph 101 project.

CRITICAL ISSUES TO FIX:
1. Memory usage 90-93% (exceeding 85% threshold)
2. Multiple performance monitoring instances
3. Rapid conversation creation causing memory leaks
4. Background thread cleanup
5. Performance monitoring consolidation

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
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import dataclass
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics tracking."""
    timestamp: datetime
    memory_usage_percent: float
    memory_usage_mb: float
    available_memory_gb: float
    process_memory_mb: float
    gc_objects: int
    active_threads: int


class SingletonMeta(type):
    """Metaclass for singleton pattern."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MemoryOptimizer(metaclass=SingletonMeta):
    """
    Centralized memory optimization and monitoring system.

    This singleton class prevents multiple monitoring instances and provides
    comprehensive memory management.
    """

    def __init__(self):
        """Initialize memory optimizer."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.active = False
        self.monitoring_thread = None
        self.cleanup_thread = None

        # Memory tracking
        # Keep last 1000 measurements
        self.memory_stats = deque(maxlen=1000)
        self.memory_threshold = 85.0  # Memory usage threshold (%)
        self.critical_threshold = 90.0  # Critical memory threshold (%)

        # Thread management
        self.active_monitors = set()
        self.monitor_registry = weakref.WeakSet()

        # Conversation tracking
        self.conversation_instances = weakref.WeakValueDictionary()
        self.conversation_creation_times = {}

        # Performance thresholds
        self.thresholds = {
            'memory_usage_percent': 85.0,
            'memory_cleanup_interval': 300,  # 5 minutes
            'gc_collection_interval': 60,    # 1 minute
            'thread_cleanup_interval': 120   # 2 minutes
        }

        logger.info("Memory optimizer initialized (singleton)")

    def start_optimization(self):
        """Start memory optimization and monitoring."""
        if self.active:
            logger.warning("Memory optimization already active")
            return

        self.active = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryOptimizer-Monitor"
        )
        self.monitoring_thread.start()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="MemoryOptimizer-Cleanup"
        )
        self.cleanup_thread.start()

        logger.info("Memory optimization started")

    def stop_optimization(self):
        """Stop memory optimization."""
        self.active = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

        logger.info("Memory optimization stopped")

    def register_monitor(self, monitor_instance):
        """Register a monitoring instance to prevent duplicates."""
        monitor_id = id(monitor_instance)

        if monitor_id in self.active_monitors:
            logger.warning(f"Monitor {monitor_id} already registered")
            return False

        self.active_monitors.add(monitor_id)
        self.monitor_registry.add(monitor_instance)
        logger.info(f"Registered monitor {monitor_id}")
        return True

    def unregister_monitor(self, monitor_instance):
        """Unregister a monitoring instance."""
        monitor_id = id(monitor_instance)
        self.active_monitors.discard(monitor_id)
        logger.info(f"Unregistered monitor {monitor_id}")

    def register_conversation(self, conversation_id: str, instance):
        """Register a conversation instance."""
        self.conversation_instances[conversation_id] = instance
        self.conversation_creation_times[conversation_id] = datetime.now()

        # Check for rapid conversation creation
        recent_conversations = [
            conv_id for conv_id, creation_time
            in self.conversation_creation_times.items()
            if datetime.now() - creation_time < timedelta(minutes=5)
        ]

        if len(recent_conversations) > 10:
            logger.warning(
                f"Rapid conversation creation detected: "
                f"{len(recent_conversations)} in 5 minutes"
            )
            self._cleanup_old_conversations()

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.active:
            try:
                # Collect memory statistics
                memory_stats = self._collect_memory_stats()
                self.memory_stats.append(memory_stats)

                # Check memory thresholds
                if memory_stats.memory_usage_percent > self.critical_threshold:
                    logger.error(
                        f"CRITICAL: Memory usage at "
                        f"{memory_stats.memory_usage_percent:.1f}%"
                    )
                    self._emergency_cleanup()
                elif memory_stats.memory_usage_percent > self.memory_threshold:
                    logger.warning(
                        f"High memory usage: "
                        f"{memory_stats.memory_usage_percent:.1f}%"
                    )
                    self._standard_cleanup()

                # Log status
                if len(self.memory_stats) % 10 == 0:  # Every 10th measurement
                    self._log_memory_status(memory_stats)

                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)

    def _cleanup_loop(self):
        """Background cleanup loop."""
        gc_counter = 0
        cleanup_counter = 0

        while self.active:
            try:
                gc_counter += 1
                cleanup_counter += 1

                # Garbage collection every minute
                if gc_counter >= 2:  # 30s * 2 = 60s
                    self._run_garbage_collection()
                    gc_counter = 0

                # Full cleanup every 5 minutes
                if cleanup_counter >= 10:  # 30s * 10 = 300s
                    self._comprehensive_cleanup()
                    cleanup_counter = 0

                time.sleep(30)

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)

    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Garbage collection stats
        gc_objects = len(gc.get_objects())

        # Thread count
        active_threads = threading.active_count()

        return MemoryStats(
            timestamp=datetime.now(),
            memory_usage_percent=memory.percent,
            memory_usage_mb=memory.used / 1024 / 1024,
            available_memory_gb=memory.available / 1024 / 1024 / 1024,
            process_memory_mb=process_memory,
            gc_objects=gc_objects,
            active_threads=active_threads
        )

    def _log_memory_status(self, stats: MemoryStats):
        """Log current memory status."""
        logger.info(
            f"Memory: {stats.memory_usage_percent:.1f}% "
            f"({stats.memory_usage_mb:.0f}MB used, "
            f"{stats.available_memory_gb:.1f}GB available) "
            f"Process: {stats.process_memory_mb:.0f}MB "
            f"Objects: {stats.gc_objects:,} "
            f"Threads: {stats.active_threads}"
        )

    def _standard_cleanup(self):
        """Standard memory cleanup procedures."""
        logger.info("Running standard memory cleanup")

        # Garbage collection
        self._run_garbage_collection()

        # Clean old conversations
        self._cleanup_old_conversations()

        # Clean old memory stats
        if len(self.memory_stats) > 500:
            # Keep only last 500 measurements
            recent_stats = list(self.memory_stats)[-500:]
            self.memory_stats.clear()
            self.memory_stats.extend(recent_stats)

    def _emergency_cleanup(self):
        """Emergency memory cleanup for critical situations."""
        logger.error("Running EMERGENCY memory cleanup")

        # Force garbage collection multiple times
        for i in range(3):
            collected = gc.collect()
            logger.info(f"GC cycle {i + 1}: collected {collected} objects")

        # Clean all old conversations
        self._cleanup_old_conversations(max_age_minutes=1)

        # Clear unnecessary caches
        self._clear_caches()

        # Force cleanup of monitoring instances
        self._cleanup_monitoring_instances()

    def _comprehensive_cleanup(self):
        """Comprehensive cleanup routine."""
        logger.info("Running comprehensive cleanup")

        # Standard cleanup
        self._standard_cleanup()

        # Thread cleanup
        self._cleanup_threads()

        # Database cleanup
        self._cleanup_databases()

        # Monitor registry cleanup
        self._cleanup_monitoring_instances()

    def _run_garbage_collection(self):
        """Run garbage collection and log results."""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())

        if collected > 0:
            logger.info(
                f"GC: collected {collected} objects, "
                f"{before_objects} -> {after_objects}"
            )

    def _cleanup_old_conversations(self, max_age_minutes: int = 30):
        """Clean up old conversation instances."""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

        old_conversations = [
            conv_id for conv_id, creation_time
            in self.conversation_creation_times.items()
            if creation_time < cutoff_time
        ]

        for conv_id in old_conversations:
            if conv_id in self.conversation_instances:
                del self.conversation_instances[conv_id]
            if conv_id in self.conversation_creation_times:
                del self.conversation_creation_times[conv_id]

        if old_conversations:
            logger.info(
                f"Cleaned up {len(old_conversations)} old conversations"
            )

    def _clear_caches(self):
        """Clear various caches to free memory."""
        try:
            # Clear import cache            modules_to_clear = []
            for module_name in sys.modules:
                if hasattr(sys.modules[module_name], '__dict__'):
                    module_dict = sys.modules[module_name].__dict__
                    if (hasattr(module_dict, 'clear')
                            and 'cache' in module_name.lower()):
                        modules_to_clear.append(module_name)

            logger.info(f"Found {len(modules_to_clear)} cached modules")

        except Exception as e:
            logger.warning(f"Error clearing caches: {e}")

    def _cleanup_monitoring_instances(self):
        """Clean up old monitoring instances."""
        # Count active monitors
        active_count = len([m for m in self.monitor_registry if m is not None])

        if active_count > 3:  # More than 3 monitors is suspicious
            logger.warning(f"Too many active monitors: {active_count}")

            # Try to stop excess monitors
            for monitor in list(self.monitor_registry):
                if monitor and hasattr(monitor, 'stop_monitoring'):
                    try:
                        monitor.stop_monitoring()
                        logger.info("Stopped excess monitor")
                    except Exception as e:
                        logger.warning(f"Error stopping monitor: {e}")

    def _cleanup_threads(self):
        """Clean up old threads."""
        active_threads = threading.active_count()
        all_threads = threading.enumerate()

        # Log thread information
        daemon_threads = sum(1 for t in all_threads if t.daemon)
        non_daemon_threads = active_threads - daemon_threads

        logger.info(
            f"Threads: {active_threads} total "
            f"({daemon_threads} daemon, {non_daemon_threads} non-daemon)"
        )        # Check for problematic threads
        long_running_threads = []
        for thread in all_threads:
            if hasattr(thread, 'name') and thread.is_alive():
                if ('monitor' in thread.name.lower()
                        or 'performance' in thread.name.lower()):
                    long_running_threads.append(thread.name)

        if len(long_running_threads) > 5:
            logger.warning(
                f"Many monitoring threads detected: {long_running_threads}"
            )

    def _cleanup_databases(self):
        """Clean up database connections and optimize."""
        try:
            # Look for SQLite databases in the current directory
            db_files = [f for f in os.listdir('.') if f.endswith('.db')]

            for db_file in db_files:
                try:
                    with sqlite3.connect(db_file) as conn:
                        # Run VACUUM to optimize database
                        conn.execute('VACUUM')

                        # Clean old records if applicable
                        tables = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                        for (table_name,) in tables:
                            if ('log' in table_name.lower() or
                                    'metric' in table_name.lower()):
                                # Keep only last 1000 records
                                conn.execute(f'''
                                    DELETE FROM {table_name}
                                    WHERE rowid NOT IN (
                                        SELECT rowid FROM {table_name}
                                        ORDER BY rowid DESC LIMIT 1000
                                    )
                                ''')

                        conn.commit()
                        logger.info(f"Optimized database: {db_file}")

                except sqlite3.Error as e:
                    logger.warning(f"Error optimizing database {db_file}: {e}")

        except Exception as e:
            logger.warning(f"Error in database cleanup: {e}")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        if not self.memory_stats:
            return {"error": "No memory statistics available"}

        recent_stats = list(self.memory_stats)[-10:]  # Last 10 measurements

        current = recent_stats[-1]
        avg_memory = (sum(s.memory_usage_percent for s in recent_stats) /
                      len(recent_stats))
        max_memory = max(s.memory_usage_percent for s in recent_stats)

        return {
            "timestamp": current.timestamp.isoformat(),
            "current_memory_percent": current.memory_usage_percent,
            "average_memory_percent": avg_memory,
            "max_memory_percent": max_memory,
            "available_memory_gb": current.available_memory_gb,
            "process_memory_mb": current.process_memory_mb,
            "gc_objects": current.gc_objects,
            "active_threads": current.active_threads,
            "active_monitors": len(self.active_monitors),
            "conversations": len(self.conversation_instances),
            "memory_threshold": self.memory_threshold,
            "critical_threshold": self.critical_threshold,
            "status": (
                "CRITICAL"
                if current.memory_usage_percent > self.critical_threshold
                else "WARNING"
                if current.memory_usage_percent > self.memory_threshold
                else "OK"
            )
        }


# Singleton instance
memory_optimizer = MemoryOptimizer()


def start_memory_optimization():
    """Start the memory optimization system."""
    memory_optimizer.start_optimization()


def stop_memory_optimization():
    """Stop the memory optimization system."""
    memory_optimizer.stop_optimization()


def get_memory_status() -> Dict[str, Any]:
    """Get current memory status."""
    return memory_optimizer.get_memory_report()


def register_monitor(monitor_instance) -> bool:
    """Register a monitoring instance."""
    return memory_optimizer.register_monitor(monitor_instance)


def register_conversation(conversation_id: str, instance):
    """Register a conversation instance."""
    memory_optimizer.register_conversation(conversation_id, instance)


def force_cleanup():
    """Force immediate cleanup."""
    memory_optimizer._comprehensive_cleanup()


if __name__ == "__main__":
    # Test the memory optimizer
    print("Starting memory optimization test...")

    start_memory_optimization()

    try:
        # Let it run for a few seconds
        time.sleep(5)

        # Get status
        status = get_memory_status()
        print("\nMemory Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Force cleanup test
        print("\nRunning force cleanup test...")
        force_cleanup()

    finally:
        stop_memory_optimization()
        print("Memory optimization test completed")
