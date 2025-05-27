#!/usr/bin/env python3
"""
Quick fix for SQLite threading issues in unified monitoring system
"""

import sqlite3
import threading
from contextlib import contextmanager

class ThreadSafeDBManager:
    """Thread-safe database manager that creates fresh connections per operation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        try:
            # System metrics table
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
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON system_alerts(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON system_alerts(severity)")
            
            conn.commit()
        finally:
            conn.close()
    
    @contextmanager
    def get_connection(self):
        """Context manager for thread-safe database operations."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_write(self, query: str, params: tuple = ()):
        """Execute a write operation safely."""
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
    
    def execute_read(self, query: str, params: tuple = ()):
        """Execute a read operation safely."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

# Function to patch the existing unified monitoring system
def patch_monitoring_system():
    """Apply the database fix to the existing monitoring system."""
    try:
        from unified_monitoring_system import UnifiedMonitoringSystem
        
        # Get the singleton instance
        monitor = UnifiedMonitoringSystem()
        
        # Replace the database manager with our thread-safe version
        monitor.db_manager = ThreadSafeDBManager(monitor.db_path)
        
        # Patch the store_metrics method
        def safe_store_metrics(self, metrics):
            """Thread-safe metrics storage."""
            try:
                query = """
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, memory_used_gb,
                        memory_available_gb, disk_percent, disk_free_gb,
                        network_bytes_sent, network_bytes_recv, active_threads,
                        gc_objects, process_memory_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
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
                )
                self.db_manager.execute_write(query, params)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error storing metrics: {e}")
        
        # Patch the _store_alert method
        def safe_store_alert(self, alert):
            """Thread-safe alert storage."""
            try:
                import json
                query = """
                    INSERT OR REPLACE INTO system_alerts (
                        id, timestamp, severity, source, message, details, resolved, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.severity,
                    alert.source,
                    alert.message,
                    json.dumps(alert.details),
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                )
                self.db_manager.execute_write(query, params)
                self.alerts_buffer.append(alert)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error storing alert: {e}")
        
        # Apply the patches
        monitor.store_metrics = safe_store_metrics.__get__(monitor, UnifiedMonitoringSystem)
        monitor._store_alert = safe_store_alert.__get__(monitor, UnifiedMonitoringSystem)
        
        print("✅ Database threading fixes applied successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply database fixes: {e}")
        return False

if __name__ == "__main__":
    patch_monitoring_system()
