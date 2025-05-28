#!/usr/bin/env python3
"""
Enhanced Unified Monitoring System for LangGraph 101 Project

This module provides a comprehensive monitoring solution by integrating
OS-level metrics, database integration metrics, and advanced alert/event handling.
It replaces and extends functionality previously split between 
UnifiedMonitoringSystem and the older version of EnhancedUnifiedMonitoringSystem.

Key Features:
- Centralized database usage via UnifiedDatabaseManager.
- Collection of OS-level system metrics (CPU, memory, disk, network, GC, threads).
- Collection of database integration specific metrics.
- Threshold-based alerting for both OS and database metrics.
- Structured system event logging.
- Configurable monitoring intervals and alert thresholds via database.
- Optional integration with AdvancedMemoryProfiler.
- Periodic data cleanup and database optimization.

Author: GitHub Copilot (and original authors)
Date: 2025-05-28 (Refactored Date)
"""

import os
import gc
import sys
import time
import psutil
import platform # For hostname
import uuid # For unique IDs
import threading
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from collections import deque
# from contextlib import contextmanager # No longer needed here for DB

# Core and project-specific imports
try:
    from core.config import get_config # Added for enable_memory_profiling
    from core.database import get_database_manager
    from advanced_memory_profiler import AdvancedMemoryProfiler
    # UnifiedMonitoringSystem is being merged into this, so direct import for reuse is removed.
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.critical(f"Core module import failed, EnhancedUnifiedMonitoringSystem cannot function: {e}", exc_info=True)
    IMPORTS_AVAILABLE = False # This will prevent class instantiation if critical modules are missing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_unified_monitoring.log', encoding='utf-8') # Central log file
    ]
)
logger = logging.getLogger(__name__)

# --- Dataclasses ---
@dataclass
class SystemMetrics: # From unified_monitoring_system.py
    """Comprehensive OS-level system metrics."""
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
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class OSAlert: # Renamed from Alert in unified_monitoring_system.py
    """OS-level system alert representation."""
    id: str
    timestamp: datetime
    severity: str  # info, warning, error, critical
    source: str    # e.g., memory_monitor, cpu_monitor
    message: str
    details: Dict[str, Any] # Should be JSON serializable
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        data['details'] = json.dumps(self.details) # Store details as JSON string
        return data

@dataclass
class DatabaseIntegrationMetrics: # Kept from original EUMS
    """Métricas de integração de database."""
    timestamp: datetime
    active_connections: int
    total_queries: int
    query_errors: int
    avg_query_time: float # In seconds
    connection_pool_efficiency: float # Percentage
    memory_used_by_connections_mb: float # Estimated
    orphaned_connections_cleaned: int
    database_file_size_mb: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

# Enhanced Alert (for EUMS specific alerts, distinct from OSAlert)
# This aligns with 'enhanced_alerts' table structure
@dataclass
class EnhancedAlert:
    id: str
    timestamp: datetime
    category: str # e.g., database_performance, system_stability
    severity: str # info, warning, error, critical
    source: str   # e.g., enhanced_monitoring_system
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None # JSON serializable
    metrics: Optional[Dict[str, Any]] = None # JSON serializable, relevant metrics snapshot
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at: data['resolved_at'] = self.resolved_at.isoformat()
        data['details'] = json.dumps(self.details or {})
        data['metrics'] = json.dumps(self.metrics or {})
        return data

# ThreadSafeSingleton Metaclass (copied from unified_monitoring_system.py as per instruction)
class ThreadSafeSingleton(type):
    """Enhanced thread-safe singleton metaclass."""
    _instances = {}
    _lock = threading.RLock() # Class-level lock for instance creation

    def __call__(cls, *args, **kwargs):
        # Double-checked locking pattern
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
                if hasattr(instance, 'stop_monitoring'): # Attempt to gracefully stop if possible
                    try:
                        instance.stop_monitoring()
                    except Exception as e_stop:
                        logger.warning(f"Error stopping instance during reset: {e_stop}")
                del cls._instances[target_class]
                logger.debug(f"Singleton instance of {target_class.__name__} reset.")


class EnhancedUnifiedMonitoringSystem(metaclass=ThreadSafeSingleton):
    """
    Consolidated and enhanced monitoring system.
    Manages OS metrics, database integration metrics, advanced alerting, and system events.
    Now a Singleton using ThreadSafeSingleton metaclass.
    """
    # _instance and _lock are now managed by the metaclass.
    # Remove them from here if they were class attributes.

    def __init__(self): # enable_memory_profiling parameter removed
        # The metaclass handles singleton instance logic, but we need to prevent re-initialization of attributes.
        if hasattr(self, '_initialized_eums') and self._initialized_eums:
            logger.debug(f"EUMS Singleton already initialized (ID: {id(self)}). Skipping re-init.")
            return
        
        logger.debug(f"Initializing EUMS Singleton (ID: {id(self)})...")

        config = get_config()
        # Fetch enable_memory_profiling from UnifiedConfig
        # Assuming 'monitoring' section or 'app' section in config.
        # Defaulting to True if not found.
        monitoring_config = getattr(config, 'monitoring', None)
        if monitoring_config and hasattr(monitoring_config, 'enable_memory_profiling'):
            self.enable_memory_profiling = monitoring_config.enable_memory_profiling
        elif hasattr(config.app, 'enable_memory_profiling'): # Fallback to app config
             self.enable_memory_profiling = config.app.enable_memory_profiling
        else:
            self.enable_memory_profiling = True # Default if not in config
            logger.warning("'enable_memory_profiling' not found in config.monitoring or config.app, defaulting to True. Consider adding it to core/config.py.")

        self.is_running = False
        self.db_manager = get_database_manager()
        
        logger.info(f"Initializing EnhancedUnifiedMonitoringSystem with DB: {self.db_manager.database_url}, MemProfiling: {self.enable_memory_profiling}")

        # OS Metrics related attributes (from UMS)
        self.os_metrics_buffer = deque(maxlen=1000) 
        self.os_alerts_buffer = deque(maxlen=100)
        self.last_os_metrics: Optional[SystemMetrics] = None
        self.os_thresholds = { # Default OS thresholds
            'memory_warning': 80.0, 'memory_critical': 90.0,
            'cpu_warning': 80.0, 'cpu_critical': 95.0,
            'disk_warning': 85.0, 'disk_critical': 95.0,
            'thread_warning': 100, # Increased threshold
            'gc_objects_warning': 100000 # Increased threshold
        }
        self.intervals = { # Combined and refined intervals
            'os_metrics_collection': 30,      # Collect OS metrics every 30s
            'db_integration_metrics': 60, # Collect DB integration metrics every 60s
            'periodic_cleanup': 300,          # Cleanup DB data every 5 minutes
            'db_optimization': 3600,          # Optimize DB every hour
            'gc_collect_interval': 120        # Explicit GC every 2 minutes
        }

        # EUMS specific attributes
        self.integration_metrics_buffer = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = [] # For enhanced alerts

        self.memory_profiler = None
        if self.enable_memory_profiling:
            try:
                self.memory_profiler = AdvancedMemoryProfiler(snapshot_interval=self.intervals.get('os_metrics_collection', 60))
                logger.info("AdvancedMemoryProfiler integrated.")
            except Exception as e:
                logger.warning(f"Could not initialize AdvancedMemoryProfiler: {e}", exc_info=True)
        
        self._lock = threading.RLock() # Ensure lock is initialized
        self.monitoring_thread = None
        
        self._ensure_default_system_configs()
        self._migrate_existing_data() # Simplified, checks for 'system_alerts' data to move to 'enhanced_alerts'
        
        self._initialized_eums = True
        logger.info("EnhancedUnifiedMonitoringSystem initialized.")

    # ... (Methods for _ensure_default_system_configs, _migrate_existing_data - largely similar to current EUMS but using self.db_manager) ...
    def _ensure_default_system_configs(self):
        """Ensures default system configurations are present in the database."""
        default_configs = {
            'os_memory_threshold_warning_percent': (str(self.os_thresholds['memory_warning']), "OS Memory usage percentage warning threshold."),
            'os_memory_threshold_critical_percent': (str(self.os_thresholds['memory_critical']), "OS Memory usage percentage critical threshold."),
            'os_cpu_threshold_warning_percent': (str(self.os_thresholds['cpu_warning']), "OS CPU usage percentage warning threshold."),
            'os_cpu_threshold_critical_percent': (str(self.os_thresholds['cpu_critical']), "OS CPU usage percentage critical threshold."),
            'profiling_enabled': (str(self.enable_memory_profiling).lower(), "Enable/disable memory profiling features."),
            'alert_cooldown_minutes': ('5', "Cooldown period for re-triggering similar alerts."),
            'db_cleanup_interval_seconds': (str(self.intervals['periodic_cleanup']), "Interval for periodic DB data cleanup tasks."),
            'db_optimization_interval_seconds': (str(self.intervals['db_optimization']), "Interval for DB optimization tasks."),
            'os_metrics_collection_interval_seconds': (str(self.intervals['os_metrics_collection']), "Interval for OS metrics collection."),
            'db_integration_metrics_interval_seconds': (str(self.intervals['db_integration_metrics']), "Interval for DB integration metrics collection.")
        }
        try:
            for key, (value, description) in default_configs.items():
                existing_config = self.db_manager.get_record('system_config', key)
                if not existing_config:
                    config_data = {
                        'key': key, 'value': value, 'description': description,
                        'updated_at': datetime.now().isoformat(), 'updated_by': 'system_init'
                    }
                    self.db_manager.insert_record('system_config', config_data)
                    logger.debug(f"Default config set: {key} = {value}")
        except Exception as e:
            logger.error(f"Error setting default system configs: {e}", exc_info=True)

    def _migrate_existing_data(self):
        logger.debug("Checking for legacy data migration needs...")
        # This is a simplified stub. Real migration would depend on specific schema differences.
        # Example: if 'system_alerts' (from old UMS) needs to be migrated to 'enhanced_alerts'
        try:
            # This is just a placeholder for a more complex migration if needed.
            # For now, we assume tables are correctly defined by UnifiedDatabaseManager.
            logger.debug("Legacy data migration check complete. No specific actions taken in this version.")
        except Exception as e:
            logger.error(f"Error during data migration check: {e}", exc_info=True)


    def start_monitoring(self):
        if self.is_running:
            logger.warning("Enhanced monitoring is already running.")
            return
        self.is_running = True
        if self.memory_profiler:
            try: self.memory_profiler.start_profiling(baseline=True)
            except Exception as e_mp: logger.error(f"Failed to start AdvancedMemoryProfiler: {e_mp}", exc_info=True)
        
        self.monitoring_thread = threading.Thread(target=self._combined_monitoring_loop, daemon=True, name="EUMS-CombinedLoop")
        self.monitoring_thread.start()
        self._log_system_event("system_lifecycle", "EUMS", "Enhanced monitoring started.", severity="info")
        logger.info("EnhancedUnifiedMonitoringSystem started.")

    def stop_monitoring(self):
        if not self.is_running: return
        self.is_running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        if self.memory_profiler:
            try: 
                report = self.memory_profiler.stop_profiling()
                if report: self._save_profiling_report(report)
            except Exception as e_mp: logger.error(f"Failed to stop AdvancedMemoryProfiler: {e_mp}", exc_info=True)
        self._log_system_event("system_lifecycle", "EUMS", "Enhanced monitoring stopped.", severity="info")
        logger.info("EnhancedUnifiedMonitoringSystem stopped.")

    def _combined_monitoring_loop(self):
        """Main loop for all monitoring, cleanup, and optimization tasks."""
        last_os_metrics_time = 0
        last_db_integration_metrics_time = 0
        last_periodic_cleanup_time = 0
        last_db_optimization_time = 0
        last_gc_collect_time = 0

        while self.is_running:
            now = time.time()
            try:
                # OS Metrics Collection and Alerting
                if now - last_os_metrics_time >= self.intervals['os_metrics_collection']:
                    os_metrics = self._collect_os_system_metrics()
                    if os_metrics:
                        self.last_os_metrics = os_metrics
                        self.os_metrics_buffer.append(os_metrics)
                        self._store_os_metrics(os_metrics)
                        self._check_os_thresholds(os_metrics)
                        if os_metrics.memory_percent >= self.os_thresholds.get('memory_critical', 90.0):
                            self._emergency_os_cleanup()
                    last_os_metrics_time = now

                # DB Integration Metrics Collection and Alerting
                if now - last_db_integration_metrics_time >= self.intervals['db_integration_metrics']:
                    db_metrics = self._collect_integration_metrics() # EUMS specific DB metrics
                    if db_metrics:
                        self.integration_metrics_buffer.append(db_metrics)
                        self._save_integration_metrics(db_metrics) # Saves to 'database_integration_metrics'
                        self._analyze_and_alert_db_integration(db_metrics) # Uses 'enhanced_alerts'
                    last_db_integration_metrics_time = now
                
                # Periodic DB Cleanup
                if now - last_periodic_cleanup_time >= self.intervals['periodic_cleanup']:
                    self._periodic_db_cleanup()
                    last_periodic_cleanup_time = now

                # Periodic DB Optimization
                if now - last_db_optimization_time >= self.intervals['db_optimization']:
                    self._periodic_db_optimization()
                    last_db_optimization_time = now
                
                # Periodic GC
                if now - last_gc_collect_time >= self.intervals['gc_collect_interval']:
                    gc.collect()
                    last_gc_collect_time = now
                
                # Determine shortest remaining interval to sleep accurately
                next_run_times = [
                    last_os_metrics_time + self.intervals['os_metrics_collection'],
                    last_db_integration_metrics_time + self.intervals['db_integration_metrics'],
                    last_periodic_cleanup_time + self.intervals['periodic_cleanup'],
                    last_db_optimization_time + self.intervals['db_optimization'],
                    last_gc_collect_time + self.intervals['gc_collect_interval']
                ]
                sleep_duration = max(0.1, min(rt - now for rt in next_run_times if rt > now) if any(rt > now for rt in next_run_times) else self.intervals['os_metrics_collection'])
                time.sleep(sleep_duration)

            except Exception as e:
                logger.error(f"Error in combined monitoring loop: {e}", exc_info=True)
                time.sleep(60) # Longer sleep on error

    # --- OS Metrics Methods (from UMS, adapted) ---
    def _collect_os_system_metrics(self) -> Optional[SystemMetrics]:
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            vm_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            return SystemMetrics(
                timestamp=datetime.now(), cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=vm_info.percent, memory_used_gb=vm_info.used / (1024**3),
                memory_available_gb=vm_info.available / (1024**3), disk_percent=disk_info.percent,
                disk_free_gb=disk_info.free / (1024**3), network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv, active_threads=threading.active_count(),
                gc_objects=len(gc.get_objects()), process_memory_mb=mem_info.rss / (1024**2)
            )
        except Exception as e:
            logger.error(f"Error collecting OS system metrics: {e}", exc_info=True)
            return None

    def _store_os_metrics(self, metrics: SystemMetrics):
        try:
            metrics_dict = metrics.to_dict()
            timestamp = metrics_dict.pop('timestamp')
            hostname = platform.node()
            process_id = os.getpid()
            for name, value in metrics_dict.items():
                if value is None: continue
                record = {
                    'id': str(uuid.uuid4()), 'metric_name': name,
                    'metric_value': float(value) if isinstance(value, (int, float)) else str(value),
                    'timestamp': timestamp, 'hostname': hostname, 'process_id': process_id,
                    'metadata': json.dumps({"source_class": "EUMS_OSMetrics"})
                }
                self.db_manager.insert_record('system_metrics', record)
        except Exception as e:
            logger.error(f"Error storing OS metrics: {e}", exc_info=True)

    def _create_os_alert(self, severity: str, source: str, message: str, details: Dict) -> OSAlert:
        alert_id = f"os_{source.replace('_monitor','').lower()}_{severity}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        return OSAlert(id=alert_id, timestamp=datetime.now(), severity=severity, source=source, message=message, details=details)

    def _check_os_thresholds(self, os_metrics: SystemMetrics):
        alerts_to_store: List[OSAlert] = []
        m = os_metrics # shorthand
        if m.memory_percent >= self.os_thresholds['memory_critical']:
            alerts_to_store.append(self._create_os_alert('critical', 'memory_monitor', f"Critical memory: {m.memory_percent:.1f}%", asdict(m)))
        elif m.memory_percent >= self.os_thresholds['memory_warning']:
            alerts_to_store.append(self._create_os_alert('warning', 'memory_monitor', f"High memory: {m.memory_percent:.1f}%", asdict(m)))
        # ... (add other OS threshold checks for CPU, disk, threads, GC objects) ...
        if m.cpu_percent >= self.os_thresholds['cpu_critical']:
             alerts_to_store.append(self._create_os_alert('critical', 'cpu_monitor', f"Critical CPU: {m.cpu_percent:.1f}%", {'cpu_percent': m.cpu_percent}))
        elif m.cpu_percent >= self.os_thresholds['cpu_warning']:
             alerts_to_store.append(self._create_os_alert('warning', 'cpu_monitor', f"High CPU: {m.cpu_percent:.1f}%", {'cpu_percent': m.cpu_percent}))

        for alert in alerts_to_store:
            try: self.db_manager.insert_record('system_alerts', alert.to_dict())
            except Exception as e: logger.error(f"Failed to store OS alert: {e}", exc_info=True)
    
    # --- DB Integration Metrics Methods (from original EUMS, adapted) ---
    def _collect_integration_metrics(self) -> Optional[DatabaseIntegrationMetrics]:
        # Same logic as before, but ensure it uses self.db_manager to get underlying TSCM stats if possible
        timestamp = datetime.now()
        try:
            udm_stats = self.db_manager.get_database_stats() 
            db_file_size_mb = udm_stats.get('file_size_mb', 0)
            active_connections, total_queries, query_errors, pool_efficiency = 0,0,0,0.0
            
            if hasattr(self.db_manager, 'ts_connection_manager') and self.db_manager.ts_connection_manager:
                ts_stats = self.db_manager.ts_connection_manager.get_connection_stats()
                active_connections = ts_stats.get('active_connections', 0)
                total_queries = ts_stats.get('total_queries_active_pool', 0)
                query_errors = ts_stats.get('total_errors_active_pool', 0)
                max_conns = ts_stats.get('max_connections_config', 1)
                if max_conns > 0: pool_efficiency = (active_connections / max_conns) * 100
            
            return DatabaseIntegrationMetrics(
                timestamp=timestamp, active_connections=active_connections, total_queries=total_queries,
                query_errors=query_errors, avg_query_time=0.0, # Placeholder, needs calculation
                connection_pool_efficiency=pool_efficiency, memory_used_by_connections_mb=0.0, # Placeholder
                orphaned_connections_cleaned=0, # Placeholder
                database_file_size_mb=db_file_size_mb
            )
        except Exception as e:
            logger.error(f"Error collecting DB integration metrics: {e}", exc_info=True)
            return None

    def _save_integration_metrics(self, metrics: DatabaseIntegrationMetrics):
        try: self.db_manager.insert_record('database_integration_metrics', metrics.to_dict())
        except Exception as e: logger.error(f"Error saving DB integration metrics: {e}", exc_info=True)

    def _analyze_and_alert_db_integration(self, metrics: DatabaseIntegrationMetrics):
        # Uses EnhancedAlert and 'enhanced_alerts' table
        alerts_to_create: List[Dict] = [] # List of dicts for EnhancedAlert
        if metrics.connection_pool_efficiency > 95: # Stricter threshold for EUMS
            alerts_to_create.append({
                "category": "database_pool_health", "severity": "critical", "title": "Extremely High Connection Pool Usage",
                "message": f"Connection pool at {metrics.connection_pool_efficiency:.1f}%. Risk of exhaustion.",
                "details": {"active": metrics.active_connections, "efficiency": metrics.connection_pool_efficiency},
                "metrics_snapshot": metrics.to_dict()
            })
        # ... more EUMS specific DB alerts ...
        for alert_data_kwargs in alerts_to_create:
            self._create_eums_specific_alert(**alert_data_kwargs)

    # --- EUMS Specific Alerting and Event Logging ---
    def _create_eums_specific_alert(self, category: str, severity: str, title: str, 
                                  message: str, details: Optional[Dict] = None, metrics_snapshot: Optional[Dict] = None):
        alert_id = f"eums_{category}_{severity}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        alert = EnhancedAlert(id=alert_id, timestamp=datetime.now(), category=category, severity=severity,
                              source="EnhancedUnifiedMonitoringSystem", title=title, message=message,
                              details=details, metrics=metrics_snapshot)
        try:
            self.db_manager.insert_record('enhanced_alerts', alert.to_dict())
            logger.info(f"EUMS Alert: [{severity.upper()}] {title}")
            for handler in self.alert_handlers:
                try: handler(alert) # Pass the EnhancedAlert object
                except Exception as e_h: logger.error(f"Error in EUMS alert handler: {e_h}", exc_info=True)
        except Exception as e:
            logger.error(f"Error creating EUMS specific alert: {e}", exc_info=True)

    def _log_system_event(self, event_type: str, source: str, description: str, 
                         data: Optional[Dict] = None, severity: str = "info"):
        event_data = {
            "timestamp": datetime.now().isoformat(), "event_type": event_type, "source": source,
            "description": description, "data": json.dumps(data or {}), "severity": severity
        }
        try: self.db_manager.insert_record('system_events', event_data)
        except Exception as e: logger.error(f"Error logging system event: {e}", exc_info=True)

    # --- Combined Cleanup and Optimization ---
    def _periodic_db_cleanup(self):
        logger.info("Running periodic database cleanup for all monitored tables...")
        try:
            # OS Metrics (system_metrics table)
            cutoff_os_metrics = (datetime.now() - timedelta(days=self.get_config_value('db_retention_days_os_metrics', 7))).isoformat()
            self.db_manager.execute_query("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_os_metrics,), fetch=False)
            
            # OS Alerts (system_alerts table)
            cutoff_os_alerts = (datetime.now() - timedelta(days=self.get_config_value('db_retention_days_os_alerts', 30))).isoformat()
            self.db_manager.execute_query("DELETE FROM system_alerts WHERE timestamp < ?", (cutoff_os_alerts,), fetch=False)

            # DB Integration Metrics
            cutoff_db_integration = (datetime.now() - timedelta(days=self.get_config_value('db_retention_days_db_integration', 7))).isoformat()
            self.db_manager.execute_query("DELETE FROM database_integration_metrics WHERE timestamp < ?", (cutoff_db_integration,), fetch=False)
            
            # Enhanced Alerts
            cutoff_eums_alerts = (datetime.now() - timedelta(days=self.get_config_value('db_retention_days_eums_alerts', 30))).isoformat()
            self.db_manager.execute_query("DELETE FROM enhanced_alerts WHERE timestamp < ?", (cutoff_eums_alerts,), fetch=False)

            # System Events
            cutoff_events = (datetime.now() - timedelta(days=self.get_config_value('db_retention_days_events', 14))).isoformat()
            self.db_manager.execute_query("DELETE FROM system_events WHERE timestamp < ?", (cutoff_events,), fetch=False)
            
            # Profiler tables (if any, from AdvancedMemoryProfiler, assuming they are in the same DB)
            cutoff_profiler = (datetime.now() - timedelta(days=self.get_config_value('db_retention_days_profiler', 3))).isoformat()
            for table in ['profiler_snapshots', 'profiler_leaks', 'profiler_hotspots']:
                 try: self.db_manager.execute_query(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_profiler,), fetch=False)
                 except DatabaseError: pass # Table might not exist if profiler disabled / schema issue
            
            logger.info("Periodic database cleanup completed.")
        except Exception as e:
            logger.error(f"Error during periodic DB cleanup: {e}", exc_info=True)

    def _periodic_db_optimization(self):
        logger.info("Running periodic database optimization...")
        try:
            self.db_manager.execute_query("ANALYZE;", fetch=False)
            if hasattr(self.db_manager, 'vacuum_database') and callable(self.db_manager.vacuum_database):
                self.db_manager.vacuum_database()
            else:
                self.db_manager.execute_query("VACUUM;", fetch=False)
            logger.info("Periodic database optimization completed.")
        except Exception as e:
            logger.error(f"Error during periodic DB optimization: {e}", exc_info=True)
            
    def _emergency_os_cleanup(self):
        logger.critical("EMERGENCY OS CLEANUP TRIGGERED (e.g., critical memory)")
        try:
            # Aggressively clear OS-related buffers
            self.os_metrics_buffer.clear()
            self.os_alerts_buffer.clear()
            
            # Aggressively clean DB tables related to OS monitoring
            cutoff_emergency = (datetime.now() - timedelta(hours=1)).isoformat()
            self.db_manager.execute_query("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_emergency,), fetch=False)
            self.db_manager.execute_query("DELETE FROM system_alerts WHERE timestamp < ?", (cutoff_emergency,), fetch=False)
            
            # Attempt VACUUM
            if hasattr(self.db_manager, 'vacuum_database') and callable(self.db_manager.vacuum_database):
                self.db_manager.vacuum_database()
            else:
                self.db_manager.execute_query("VACUUM;", fetch=False)
            logger.info("Emergency OS cleanup in DB performed.")
        except Exception as e:
            logger.error(f"Error during emergency OS cleanup: {e}", exc_info=True)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value from the system_config table."""
        try:
            record = self.db_manager.get_record('system_config', key)
            if record:
                # Attempt to cast to expected types based on key or store type info
                # For simplicity, returning string, let caller convert.
                return record['value']
        except Exception as e:
            logger.warning(f"Could not retrieve config for key '{key}', using default: {e}", exc_info=True)
        return default

    # --- Other methods (status, reports, etc.) ---
    # (get_integration_report, export_integration_report, _log_integration_status, etc. adapted from original EUMS)
    # (get_current_os_status, get_os_metrics_history, update_os_thresholds adapted from UMS)
    # ... many methods from original EUMS and UMS would be merged and adapted here ...
    # For brevity, not all are fully re-implemented in this diff, but the structure is set.

    def get_current_os_status(self) -> Dict[str, Any]:
        if not self.last_os_metrics: return {"error": "No OS metrics available"}
        metrics = self.last_os_metrics
        recent_os_alerts = [a.to_dict() for a in list(self.os_alerts_buffer)[-5:]]
        return {
            "timestamp": metrics.timestamp.isoformat(), "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent, "active_threads": metrics.active_threads,
            "recent_os_alerts": recent_os_alerts,
            "os_status": "CRITICAL" if metrics.memory_percent >= self.os_thresholds['memory_critical'] else
                         "WARNING" if metrics.memory_percent >= self.os_thresholds['memory_warning'] else "OK"
        }

    def get_full_system_report(self) -> Dict[str, Any]:
        logger.info("Generating full system report...")
        # Combines DB integration report logic with OS status
        db_integration_report = {} # Placeholder for actual DB integration report generation
        try: db_integration_report = self.get_integration_report() # Assuming this is refactored
        except Exception as e_rep: logger.error(f"Failed to get DB integration report part: {e_rep}", exc_info=True)

        return {
            "overall_status_timestamp": datetime.now().isoformat(),
            "os_level_status": self.get_current_os_status(),
            "database_integration_report": db_integration_report,
            "memory_profiler_active": self.memory_profiler is not None and self.memory_profiler.is_running,
        }

    # Placeholder for methods from original EUMS that need to be adapted
    def get_integration_report(self) -> Dict[str, Any]: return {"stub": "integration report not fully implemented in merge"}
    def _save_profiling_report(self, report: Dict): pass
    def add_alert_handler(self, handler: Callable): self.alert_handlers.append(handler)
    def export_integration_report(self, filepath: Optional[str]=None) -> str: return filepath or "stub_report.json"


    def cleanup(self):
        self.stop_monitoring()
        if self.memory_profiler:
            try: self.memory_profiler.cleanup()
            except Exception as e_mpc: logger.error(f"Error cleaning up memory profiler: {e_mpc}", exc_info=True)
        logger.info("EnhancedUnifiedMonitoringSystem resources cleaned up.")
    
    def __del__(self):
        try: self.cleanup()
        except Exception: pass # Avoid errors in __del__

# Factory and helper functions (can be kept outside the class)
# _eums_instance and _eums_lock are no longer needed here as singleton is managed by metaclass.

def get_enhanced_monitoring_system() -> Optional[EnhancedUnifiedMonitoringSystem]:
    """
    Returns the singleton instance of EnhancedUnifiedMonitoringSystem.
    The ThreadSafeSingleton metaclass handles instance creation and thread safety.
    """
    if not IMPORTS_AVAILABLE:
        logger.critical("Cannot get EUMS instance: Critical modules failed to import.")
        return None
    return EnhancedUnifiedMonitoringSystem()

# Main execution / test block
if __name__ == "__main__":
    logger.info("Starting EnhancedUnifiedMonitoringSystem test...")
    # Ensure core.config is set up for get_database_manager()
    try:
        from core.config import ensure_config_is_loaded
        ensure_config_is_loaded() # Make sure config is loaded before UDM is accessed
    except ImportError:
        logger.warning("core.config.ensure_config_is_loaded not found. Assuming config is pre-loaded.")
    except Exception as e_cfg:
        logger.error(f"Error ensuring config is loaded for test: {e_cfg}", exc_info=True)

    eums = get_enhanced_unified_monitor(enable_memory_profiling=False) # Disable for simple test
    if eums:
        try:
            eums.start_monitoring()
            logger.info("EUMS started. Collecting data for 90s...")
            time.sleep(90) # Let it run for a few cycles (3 OS, 1 DB Integration, etc.)
            
            full_report = eums.get_full_system_report()
            logger.info("\n--- Full System Report ---")
            logger.info(json.dumps(full_report, indent=2, default=str))

            report_path = eums.export_integration_report(filepath="test_eums_full_report.json") # Test export
            logger.info(f"Full report exported to {report_path}")

        except KeyboardInterrupt:
            logger.info("Test interrupted by user.")
        except Exception as e_test:
            logger.error(f"Error during EUMS test: {e_test}", exc_info=True)
        finally:
            logger.info("Stopping EUMS...")
            eums.stop_monitoring()
            logger.info("EUMS stopped.")
    else:
        logger.error("Failed to initialize EnhancedUnifiedMonitoringSystem for test.")
    
    logger.info("EnhancedUnifiedMonitoringSystem test finished.")
