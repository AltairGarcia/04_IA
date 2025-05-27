"""
Production Performance Monitoring Dashboard
Real-time monitoring and alerting for LangGraph 101 application.
"""

import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import psutil
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    source: str
    metadata: Dict[str, Any] = None


@dataclass
class SystemAlert:
    """System alert data structure."""
    id: str
    timestamp: datetime
    severity: str  # info, warning, error, critical
    source: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


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

class PerformanceMonitor(metaclass=SingletonMeta):
    """Monitors application performance and system health."""
    
    def __init__(self, db_path: str = "performance_monitoring.db"):
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.db_path = db_path
        self.metrics_queue = deque(maxlen=1000)
        self.alerts_queue = deque(maxlen=100)
        self.running = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'response_time': 5.0,
            'error_rate': 5.0,
            'disk_usage': 90.0
        }
        
        # Initialize database
        self._init_database()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("PerformanceMonitor initialized as singleton")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON performance_metrics(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name 
                ON performance_metrics(metric_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON system_alerts(timestamp)
            """)
            
            conn.commit()
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk_percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def record_metric(self, metric_name: str, value: float, unit: str, 
                     source: str = "system", metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            source=source,
            metadata=metadata or {}
        )
        
        # Add to queue for real-time monitoring
        self.metrics_queue.append(metric)
        
        # Store in database
        self._store_metric(metric)
        
        # Check thresholds
        self._check_thresholds(metric)
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, metric_name, value, unit, source, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    metric.source,
                    json.dumps(metric.metadata) if metric.metadata else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and create alerts."""
        threshold = self.thresholds.get(metric.metric_name)
        if threshold and metric.value > threshold:
            self.create_alert(
                severity="warning" if metric.value < threshold * 1.2 else "error",
                source=f"threshold_monitor",
                message=f"{metric.metric_name} exceeded threshold",
                details={
                    "metric_name": metric.metric_name,
                    "current_value": metric.value,
                    "threshold": threshold,
                    "unit": metric.unit
                }
            )
    
    def create_alert(self, severity: str, source: str, message: str, 
                    details: Dict[str, Any]):
        """Create a system alert."""
        alert = SystemAlert(
            id=f"{source}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=severity,
            source=source,
            message=message,
            details=details
        )
        
        # Add to queue
        self.alerts_queue.append(alert)
        
        # Store in database
        self._store_alert(alert)
        
        # Log alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"Alert: {message} - {details}")
    
    def _store_alert(self, alert: SystemAlert):
        """Store alert in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO system_alerts 
                    (id, timestamp, severity, source, message, details, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.severity,
                    alert.source,
                    alert.message,
                    json.dumps(alert.details),
                    int(alert.resolved),
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def get_metrics(self, metric_name: str = None, hours: int = 24) -> List[PerformanceMetric]:
        """Retrieve metrics from database."""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if metric_name:
                    cursor.execute("""
                        SELECT timestamp, metric_name, value, unit, source, metadata
                        FROM performance_metrics
                        WHERE metric_name = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (metric_name, since.isoformat()))
                else:
                    cursor.execute("""
                        SELECT timestamp, metric_name, value, unit, source, metadata
                        FROM performance_metrics
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (since.isoformat(),))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append(PerformanceMetric(
                        timestamp=datetime.fromisoformat(row[0]),
                        metric_name=row[1],
                        value=row[2],
                        unit=row[3],
                        source=row[4],
                        metadata=json.loads(row[5]) if row[5] else {}
                    ))
                
                return metrics
        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            return []
    
    def get_alerts(self, resolved: bool = None, hours: int = 24) -> List[SystemAlert]:
        """Retrieve alerts from database."""
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, timestamp, severity, source, message, details, resolved, resolved_at
                    FROM system_alerts
                    WHERE timestamp >= ?
                """
                params = [since.isoformat()]
                  if resolved is not None:
                    query += " AND resolved = ?"
                    params.append(int(resolved))
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                alerts = []
                for row in cursor.fetchall():
                    alerts.append(SystemAlert(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        severity=row[2],
                        source=row[3],
                        message=row[4],
                        details=json.loads(row[5]),
                        resolved=bool(row[6]),
                        resolved_at=datetime.fromisoformat(row[7]) if row[7] else None
                    ))
                
                return alerts
        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop, 
                daemon=True, 
                name="PerformanceMonitor-Main"
            )
            self.monitor_thread.start()
            logger.info("Performance monitoring started (singleton)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop - optimized for memory efficiency."""
        cleanup_counter = 0
        
        while self.running:
            try:
                # Collect system metrics
                metrics = self.collect_system_metrics()
                
                for metric_name, value in metrics.items():
                    unit = self._get_metric_unit(metric_name)
                    self.record_metric(metric_name, value, unit, "system_monitor")
                
                # Periodic cleanup every 10 cycles (5 minutes)
                cleanup_counter += 1
                if cleanup_counter >= 10:
                    self._cleanup_old_metrics()
                    cleanup_counter = 0
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory growth."""
        try:
            # Keep only recent metrics in memory
            if len(self.metrics_queue) > 500:
                # Remove oldest 200 metrics
                for _ in range(200):
                    if self.metrics_queue:
                        self.metrics_queue.popleft()
            
            # Clean up old database entries (keep last 10000 records)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM performance_metrics 
                    WHERE id NOT IN (
                        SELECT id FROM performance_metrics 
                        ORDER BY timestamp DESC LIMIT 10000
                    )
                """)
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old performance metrics")
                
                # Clean up old alerts (keep last 1000)
                cursor.execute("""
                    DELETE FROM system_alerts 
                    WHERE id NOT IN (
                        SELECT id FROM system_alerts 
                        ORDER BY timestamp DESC LIMIT 1000
                    )
                """)
                deleted_alerts = cursor.rowcount
                if deleted_alerts > 0:
                    logger.info(f"Cleaned up {deleted_alerts} old alerts")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric name."""
        unit_map = {
            'cpu_usage': '%',
            'memory_usage': '%',
            'memory_available_gb': 'GB',
            'disk_usage': '%',
            'disk_free_gb': 'GB',
            'network_bytes_sent': 'bytes',
            'network_bytes_recv': 'bytes',
            'response_time': 'seconds',
            'error_rate': '%'
        }
        return unit_map.get(metric_name, 'units')


class MonitoringDashboard:
    """Streamlit dashboard for performance monitoring."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics - delegate to PerformanceMonitor."""
        return self.monitor.collect_system_metrics()
    
    def render_dashboard(self):
        """Render the complete monitoring dashboard."""
        self.render()
    
    def render(self):
        """Render the monitoring dashboard."""
        st.title("📊 Performance Monitoring Dashboard")
        
        # Real-time metrics overview
        self._render_realtime_overview()
        
        # Detailed metrics charts
        self._render_metrics_charts()
        
        # System alerts
        self._render_alerts_section()
        
        # System information
        self._render_system_info()
    
    def _render_realtime_overview(self):
        """Render real-time metrics overview."""
        st.subheader("🔴 Real-time System Status")
        
        # Get latest metrics
        latest_metrics = {}
        for metric in list(self.monitor.metrics_queue)[-10:]:
            latest_metrics[metric.metric_name] = metric.value
        
        # Create columns for metrics
        cols = st.columns(4)
        
        with cols[0]:
            cpu = latest_metrics.get('cpu_usage', 0)
            delta_color = "normal" if cpu < 70 else "inverse"
            st.metric("CPU Usage", f"{cpu:.1f}%", delta=None, delta_color=delta_color)
        
        with cols[1]:
            memory = latest_metrics.get('memory_usage', 0)
            delta_color = "normal" if memory < 80 else "inverse"
            st.metric("Memory Usage", f"{memory:.1f}%", delta=None, delta_color=delta_color)
        
        with cols[2]:
            disk = latest_metrics.get('disk_usage', 0)
            delta_color = "normal" if disk < 85 else "inverse"
            st.metric("Disk Usage", f"{disk:.1f}%", delta=None, delta_color=delta_color)
        
        with cols[3]:
            alerts = len([a for a in self.monitor.alerts_queue if not a.resolved])
            delta_color = "normal" if alerts == 0 else "inverse"
            st.metric("Active Alerts", alerts, delta=None, delta_color=delta_color)
    
    def _render_metrics_charts(self):
        """Render detailed metrics charts."""
        st.subheader("📈 Performance Trends")
        
        # Time range selector
        hours = st.selectbox("Time Range", [1, 6, 12, 24, 48], index=3)
        
        # Get metrics
        metrics = self.monitor.get_metrics(hours=hours)
        
        if not metrics:
            st.info("No metrics data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'metric_name': m.metric_name,
                'value': m.value,
                'unit': m.unit
            }
            for m in metrics
        ])
        
        # Create subplots for different metric types
        system_metrics = ['cpu_usage', 'memory_usage', 'disk_usage']
        network_metrics = ['network_bytes_sent', 'network_bytes_recv']
        
        # System metrics chart
        if any(metric in df['metric_name'].values for metric in system_metrics):
            st.markdown("#### System Resources")
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            for metric in system_metrics:
                metric_data = df[df['metric_name'] == metric]
                if not metric_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=metric_data['timestamp'],
                            y=metric_data['value'],
                            name=metric.replace('_', ' ').title(),
                            mode='lines+markers'
                        )
                    )
            
            fig.update_layout(
                title="System Resource Usage",
                xaxis_title="Time",
                yaxis_title="Percentage (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Network metrics chart
        if any(metric in df['metric_name'].values for metric in network_metrics):
            st.markdown("#### Network Activity")
            network_df = df[df['metric_name'].isin(network_metrics)]
            
            if not network_df.empty:
                fig = px.line(
                    network_df,
                    x='timestamp',
                    y='value',
                    color='metric_name',
                    title="Network I/O"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts_section(self):
        """Render system alerts section."""
        st.subheader("🚨 System Alerts")
        
        # Get recent alerts
        alerts = self.monitor.get_alerts(hours=24)
        
        if not alerts:
            st.success("No alerts in the last 24 hours")
            return
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            show_resolved = st.checkbox("Show resolved alerts", value=False)
        with col2:
            severity_filter = st.multiselect(
                "Filter by severity",
                ['info', 'warning', 'error', 'critical'],
                default=['warning', 'error', 'critical']
            )
        
        # Filter alerts
        filtered_alerts = [
            a for a in alerts
            if (show_resolved or not a.resolved) and a.severity in severity_filter
        ]
        
        # Display alerts
        for alert in filtered_alerts[:20]:  # Show last 20 alerts
            severity_color = {
                'info': '🔵',
                'warning': '🟡',
                'error': '🟠',
                'critical': '🔴'
            }.get(alert.severity, '⚪')
            
            status = "✅ Resolved" if alert.resolved else "❌ Active"
            
            with st.expander(f"{severity_color} {alert.message} - {status}"):
                st.write(f"**Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Source:** {alert.source}")
                st.write(f"**Severity:** {alert.severity}")
                
                if alert.details:
                    st.write("**Details:**")
                    st.json(alert.details)
                
                if alert.resolved and alert.resolved_at:
                    st.write(f"**Resolved at:** {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _render_system_info(self):
        """Render system information."""
        st.subheader("💻 System Information")
        
        try:
            # System info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Hardware")
                st.write(f"**CPU Cores:** {psutil.cpu_count()}")
                st.write(f"**CPU Frequency:** {psutil.cpu_freq().current:.0f} MHz")
                
                memory = psutil.virtual_memory()
                st.write(f"**Total Memory:** {memory.total / (1024**3):.1f} GB")
                st.write(f"**Available Memory:** {memory.available / (1024**3):.1f} GB")
            
            with col2:
                st.markdown("#### Storage")
                disk = psutil.disk_usage('/')
                st.write(f"**Total Disk:** {disk.total / (1024**3):.1f} GB")
                st.write(f"**Free Disk:** {disk.free / (1024**3):.1f} GB")
                st.write(f"**Used Disk:** {disk.used / (1024**3):.1f} GB")
        
        except Exception as e:
            st.error(f"Error getting system info: {e}")


# Global monitor instance - singleton will ensure only one instance
def get_performance_monitor():
    """Get the global performance monitor instance."""
    return PerformanceMonitor()

# Initialize the singleton instance
_performance_monitor = None

def get_global_monitor():
    """Get the global monitor instance, creating it if necessary."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Helper functions for integration
def record_response_time(endpoint: str, duration: float):
    """Record API response time."""
    monitor = get_global_monitor()
    monitor.record_metric(
        'response_time',
        duration,
        'seconds',
        'api',
        {'endpoint': endpoint}
    )


def record_error_rate(endpoint: str, error_count: int, total_count: int):
    """Record error rate for endpoint."""
    error_rate = (error_count / total_count) * 100 if total_count > 0 else 0
    monitor = get_global_monitor()
    monitor.record_metric(
        'error_rate',
        error_rate,
        '%',
        'api',
        {'endpoint': endpoint, 'error_count': error_count, 'total_count': total_count}
    )


if __name__ == "__main__":
    # Test the monitoring system using singleton
    monitor = get_global_monitor()
    
    # Record some test metrics
    monitor.record_metric('cpu_usage', 45.2, '%', 'test')
    monitor.record_metric('memory_usage', 67.8, '%', 'test')
    
    # Create test alert
    monitor.create_alert(
        'warning',
        'test',
        'Test alert message',
        {'test_key': 'test_value'}
    )
    
    print("Test metrics and alerts recorded using singleton")
    
    # Stop monitoring
    monitor.stop_monitoring()
