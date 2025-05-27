"""
Security Monitoring Dashboard
Real-time security monitoring and visualization for LangGraph 101.

Features:
- Real-time security metrics
- Threat visualization
- Alert management
- Security score calculation
- Incident response
- Performance monitoring
- Threat intelligence
- Dashboard widgets
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
import asyncio
import websockets
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
from audit_system import AuditManager, AuditEventType, AuditSeverity, AuditLog
from ddos_protection import DDoSProtectionManager


class DashboardWidget(Enum):
    """Dashboard widget types."""
    THREAT_LEVEL = "threat_level"
    SECURITY_SCORE = "security_score"
    ALERT_COUNT = "alert_count"
    EVENT_TIMELINE = "event_timeline"
    TOP_THREATS = "top_threats"
    USER_ACTIVITY = "user_activity"
    GEOLOCATION = "geolocation"
    PERFORMANCE_METRICS = "performance_metrics"
    INCIDENT_STATUS = "incident_status"
    COMPLIANCE_STATUS = "compliance_status"


class ThreatLevel(Enum):
    """Threat level classifications."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status types."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityMetric:
    """Security metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    category: str
    description: str = ""
    trend: Optional[str] = None  # "up", "down", "stable"


@dataclass
class ThreatIndicator:
    """Threat indicator data."""
    threat_id: str
    name: str
    description: str
    severity: AuditSeverity
    category: str
    count: int
    first_seen: datetime
    last_seen: datetime
    sources: List[str]
    indicators: Dict[str, Any]


@dataclass
class SecurityIncident:
    """Security incident data."""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    severity: AuditSeverity = AuditSeverity.MEDIUM
    status: IncidentStatus = IncidentStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)  # Event IDs
    notes: List[str] = field(default_factory=list)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    update_interval: int = 30  # seconds
    history_window: int = 3600  # seconds (1 hour)
    max_data_points: int = 1000
    websocket_port: int = 8765
    dashboard_host: str = "localhost"
    dashboard_port: int = 8080
    enable_real_time: bool = True
    enable_geolocation: bool = True
    threat_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'failed_logins': 0.3,
        'security_violations': 0.4,
        'suspicious_activity': 0.2,
        'performance_issues': 0.1
    })


class SecurityDashboard:
    """
    Real-time security monitoring dashboard.
    
    Provides comprehensive security monitoring with real-time metrics,
    threat visualization, alert management, and incident response.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None,
                 audit_manager: Optional[AuditManager] = None):
        """Initialize security dashboard."""
        self.config = config or DashboardConfig()
        self.audit_manager = audit_manager
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_data_points))
        self.threats: Dict[str, ThreatIndicator] = {}
        self.incidents: Dict[str, SecurityIncident] = {}
        
        # Dashboard state
        self.running = False
        self.connected_clients = set()
        self.last_update = datetime.now()
        
        # Initialize components
        self._setup_metrics_collection()
        self._setup_threat_detection()
        
        # Background tasks
        self.update_thread = None
        self.websocket_server = None
    
    def _setup_metrics_collection(self):
        """Setup metrics collection."""
        # Initialize metric categories
        self.metric_categories = {
            'security': ['threat_score', 'failed_logins', 'security_violations', 'blocked_ips'],
            'performance': ['response_time', 'cpu_usage', 'memory_usage', 'disk_usage'],
            'activity': ['active_users', 'requests_per_minute', 'data_transfers', 'api_calls'],
            'compliance': ['audit_coverage', 'policy_violations', 'access_reviews', 'data_retention']
        }
    
    def _setup_threat_detection(self):
        """Setup threat detection rules."""
        self.threat_patterns = {
            'brute_force': {
                'pattern': 'multiple_failed_logins',
                'threshold': 5,
                'window': 300,  # 5 minutes
                'severity': AuditSeverity.HIGH
            },
            'sql_injection': {
                'pattern': 'sql_injection_attempt',
                'threshold': 1,
                'window': 60,
                'severity': AuditSeverity.CRITICAL
            },
            'xss_attack': {
                'pattern': 'xss_attempt',
                'threshold': 1,
                'window': 60,
                'severity': AuditSeverity.HIGH
            },
            'dos_attack': {
                'pattern': 'rate_limit_exceeded',
                'threshold': 10,
                'window': 60,
                'severity': AuditSeverity.HIGH
            }
        }
    
    def start(self):
        """Start dashboard monitoring."""
        if not self.running:
            self.running = True
            
            # Start metrics collection thread
            self.update_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
            self.update_thread.start()
            
            # Start WebSocket server for real-time updates
            if self.config.enable_real_time:
                asyncio.run(self._start_websocket_server())
    
    def stop(self):
        """Stop dashboard monitoring."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
    
    def _update_metrics_loop(self):
        """Background thread for updating metrics."""
        while self.running:
            try:
                self.update_all_metrics()
                time.sleep(self.config.update_interval)
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                time.sleep(5)
    
    def update_all_metrics(self):
        """Update all dashboard metrics."""
        current_time = datetime.now()
        
        # Update security metrics
        self._update_security_metrics(current_time)
        
        # Update performance metrics
        self._update_performance_metrics(current_time)
        
        # Update activity metrics
        self._update_activity_metrics(current_time)
        
        # Update compliance metrics
        self._update_compliance_metrics(current_time)
        
        # Detect threats
        self._detect_threats(current_time)
        
        # Calculate security score
        self._calculate_security_score(current_time)
        
        # Send real-time updates
        if self.config.enable_real_time:
            asyncio.run(self._broadcast_updates())
        
        self.last_update = current_time
    
    def _update_security_metrics(self, timestamp: datetime):
        """Update security-related metrics."""
        if not self.audit_manager:
            return
        
        # Time window for metrics
        window_start = timestamp - timedelta(seconds=self.config.history_window)
        
        try:
            # Get recent events
            events = self.audit_manager.query_events(
                start_time=window_start,
                end_time=timestamp,
                limit=1000
            )
            
            # Count failed logins
            failed_logins = sum(1 for event in events 
                               if event['event_type'] == 'authentication' and not event['success'])
            self._add_metric('failed_logins', failed_logins, timestamp, 'security')
            
            # Count security violations
            security_violations = sum(1 for event in events 
                                    if event['event_type'] == 'security_violation')
            self._add_metric('security_violations', security_violations, timestamp, 'security')
            
            # Count unique blocked IPs (placeholder - would come from DDoS protection)
            blocked_ips = len(set(event['ip_address'] for event in events 
                                if event.get('ip_address') and not event['success']))
            self._add_metric('blocked_ips', blocked_ips, timestamp, 'security')
            
        except Exception as e:
            self.logger.error(f"Error updating security metrics: {e}")
    
    def _update_performance_metrics(self, timestamp: datetime):
        """Update performance metrics."""
        import psutil
        
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self._add_metric('cpu_usage', cpu_usage, timestamp, 'performance')
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self._add_metric('memory_usage', memory_usage, timestamp, 'performance')
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            self._add_metric('disk_usage', disk_usage, timestamp, 'performance')
            
            # Response time (placeholder - would come from actual monitoring)
            import random
            response_time = random.uniform(50, 200)  # ms
            self._add_metric('response_time', response_time, timestamp, 'performance')
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _update_activity_metrics(self, timestamp: datetime):
        """Update activity metrics."""
        if not self.audit_manager:
            return
        
        try:
            # Time window for activity
            window_start = timestamp - timedelta(minutes=5)
            
            events = self.audit_manager.query_events(
                start_time=window_start,
                end_time=timestamp,
                limit=1000
            )
            
            # Active users
            active_users = len(set(event['user_id'] for event in events 
                                 if event.get('user_id')))
            self._add_metric('active_users', active_users, timestamp, 'activity')
            
            # Requests per minute
            requests_per_minute = len(events) / 5  # 5-minute window
            self._add_metric('requests_per_minute', requests_per_minute, timestamp, 'activity')
            
            # API calls
            api_calls = sum(1 for event in events 
                           if event.get('resource', '').startswith('/api/'))
            self._add_metric('api_calls', api_calls, timestamp, 'activity')
            
        except Exception as e:
            self.logger.error(f"Error updating activity metrics: {e}")
    
    def _update_compliance_metrics(self, timestamp: datetime):
        """Update compliance metrics."""
        try:
            # Placeholder compliance metrics
            audit_coverage = 95.5  # Percentage of systems audited
            self._add_metric('audit_coverage', audit_coverage, timestamp, 'compliance')
            
            policy_violations = 2  # Number of policy violations
            self._add_metric('policy_violations', policy_violations, timestamp, 'compliance')
            
            access_reviews = 98.2  # Percentage of access reviews completed
            self._add_metric('access_reviews', access_reviews, timestamp, 'compliance')
            
            data_retention = 99.1  # Percentage compliance with data retention
            self._add_metric('data_retention', data_retention, timestamp, 'compliance')
            
        except Exception as e:
            self.logger.error(f"Error updating compliance metrics: {e}")
    
    def _add_metric(self, name: str, value: Union[int, float], 
                   timestamp: datetime, category: str, description: str = ""):
        """Add metric data point."""
        metric = SecurityMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            category=category,
            description=description
        )
        
        self.metrics[name].append(metric)
    
    def _detect_threats(self, timestamp: datetime):
        """Detect security threats based on metrics."""
        if not self.audit_manager:
            return
        
        # Time window for threat detection
        window_start = timestamp - timedelta(seconds=300)  # 5 minutes
        
        try:
            events = self.audit_manager.query_events(
                start_time=window_start,
                end_time=timestamp,
                limit=1000
            )
            
            # Group events by IP and check for threats
            events_by_ip = defaultdict(list)
            for event in events:
                if event.get('ip_address'):
                    events_by_ip[event['ip_address']].append(event)
            
            for ip_address, ip_events in events_by_ip.items():
                self._check_ip_threats(ip_address, ip_events, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error detecting threats: {e}")
    
    def _check_ip_threats(self, ip_address: str, events: List[Dict], timestamp: datetime):
        """Check for threats from specific IP."""
        failed_logins = sum(1 for event in events 
                           if event['event_type'] == 'authentication' and not event['success'])
        
        # Brute force detection
        if failed_logins >= 5:
            threat_id = f"brute_force_{ip_address}"
            if threat_id not in self.threats:
                self.threats[threat_id] = ThreatIndicator(
                    threat_id=threat_id,
                    name="Brute Force Attack",
                    description=f"Multiple failed login attempts from {ip_address}",
                    severity=AuditSeverity.HIGH,
                    category="authentication",
                    count=failed_logins,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    sources=[ip_address],
                    indicators={"failed_logins": failed_logins}
                )
            else:
                self.threats[threat_id].count += failed_logins
                self.threats[threat_id].last_seen = timestamp
    
    def _calculate_security_score(self, timestamp: datetime):
        """Calculate overall security score."""
        try:
            # Get recent metrics
            score = 100.0  # Start with perfect score
            
            # Deduct points for threats
            recent_metrics = self._get_recent_metrics(300)  # 5 minutes
            
            failed_logins = recent_metrics.get('failed_logins', 0)
            security_violations = recent_metrics.get('security_violations', 0)
            
            # Apply scoring weights
            score -= failed_logins * self.config.threat_score_weights.get('failed_logins', 0.3)
            score -= security_violations * self.config.threat_score_weights.get('security_violations', 0.4)
            
            # Ensure score is between 0 and 100
            score = max(0, min(100, score))
            
            self._add_metric('security_score', score, timestamp, 'security', 
                           "Overall security score (0-100)")
            
        except Exception as e:
            self.logger.error(f"Error calculating security score: {e}")
    
    def _get_recent_metrics(self, window_seconds: int) -> Dict[str, float]:
        """Get recent metric values."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_metrics = {}
        
        for metric_name, metric_deque in self.metrics.items():
            recent_values = [m.value for m in metric_deque 
                           if m.timestamp >= cutoff_time]
            if recent_values:
                recent_metrics[metric_name] = sum(recent_values) / len(recent_values)
        
        return recent_metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        current_metrics = self._get_recent_metrics(60)  # Last minute
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(current_metrics)
        
        # Get top threats
        top_threats = sorted(self.threats.values(), 
                           key=lambda t: t.count, reverse=True)[:5]
        
        # Get recent incidents
        recent_incidents = [incident for incident in self.incidents.values()
                          if incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'security_score': current_metrics.get('security_score', 100),
            'threat_level': threat_level.value,
            'metrics': {
                'security': {k: v for k, v in current_metrics.items() 
                           if k in ['failed_logins', 'security_violations', 'blocked_ips']},
                'performance': {k: v for k, v in current_metrics.items() 
                              if k in ['cpu_usage', 'memory_usage', 'response_time']},
                'activity': {k: v for k, v in current_metrics.items() 
                           if k in ['active_users', 'requests_per_minute', 'api_calls']},
                'compliance': {k: v for k, v in current_metrics.items() 
                             if k in ['audit_coverage', 'policy_violations']}
            },
            'threats': [
                {
                    'name': threat.name,
                    'severity': threat.severity.value,
                    'count': threat.count,
                    'last_seen': threat.last_seen.isoformat()
                }
                for threat in top_threats
            ],
            'incidents': [
                {
                    'id': incident.incident_id,
                    'title': incident.title,
                    'severity': incident.severity.value,
                    'status': incident.status.value,
                    'created_at': incident.created_at.isoformat()
                }
                for incident in recent_incidents[:10]
            ]
        }
    
    def _calculate_threat_level(self, metrics: Dict[str, float]) -> ThreatLevel:
        """Calculate current threat level."""
        security_score = metrics.get('security_score', 100)
        
        if security_score >= 90:
            return ThreatLevel.MINIMAL
        elif security_score >= 75:
            return ThreatLevel.LOW
        elif security_score >= 50:
            return ThreatLevel.MODERATE
        elif security_score >= 25:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
    
    def create_incident(self, title: str, description: str, 
                       severity: AuditSeverity = AuditSeverity.MEDIUM) -> str:
        """Create security incident."""
        incident = SecurityIncident(
            title=title,
            description=description,
            severity=severity
        )
        
        self.incidents[incident.incident_id] = incident
        self.logger.info(f"Created security incident: {incident.incident_id}")
        
        return incident.incident_id
    
    def update_incident(self, incident_id: str, status: IncidentStatus = None,
                       assigned_to: str = None, notes: str = None):
        """Update security incident."""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            
            if status:
                incident.status = status
            if assigned_to:
                incident.assigned_to = assigned_to
            if notes:
                incident.notes.append(f"{datetime.now().isoformat()}: {notes}")
            
            incident.updated_at = datetime.now()
            self.logger.info(f"Updated security incident: {incident_id}")
    
    def generate_chart_data(self, metric_name: str, 
                           time_range: int = 3600) -> Dict[str, Any]:
        """Generate chart data for specific metric."""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(seconds=time_range)
        data_points = [m for m in self.metrics[metric_name] 
                      if m.timestamp >= cutoff_time]
        
        if not data_points:
            return {}
        
        timestamps = [m.timestamp.isoformat() for m in data_points]
        values = [m.value for m in data_points]
        
        # Create Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=metric_name.replace('_', ' ').title(),
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=f"{metric_name.replace('_', ' ').title()} - Last {time_range//60} Minutes",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white"
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            """Handle WebSocket client connection."""
            self.connected_clients.add(websocket)
            try:
                # Send initial dashboard data
                await websocket.send(json.dumps({
                    'type': 'dashboard_data',
                    'data': self.get_dashboard_data()
                }))
                
                # Keep connection alive
                await websocket.wait_closed()
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                self.connected_clients.discard(websocket)
        
        try:
            await websockets.serve(
                handle_client,
                self.config.dashboard_host,
                self.config.websocket_port
            )
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
    
    async def _broadcast_updates(self):
        """Broadcast updates to connected clients."""
        if not self.connected_clients:
            return
        
        dashboard_data = self.get_dashboard_data()
        message = json.dumps({
            'type': 'update',
            'data': dashboard_data
        })
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients


# Example usage and testing
if __name__ == "__main__":
    from audit_system import AuditConfig, AuditManager
    
    # Initialize components
    audit_config = AuditConfig()
    audit_manager = AuditManager(audit_config)
    
    dashboard_config = DashboardConfig()
    dashboard = SecurityDashboard(dashboard_config, audit_manager)
    
    # Test dashboard functionality
    print("=== Security Dashboard Test ===")
    
    # Generate some test data
    audit_manager.log_authentication("user1", False, "192.168.1.100")
    audit_manager.log_authentication("user2", False, "192.168.1.100")
    audit_manager.log_security_violation("user3", "sql_injection", "192.168.1.200")
    
    # Update metrics
    dashboard.update_all_metrics()
    
    # Get dashboard data
    data = dashboard.get_dashboard_data()
    print(f"Security Score: {data['security_score']:.1f}")
    print(f"Threat Level: {data['threat_level']}")
    print(f"Active Threats: {len(data['threats'])}")
    print(f"Open Incidents: {len(data['incidents'])}")
    
    # Create test incident
    incident_id = dashboard.create_incident(
        "Brute Force Attack Detected",
        "Multiple failed login attempts from suspicious IP",
        AuditSeverity.HIGH
    )
    print(f"Created incident: {incident_id}")
    
    # Generate chart data
    chart_data = dashboard.generate_chart_data('security_score')
    if chart_data:
        print("Generated security score chart data")
    
    # Stop components
    dashboard.stop()
    audit_manager.stop_processing()
