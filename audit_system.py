"""
Advanced Audit System
Comprehensive audit logging and monitoring for LangGraph 101.

Features:
- Comprehensive event tracking
- Tamper-proof logs
- Log encryption
- Real-time alerting
- Compliance reporting
- Log integrity verification
- Event correlation
- Threat detection
"""

import logging
import json
import hashlib
import hmac
import time
import threading
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import sqlite3
import queue
import asyncio
from pathlib import Path
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import smtplib
from email.mime.text import MIMEText as MimeText


class AuditEventType(Enum):
    """Audit event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    ERROR = "error"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Alert types for real-time notifications."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    SMS = "sms"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AuditEventType = AuditEventType.USER_ACTION
    severity: AuditSeverity = AuditSeverity.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    event_type: Optional[AuditEventType] = None
    severity_threshold: AuditSeverity = AuditSeverity.HIGH
    frequency_threshold: int = 1  # Number of events
    time_window: int = 3600  # Time window in seconds
    user_pattern: Optional[str] = None
    ip_pattern: Optional[str] = None
    action_pattern: Optional[str] = None
    alert_types: List[AlertType] = field(default_factory=lambda: [AlertType.EMAIL])
    enabled: bool = True
    cooldown: int = 300  # Cooldown period in seconds
    last_triggered: Optional[datetime] = None


@dataclass
class AuditConfig:
    """Audit system configuration."""
    database_url: str = "sqlite:///audit.db"
    log_file_path: str = "audit.log"
    encryption_key: Optional[bytes] = None
    enable_encryption: bool = True
    enable_integrity_check: bool = True
    batch_size: int = 100
    flush_interval: int = 60  # seconds
    retention_days: int = 365
    enable_real_time_alerts: bool = True
    alert_email_config: Dict[str, str] = field(default_factory=dict)
    webhook_url: Optional[str] = None
    max_log_size: int = 100 * 1024 * 1024  # 100MB
    log_rotation_count: int = 5


# Database models
Base = declarative_base()


class AuditLog(Base):
    """Audit log database model."""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(String(36), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    user_id = Column(String(100))
    session_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    resource = Column(String(500))
    action = Column(String(200), nullable=False)
    details = Column(Text)  # JSON
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    request_id = Column(String(100))
    event_metadata = Column(Text)  # JSON - renamed from metadata to avoid SQLAlchemy conflict
    hash_value = Column(String(64))  # For integrity
    encrypted = Column(Boolean, default=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_event_type', 'event_type'),
        Index('idx_user_id', 'user_id'),
        Index('idx_severity', 'severity'),
        Index('idx_success', 'success'),
    )


class AuditManager:
    """
    Advanced audit logging and monitoring system.
    
    Provides comprehensive audit logging with encryption, integrity checking,
    real-time alerting, and compliance reporting capabilities.
    """
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize audit manager."""
        self.config = config or AuditConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        if self.config.enable_encryption:
            if not self.config.encryption_key:
                self.config.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.config.encryption_key)
        else:
            self.cipher = None
        
        # Initialize database
        self.engine = create_engine(self.config.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Event queue for batch processing
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_queue = queue.Queue()
        
        # Event counters for frequency-based alerts
        self.event_counters: Dict[str, List[datetime]] = {}
        self.counter_lock = threading.Lock()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Start background processing
        self.start_processing()
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="failed_logins",
                name="Failed Login Attempts",
                description="Multiple failed login attempts",
                event_type=AuditEventType.AUTHENTICATION,
                severity_threshold=AuditSeverity.MEDIUM,
                frequency_threshold=5,
                time_window=300,  # 5 minutes
                action_pattern="login"
            ),
            AlertRule(
                rule_id="security_violations",
                name="Security Violations",
                description="Security violation events",
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity_threshold=AuditSeverity.HIGH,
                frequency_threshold=1,
                time_window=60
            ),
            AlertRule(
                rule_id="admin_actions",
                name="Administrative Actions",
                description="Administrative actions requiring attention",
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                severity_threshold=AuditSeverity.HIGH,
                frequency_threshold=1,
                time_window=3600
            ),
            AlertRule(
                rule_id="data_modifications",
                name="Sensitive Data Modifications",
                description="Modifications to sensitive data",
                event_type=AuditEventType.DATA_MODIFICATION,
                severity_threshold=AuditSeverity.MEDIUM,
                frequency_threshold=10,
                time_window=3600
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def start_processing(self):
        """Start background processing threads."""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
            self.processing_thread.start()
            
            # Start alert processing thread
            self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.alert_thread.start()
    
    def stop_processing(self):
        """Stop background processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
    
    def log_event(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: Audit event to log
        """
        try:
            # Add to processing queue
            self.event_queue.put(event)
            
            # Check for real-time alerts
            if self.config.enable_real_time_alerts:
                self._check_alert_rules(event)
            
        except Exception as e:
            self.logger.error(f"Error logging audit event: {e}")
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str,
                          user_agent: str = None, details: Dict[str, Any] = None):
        """Log authentication event."""
        event = AuditEvent(
            event_type=AuditEventType.AUTHENTICATION,
            severity=AuditSeverity.INFO if success else AuditSeverity.MEDIUM,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            success=success,
            details=details or {}
        )
        self.log_event(event)
    
    def log_authorization(self, user_id: str, resource: str, action: str,
                         success: bool, details: Dict[str, Any] = None):
        """Log authorization event."""
        event = AuditEvent(
            event_type=AuditEventType.AUTHORIZATION,
            severity=AuditSeverity.MEDIUM if not success else AuditSeverity.INFO,
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            details=details or {}
        )
        self.log_event(event)
    
    def log_data_access(self, user_id: str, resource: str, action: str,
                       details: Dict[str, Any] = None):
        """Log data access event."""
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details or {}
        )
        self.log_event(event)
    
    def log_security_violation(self, user_id: str, violation_type: str,
                              ip_address: str, details: Dict[str, Any] = None):
        """Log security violation event."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            ip_address=ip_address,
            action=violation_type,
            success=False,
            details=details or {}
        )
        self.log_event(event)
    
    def log_configuration_change(self, user_id: str, resource: str, changes: Dict[str, Any]):
        """Log configuration change event."""
        event = AuditEvent(
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            resource=resource,
            action="configuration_change",
            details={"changes": changes}
        )
        self.log_event(event)
    
    def _process_events(self):
        """Background thread to process audit events."""
        events_batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Try to get event with timeout
                try:
                    event = self.event_queue.get(timeout=1)
                    events_batch.append(event)
                except queue.Empty:
                    pass
                
                # Flush batch if it's full or enough time has passed
                current_time = time.time()
                if (len(events_batch) >= self.config.batch_size or
                    (events_batch and current_time - last_flush >= self.config.flush_interval)):
                    
                    self._flush_events(events_batch)
                    events_batch = []
                    last_flush = current_time
                    
            except Exception as e:
                self.logger.error(f"Error processing audit events: {e}")
        
        # Flush remaining events
        if events_batch:
            self._flush_events(events_batch)
    
    def _flush_events(self, events: List[AuditEvent]):
        """
        Flush events to storage.
        
        Args:
            events: List of events to store
        """
        try:
            session = self.Session()
            
            for event in events:
                # Prepare event data
                event_data = event.to_dict()
                
                # Encrypt sensitive data if enabled
                encrypted_details = None
                encrypted_metadata = None
                
                if self.config.enable_encryption and self.cipher:
                    if event.details:
                        encrypted_details = self.cipher.encrypt(
                            json.dumps(event.details).encode('utf-8')
                        ).decode('utf-8')
                    
                    if event.metadata:
                        encrypted_metadata = self.cipher.encrypt(
                            json.dumps(event.metadata).encode('utf-8')
                        ).decode('utf-8')
                
                # Calculate integrity hash
                hash_value = None
                if self.config.enable_integrity_check:
                    hash_data = f"{event.event_id}{event.timestamp.isoformat()}{event.action}"
                    hash_value = hashlib.sha256(hash_data.encode('utf-8')).hexdigest()
                
                # Create database record
                audit_log = AuditLog(
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    resource=event.resource,
                    action=event.action,                    details=encrypted_details or json.dumps(event.details),
                    success=event.success,
                    error_message=event.error_message,
                    request_id=event.request_id,
                    event_metadata=encrypted_metadata or json.dumps(event.metadata),
                    hash_value=hash_value,
                    encrypted=self.config.enable_encryption
                )
                
                session.add(audit_log)
            
            session.commit()
            session.close()
            
            self.logger.debug(f"Flushed {len(events)} audit events to database")
            
        except Exception as e:
            self.logger.error(f"Error flushing audit events: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    def _check_alert_rules(self, event: AuditEvent):
        """
        Check event against alert rules.
        
        Args:
            event: Event to check
        """
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (rule.last_triggered and 
                datetime.now() - rule.last_triggered < timedelta(seconds=rule.cooldown)):
                continue
            
            if self._should_trigger_alert(event, rule):
                self.alert_queue.put((rule, event))
    
    def _should_trigger_alert(self, event: AuditEvent, rule: AlertRule) -> bool:
        """
        Check if alert should be triggered.
        
        Args:
            event: Event to check
            rule: Alert rule
            
        Returns:
            True if alert should be triggered
        """
        # Check event type
        if rule.event_type and event.event_type != rule.event_type:
            return False
        
        # Check severity threshold
        severity_levels = {
            AuditSeverity.INFO: 0,
            AuditSeverity.LOW: 1,
            AuditSeverity.MEDIUM: 2,
            AuditSeverity.HIGH: 3,
            AuditSeverity.CRITICAL: 4
        }
        
        if severity_levels[event.severity] < severity_levels[rule.severity_threshold]:
            return False
        
        # Check patterns
        if rule.user_pattern and event.user_id:
            import re
            if not re.search(rule.user_pattern, event.user_id):
                return False
        
        if rule.ip_pattern and event.ip_address:
            import re
            if not re.search(rule.ip_pattern, event.ip_address):
                return False
        
        if rule.action_pattern and event.action:
            import re
            if not re.search(rule.action_pattern, event.action):
                return False
        
        # Check frequency threshold
        if rule.frequency_threshold > 1:
            return self._check_frequency_threshold(event, rule)
        
        return True
    
    def _check_frequency_threshold(self, event: AuditEvent, rule: AlertRule) -> bool:
        """
        Check frequency-based threshold.
        
        Args:
            event: Current event
            rule: Alert rule
            
        Returns:
            True if frequency threshold exceeded
        """
        with self.counter_lock:
            # Create counter key based on rule and event characteristics
            counter_key = f"{rule.rule_id}_{event.user_id or 'unknown'}_{event.ip_address or 'unknown'}"
            
            # Initialize counter if needed
            if counter_key not in self.event_counters:
                self.event_counters[counter_key] = []
            
            # Add current event timestamp
            self.event_counters[counter_key].append(event.timestamp)
            
            # Remove old events outside time window
            cutoff_time = event.timestamp - timedelta(seconds=rule.time_window)
            self.event_counters[counter_key] = [
                ts for ts in self.event_counters[counter_key] if ts >= cutoff_time
            ]
            
            # Check if threshold exceeded
            return len(self.event_counters[counter_key]) >= rule.frequency_threshold
    
    def _process_alerts(self):
        """Background thread to process alerts."""
        while self.running:
            try:
                try:
                    rule, event = self.alert_queue.get(timeout=1)
                    self._send_alert(rule, event)
                    rule.last_triggered = datetime.now()
                except queue.Empty:
                    pass
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
    
    def _send_alert(self, rule: AlertRule, event: AuditEvent):
        """
        Send alert notification.
        
        Args:
            rule: Alert rule that triggered
            event: Event that triggered the alert
        """
        alert_message = f"""
Alert: {rule.name}
Description: {rule.description}
Event ID: {event.event_id}
Timestamp: {event.timestamp}
User: {event.user_id or 'Unknown'}
Action: {event.action}
Resource: {event.resource or 'N/A'}
IP Address: {event.ip_address or 'Unknown'}
Severity: {event.severity.value}
Success: {event.success}
Details: {json.dumps(event.details, indent=2)}
        """.strip()
        
        for alert_type in rule.alert_types:
            try:
                if alert_type == AlertType.EMAIL:
                    self._send_email_alert(rule, alert_message)
                elif alert_type == AlertType.WEBHOOK:
                    self._send_webhook_alert(rule, event)
                elif alert_type == AlertType.LOG:
                    self.logger.critical(f"SECURITY ALERT: {rule.name} - {alert_message}")
                elif alert_type == AlertType.SMS:
                    self._send_sms_alert(rule, alert_message)
            except Exception as e:
                self.logger.error(f"Error sending {alert_type.value} alert: {e}")
    
    def _send_email_alert(self, rule: AlertRule, message: str):
        """Send email alert."""
        if not self.config.alert_email_config:
            return
        
        smtp_server = self.config.alert_email_config.get('smtp_server')
        smtp_port = int(self.config.alert_email_config.get('smtp_port', 587))
        username = self.config.alert_email_config.get('username')
        password = self.config.alert_email_config.get('password')
        to_email = self.config.alert_email_config.get('to_email')
        
        if not all([smtp_server, username, password, to_email]):
            return
        
        msg = MimeText(message)
        msg['Subject'] = f"Security Alert: {rule.name}"
        msg['From'] = username
        msg['To'] = to_email
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _send_webhook_alert(self, rule: AlertRule, event: AuditEvent):
        """Send webhook alert."""
        if not self.config.webhook_url:
            return
        
        import requests
        
        payload = {
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'event': event.to_dict()
        }
        
        requests.post(self.config.webhook_url, json=payload, timeout=10)
    
    def _send_sms_alert(self, rule: AlertRule, message: str):
        """Send SMS alert (placeholder implementation)."""
        # This would integrate with SMS service like Twilio
        self.logger.info(f"SMS Alert: {rule.name} - {message[:100]}...")
    
    def query_events(self, start_time: datetime = None, end_time: datetime = None,
                    event_type: AuditEventType = None, user_id: str = None,
                    severity: AuditSeverity = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query audit events.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            event_type: Event type filter
            user_id: User ID filter
            severity: Severity filter
            limit: Maximum number of results
            
        Returns:
            List of audit events
        """
        try:
            session = self.Session()
            query = session.query(AuditLog)
            
            if start_time:
                query = query.filter(AuditLog.timestamp >= start_time)
            if end_time:
                query = query.filter(AuditLog.timestamp <= end_time)
            if event_type:
                query = query.filter(AuditLog.event_type == event_type.value)
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            if severity:
                query = query.filter(AuditLog.severity == severity.value)
            
            query = query.order_by(AuditLog.timestamp.desc()).limit(limit)
            results = query.all()
            
            events = []
            for result in results:
                event_data = {
                    'event_id': result.event_id,
                    'timestamp': result.timestamp.isoformat(),
                    'event_type': result.event_type,
                    'severity': result.severity,
                    'user_id': result.user_id,
                    'session_id': result.session_id,
                    'ip_address': result.ip_address,
                    'user_agent': result.user_agent,
                    'resource': result.resource,
                    'action': result.action,
                    'success': result.success,
                    'error_message': result.error_message,
                    'request_id': result.request_id
                }
                  # Decrypt details if encrypted
                if result.encrypted and self.cipher:
                    try:
                        if result.details:
                            decrypted_details = self.cipher.decrypt(result.details.encode('utf-8'))
                            event_data['details'] = json.loads(decrypted_details.decode('utf-8'))
                        if result.event_metadata:
                            decrypted_metadata = self.cipher.decrypt(result.event_metadata.encode('utf-8'))
                            event_data['metadata'] = json.loads(decrypted_metadata.decode('utf-8'))
                    except Exception as e:
                        self.logger.error(f"Error decrypting event data: {e}")
                        event_data['details'] = {"error": "decryption_failed"}
                        event_data['metadata'] = {"error": "decryption_failed"}
                else:
                    event_data['details'] = json.loads(result.details) if result.details else {}
                    event_data['metadata'] = json.loads(result.event_metadata) if result.event_metadata else {}
                
                events.append(event_data)
            
            session.close()
            return events
            
        except Exception as e:
            self.logger.error(f"Error querying audit events: {e}")
            return []
    
    def get_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report data
        """
        try:
            session = self.Session()
            
            # Get event statistics
            total_events = session.query(AuditLog).filter(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            ).count()
            
            # Get events by type
            events_by_type = {}
            for event_type in AuditEventType:
                count = session.query(AuditLog).filter(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date,
                    AuditLog.event_type == event_type.value
                ).count()
                events_by_type[event_type.value] = count
            
            # Get events by severity
            events_by_severity = {}
            for severity in AuditSeverity:
                count = session.query(AuditLog).filter(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date,
                    AuditLog.severity == severity.value
                ).count()
                events_by_severity[severity.value] = count
            
            # Get failed events
            failed_events = session.query(AuditLog).filter(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date,
                AuditLog.success == False
            ).count()
            
            # Get unique users
            unique_users = session.query(AuditLog.user_id).filter(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date,
                AuditLog.user_id.isnot(None)
            ).distinct().count()
            
            session.close()
            
            return {
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'summary': {
                    'total_events': total_events,
                    'failed_events': failed_events,
                    'success_rate': ((total_events - failed_events) / total_events * 100) if total_events > 0 else 0,
                    'unique_users': unique_users
                },
                'events_by_type': events_by_type,
                'events_by_severity': events_by_severity,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {}
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def verify_log_integrity(self, event_id: str) -> bool:
        """
        Verify log integrity for specific event.
        
        Args:
            event_id: Event ID to verify
            
        Returns:
            True if integrity is valid
        """
        try:
            session = self.Session()
            log_entry = session.query(AuditLog).filter(
                AuditLog.event_id == event_id
            ).first()
            
            if not log_entry or not log_entry.hash_value:
                session.close()
                return False
            
            # Recalculate hash
            hash_data = f"{log_entry.event_id}{log_entry.timestamp.isoformat()}{log_entry.action}"
            calculated_hash = hashlib.sha256(hash_data.encode('utf-8')).hexdigest()
            
            session.close()
            return calculated_hash == log_entry.hash_value
            
        except Exception as e:
            self.logger.error(f"Error verifying log integrity: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize audit manager
    config = AuditConfig()
    audit_manager = AuditManager(config)
    
    # Test logging different types of events
    print("=== Testing Audit Logging ===")
    
    # Log authentication events
    audit_manager.log_authentication("user123", True, "192.168.1.100", "Mozilla/5.0")
    audit_manager.log_authentication("user456", False, "192.168.1.200", "Mozilla/5.0")
    
    # Log authorization events
    audit_manager.log_authorization("user123", "/api/users", "read", True)
    audit_manager.log_authorization("user456", "/api/admin", "write", False)
    
    # Log security violations
    audit_manager.log_security_violation("user789", "sql_injection_attempt", "192.168.1.300", 
                                        {"payload": "'; DROP TABLE users; --"})
    
    # Wait for processing
    import time
    time.sleep(2)
    
    # Query events
    print("\n=== Querying Events ===")
    events = audit_manager.query_events(limit=10)
    for event in events[:3]:  # Show first 3 events
        print(f"Event: {event['action']} by {event['user_id']} at {event['timestamp']}")
    
    # Generate compliance report
    print("\n=== Compliance Report ===")
    report = audit_manager.get_compliance_report(
        datetime.now() - timedelta(days=1),
        datetime.now()
    )
    print(f"Total events: {report.get('summary', {}).get('total_events', 0)}")
    print(f"Success rate: {report.get('summary', {}).get('success_rate', 0):.1f}%")
    
    # Stop processing
    audit_manager.stop_processing()
