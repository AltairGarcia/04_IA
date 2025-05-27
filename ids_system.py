"""
Intrusion Detection System (IDS)
Advanced intrusion detection and automated response for LangGraph 101.

Features:
- Anomaly detection
- Pattern recognition
- Automated response
- Machine learning-based detection
- Behavioral analysis
- Network traffic analysis
- Host-based monitoring
- Real-time alerting
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import numpy as np
import pickle
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
import hashlib
import ipaddress
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
from audit_system import AuditManager, AuditEventType, AuditSeverity
from ddos_protection import DDoSProtectionManager


class DetectionType(Enum):
    """IDS detection types."""
    ANOMALY = "anomaly"
    SIGNATURE = "signature"
    BEHAVIORAL = "behavioral"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"


class IntrusionType(Enum):
    """Types of intrusions."""
    BRUTE_FORCE = "brute_force"
    DOS_ATTACK = "dos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE = "malware"
    INSIDER_THREAT = "insider_threat"
    RECONNAISSANCE = "reconnaissance"
    COMMAND_INJECTION = "command_injection"


class ResponseAction(Enum):
    """Automated response actions."""
    BLOCK_IP = "block_ip"
    RATE_LIMIT = "rate_limit"
    QUARANTINE_USER = "quarantine_user"
    LOG_ALERT = "log_alert"
    EMAIL_ADMIN = "email_admin"
    DISABLE_ACCOUNT = "disable_account"
    INCREASE_MONITORING = "increase_monitoring"
    CAPTURE_TRAFFIC = "capture_traffic"


@dataclass
class IntrusionSignature:
    """Intrusion detection signature."""
    signature_id: str
    name: str
    description: str
    intrusion_type: IntrusionType
    pattern: str  # Regex pattern or rule
    severity: AuditSeverity
    confidence: float  # 0.0 - 1.0
    enabled: bool = True
    false_positive_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class IntrusionEvent:
    """Detected intrusion event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    intrusion_type: IntrusionType = IntrusionType.RECONNAISSANCE
    detection_type: DetectionType = DetectionType.SIGNATURE
    severity: AuditSeverity = AuditSeverity.MEDIUM
    confidence: float = 0.5
    source_ip: Optional[str] = None
    target: Optional[str] = None
    user_id: Optional[str] = None
    signature_id: Optional[str] = None
    description: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[ResponseAction] = field(default_factory=list)
    resolved: bool = False


@dataclass
class UserBehaviorProfile:
    """User behavioral profile for anomaly detection."""
    user_id: str
    login_times: List[int] = field(default_factory=list)  # Hour of day
    login_locations: Set[str] = field(default_factory=set)  # IP addresses
    access_patterns: Dict[str, int] = field(default_factory=dict)  # Resource access counts
    session_durations: List[float] = field(default_factory=list)  # Minutes
    failed_login_rate: float = 0.0
    privilege_usage: Dict[str, int] = field(default_factory=dict)
    data_transfer_volumes: List[float] = field(default_factory=list)  # MB
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_profile(self, event_data: Dict[str, Any]):
        """Update profile with new event data."""
        if event_data.get('event_type') == 'authentication':
            if event_data.get('success'):
                login_hour = datetime.fromisoformat(event_data['timestamp']).hour
                self.login_times.append(login_hour)
                if len(self.login_times) > 100:  # Keep last 100 logins
                    self.login_times.pop(0)
                
                if event_data.get('ip_address'):
                    self.login_locations.add(event_data['ip_address'])
        
        self.last_updated = datetime.now()


@dataclass
class IDSConfig:
    """IDS configuration."""
    enable_anomaly_detection: bool = True
    enable_signature_detection: bool = True
    enable_behavioral_analysis: bool = True
    enable_ml_detection: bool = True
    signature_update_interval: int = 3600  # seconds
    profile_update_interval: int = 300  # seconds
    anomaly_threshold: float = 0.1  # Lower = more sensitive
    max_events_memory: int = 10000
    auto_response_enabled: bool = True
    learning_period_days: int = 30
    min_events_for_profile: int = 50


class IntrusionDetectionSystem:
    """
    Advanced Intrusion Detection System.
    
    Provides comprehensive intrusion detection using multiple techniques
    including signature-based detection, anomaly detection, behavioral
    analysis, and machine learning.
    """
    
    def __init__(self, config: Optional[IDSConfig] = None,
                 audit_manager: Optional[AuditManager] = None):
        """Initialize IDS."""
        self.config = config or IDSConfig()
        self.audit_manager = audit_manager
        self.logger = logging.getLogger(__name__)
        
        # Detection components
        self.signatures: Dict[str, IntrusionSignature] = {}
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.intrusion_events: Dict[str, IntrusionEvent] = {}
        
        # Machine learning models
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Event processing
        self.event_queue = deque(maxlen=self.config.max_events_memory)
        self.running = False
        self.processing_thread = None
        
        # Pattern tracking
        self.ip_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize components
        self._load_signatures()
        self._setup_ml_models()
        
        # Start processing
        self.start_monitoring()
    
    def _load_signatures(self):
        """Load intrusion detection signatures."""
        default_signatures = [
            IntrusionSignature(
                signature_id="brute_force_login",
                name="Brute Force Login Attack",
                description="Multiple failed login attempts",
                intrusion_type=IntrusionType.BRUTE_FORCE,
                pattern=r"authentication.*failed.*(\d+) attempts",
                severity=AuditSeverity.HIGH,
                confidence=0.9
            ),
            IntrusionSignature(
                signature_id="sql_injection_attempt",
                name="SQL Injection Attempt",
                description="SQL injection patterns detected",
                intrusion_type=IntrusionType.SQL_INJECTION,
                pattern=r"(union|select|insert|update|delete|drop|exec|script)",
                severity=AuditSeverity.CRITICAL,
                confidence=0.85
            ),
            IntrusionSignature(
                signature_id="xss_attempt",
                name="Cross-Site Scripting Attempt",
                description="XSS patterns detected",
                intrusion_type=IntrusionType.XSS_ATTACK,
                pattern=r"(<script|javascript:|on\w+\s*=)",
                severity=AuditSeverity.HIGH,
                confidence=0.8
            ),
            IntrusionSignature(
                signature_id="command_injection",
                name="Command Injection Attempt",
                description="Command injection patterns detected",
                intrusion_type=IntrusionType.COMMAND_INJECTION,
                pattern=r"(;|\||&|`|\$\(|>|<)",
                severity=AuditSeverity.CRITICAL,
                confidence=0.75
            ),
            IntrusionSignature(
                signature_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                description="Unauthorized privilege access",
                intrusion_type=IntrusionType.PRIVILEGE_ESCALATION,
                pattern=r"(sudo|su|admin|root|escalate)",
                severity=AuditSeverity.HIGH,
                confidence=0.7
            ),
            IntrusionSignature(
                signature_id="reconnaissance",
                name="Reconnaissance Activity",
                description="Scanning or probing activity",
                intrusion_type=IntrusionType.RECONNAISSANCE,
                pattern=r"(scan|probe|enum|directory|file_listing)",
                severity=AuditSeverity.MEDIUM,
                confidence=0.6
            )
        ]
        
        for signature in default_signatures:
            self.signatures[signature.signature_id] = signature
    
    def _setup_ml_models(self):
        """Setup machine learning models."""
        if self.config.enable_ml_detection:
            # Isolation Forest for anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=self.config.anomaly_threshold,
                random_state=42
            )
    
    def start_monitoring(self):
        """Start IDS monitoring."""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.processing_thread.start()
            self.logger.info("IDS monitoring started")
    
    def stop_monitoring(self):
        """Stop IDS monitoring."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("IDS monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Process recent audit events
                if self.audit_manager:
                    self._process_audit_events()
                
                # Update behavioral profiles
                self._update_behavioral_profiles()
                
                # Retrain ML models if needed
                if self.config.enable_ml_detection:
                    self._retrain_ml_models()
                
                time.sleep(self.config.profile_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in IDS monitoring loop: {e}")
                time.sleep(10)
    
    def _process_audit_events(self):
        """Process recent audit events for intrusions."""
        try:
            # Get events from last update interval
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=self.config.profile_update_interval)
            
            events = self.audit_manager.query_events(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            for event in events:
                self.event_queue.append(event)
                self._analyze_event(event)
                
        except Exception as e:
            self.logger.error(f"Error processing audit events: {e}")
    
    def _analyze_event(self, event: Dict[str, Any]):
        """Analyze individual event for intrusions."""
        # Signature-based detection
        if self.config.enable_signature_detection:
            self._signature_detection(event)
        
        # Behavioral analysis
        if self.config.enable_behavioral_analysis:
            self._behavioral_analysis(event)
        
        # Anomaly detection
        if self.config.enable_anomaly_detection:
            self._anomaly_detection(event)
        
        # Update activity tracking
        self._update_activity_tracking(event)
    
    def _signature_detection(self, event: Dict[str, Any]):
        """Perform signature-based detection."""
        for signature in self.signatures.values():
            if not signature.enabled:
                continue
            
            # Check event content against signature pattern
            event_text = self._event_to_text(event)
            
            if re.search(signature.pattern, event_text, re.IGNORECASE):
                intrusion = IntrusionEvent(
                    intrusion_type=signature.intrusion_type,
                    detection_type=DetectionType.SIGNATURE,
                    severity=signature.severity,
                    confidence=signature.confidence,
                    source_ip=event.get('ip_address'),
                    user_id=event.get('user_id'),
                    signature_id=signature.signature_id,
                    description=f"Signature match: {signature.name}",
                    raw_data=event
                )
                
                self._handle_intrusion(intrusion)
    
    def _behavioral_analysis(self, event: Dict[str, Any]):
        """Perform behavioral analysis."""
        user_id = event.get('user_id')
        if not user_id:
            return
        
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Check for behavioral anomalies
        anomalies = self._detect_behavioral_anomalies(event, profile)
        
        for anomaly in anomalies:
            intrusion = IntrusionEvent(
                intrusion_type=anomaly['type'],
                detection_type=DetectionType.BEHAVIORAL,
                severity=anomaly['severity'],
                confidence=anomaly['confidence'],
                source_ip=event.get('ip_address'),
                user_id=user_id,
                description=anomaly['description'],
                raw_data=event
            )
            
            self._handle_intrusion(intrusion)
        
        # Update profile
        profile.update_profile(event)
    
    def _detect_behavioral_anomalies(self, event: Dict[str, Any], 
                                   profile: UserBehaviorProfile) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies."""
        anomalies = []
        
        # Unusual login time
        if event.get('event_type') == 'authentication' and event.get('success'):
            login_hour = datetime.fromisoformat(event['timestamp']).hour
            
            if profile.login_times:
                # Calculate if login hour is unusual (outside 2 standard deviations)
                mean_hour = np.mean(profile.login_times)
                std_hour = np.std(profile.login_times)
                
                if abs(login_hour - mean_hour) > 2 * std_hour:
                    anomalies.append({
                        'type': IntrusionType.INSIDER_THREAT,
                        'severity': AuditSeverity.MEDIUM,
                        'confidence': 0.6,
                        'description': f"Unusual login time: {login_hour}:00"
                    })
        
        # New location
        if event.get('ip_address') and event.get('event_type') == 'authentication':
            if (profile.login_locations and 
                event['ip_address'] not in profile.login_locations):
                
                # Check if it's a completely new geographic location
                anomalies.append({
                    'type': IntrusionType.INSIDER_THREAT,
                    'severity': AuditSeverity.MEDIUM,
                    'confidence': 0.7,
                    'description': f"New login location: {event['ip_address']}"
                })
        
        # Rapid successive logins from different IPs
        if event.get('event_type') == 'authentication':
            recent_logins = [e for e in self.user_activity[profile.user_id] 
                           if (datetime.now() - e['timestamp']).seconds < 300]  # 5 minutes
            
            unique_ips = set(e.get('ip_address') for e in recent_logins 
                           if e.get('ip_address'))
            
            if len(unique_ips) > 3:  # More than 3 different IPs in 5 minutes
                anomalies.append({
                    'type': IntrusionType.INSIDER_THREAT,
                    'severity': AuditSeverity.HIGH,
                    'confidence': 0.8,
                    'description': f"Rapid logins from {len(unique_ips)} different IPs"
                })
        
        return anomalies
    
    def _anomaly_detection(self, event: Dict[str, Any]):
        """Perform statistical anomaly detection."""
        if not self.is_trained:
            return
        
        # Extract features from event
        features = self._extract_features(event)
        
        if features is None:
            return
        
        # Predict anomaly
        try:
            features_scaled = self.scaler.transform([features])
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            
            if is_anomaly:
                confidence = min(0.9, abs(anomaly_score) / 2)  # Normalize to 0-0.9
                
                intrusion = IntrusionEvent(
                    intrusion_type=IntrusionType.RECONNAISSANCE,  # Default type
                    detection_type=DetectionType.MACHINE_LEARNING,
                    severity=AuditSeverity.MEDIUM,
                    confidence=confidence,
                    source_ip=event.get('ip_address'),
                    user_id=event.get('user_id'),
                    description=f"ML anomaly detected (score: {anomaly_score:.3f})",
                    raw_data=event
                )
                
                self._handle_intrusion(intrusion)
                
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {e}")
    
    def _extract_features(self, event: Dict[str, Any]) -> Optional[List[float]]:
        """Extract numerical features from event for ML."""
        try:
            features = []
            
            # Time-based features
            timestamp = datetime.fromisoformat(event['timestamp'])
            features.extend([
                timestamp.hour,
                timestamp.weekday(),
                timestamp.minute
            ])
            
            # Event type encoding
            event_types = {
                'authentication': 1, 'authorization': 2, 'data_access': 3,
                'data_modification': 4, 'security_violation': 5, 'error': 6
            }
            features.append(event_types.get(event.get('event_type'), 0))
            
            # Success/failure
            features.append(1 if event.get('success') else 0)
            
            # IP address features (simplified)
            ip_address = event.get('ip_address', '0.0.0.0')
            try:
                ip_int = int(ipaddress.ip_address(ip_address))
                features.extend([
                    (ip_int >> 24) & 0xFF,  # First octet
                    (ip_int >> 16) & 0xFF,  # Second octet
                ])
            except:
                features.extend([0, 0])
            
            # String length features
            features.extend([
                len(event.get('action', '')),
                len(event.get('resource', '')),
                len(str(event.get('details', {})))
            ])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _update_activity_tracking(self, event: Dict[str, Any]):
        """Update activity tracking for pattern analysis."""
        # Track IP activity
        ip_address = event.get('ip_address')
        if ip_address:
            self.ip_activity[ip_address].append({
                'timestamp': datetime.fromisoformat(event['timestamp']),
                'event_type': event.get('event_type'),
                'success': event.get('success'),
                'action': event.get('action')
            })
        
        # Track user activity
        user_id = event.get('user_id')
        if user_id:
            self.user_activity[user_id].append({
                'timestamp': datetime.fromisoformat(event['timestamp']),
                'ip_address': ip_address,
                'event_type': event.get('event_type'),
                'success': event.get('success'),
                'action': event.get('action')
            })
    
    def _update_behavioral_profiles(self):
        """Update all user behavioral profiles."""
        # Clean up old profiles
        cutoff_time = datetime.now() - timedelta(days=self.config.learning_period_days)
        
        profiles_to_remove = [
            user_id for user_id, profile in self.user_profiles.items()
            if profile.last_updated < cutoff_time
        ]
        
        for user_id in profiles_to_remove:
            del self.user_profiles[user_id]
    
    def _retrain_ml_models(self):
        """Retrain machine learning models with recent data."""
        if len(self.event_queue) < self.config.min_events_for_profile:
            return
        
        try:
            # Extract features from recent events
            features = []
            for event in list(self.event_queue):
                event_features = self._extract_features(event)
                if event_features:
                    features.append(event_features)
            
            if len(features) < 50:  # Need minimum data
                return
            
            # Fit scaler and anomaly detector
            features_array = np.array(features)
            self.scaler.fit(features_array)
            
            features_scaled = self.scaler.transform(features_array)
            self.anomaly_detector.fit(features_scaled)
            
            self.is_trained = True
            self.logger.info(f"Retrained ML models with {len(features)} samples")
            
        except Exception as e:
            self.logger.error(f"Error retraining ML models: {e}")
    
    def _handle_intrusion(self, intrusion: IntrusionEvent):
        """Handle detected intrusion."""
        # Store intrusion event
        self.intrusion_events[intrusion.event_id] = intrusion
        
        # Log intrusion
        self.logger.warning(
            f"INTRUSION DETECTED: {intrusion.intrusion_type.value} "
            f"(confidence: {intrusion.confidence:.2f}) - {intrusion.description}"
        )
        
        # Log to audit system
        if self.audit_manager:
            self.audit_manager.log_security_violation(
                user_id=intrusion.user_id or "unknown",
                violation_type=intrusion.intrusion_type.value,
                ip_address=intrusion.source_ip or "unknown",
                details={
                    'intrusion_id': intrusion.event_id,
                    'detection_type': intrusion.detection_type.value,
                    'confidence': intrusion.confidence,
                    'signature_id': intrusion.signature_id,
                    'raw_data': intrusion.raw_data
                }
            )
        
        # Automated response
        if self.config.auto_response_enabled:
            self._automated_response(intrusion)
    
    def _automated_response(self, intrusion: IntrusionEvent):
        """Execute automated response to intrusion."""
        response_actions = self._determine_response_actions(intrusion)
        
        for action in response_actions:
            try:
                self._execute_response_action(action, intrusion)
                intrusion.response_actions.append(action)
            except Exception as e:
                self.logger.error(f"Error executing response action {action.value}: {e}")
    
    def _determine_response_actions(self, intrusion: IntrusionEvent) -> List[ResponseAction]:
        """Determine appropriate response actions."""
        actions = [ResponseAction.LOG_ALERT]  # Always log
        
        # High confidence intrusions
        if intrusion.confidence > 0.8:
            if intrusion.intrusion_type == IntrusionType.BRUTE_FORCE:
                actions.extend([ResponseAction.BLOCK_IP, ResponseAction.RATE_LIMIT])
            elif intrusion.intrusion_type in [IntrusionType.SQL_INJECTION, IntrusionType.XSS_ATTACK]:
                actions.extend([ResponseAction.BLOCK_IP, ResponseAction.EMAIL_ADMIN])
            elif intrusion.intrusion_type == IntrusionType.PRIVILEGE_ESCALATION:
                actions.extend([ResponseAction.QUARANTINE_USER, ResponseAction.EMAIL_ADMIN])
        
        # Medium confidence intrusions
        elif intrusion.confidence > 0.6:
            actions.extend([ResponseAction.RATE_LIMIT, ResponseAction.INCREASE_MONITORING])
        
        # Critical severity always gets admin notification
        if intrusion.severity == AuditSeverity.CRITICAL:
            actions.append(ResponseAction.EMAIL_ADMIN)
        
        return list(set(actions))  # Remove duplicates
    
    def _execute_response_action(self, action: ResponseAction, intrusion: IntrusionEvent):
        """Execute specific response action."""
        if action == ResponseAction.BLOCK_IP and intrusion.source_ip:
            # Would integrate with firewall or DDoS protection
            self.logger.info(f"RESPONSE: Blocking IP {intrusion.source_ip}")
        
        elif action == ResponseAction.RATE_LIMIT and intrusion.source_ip:
            # Would integrate with rate limiting system
            self.logger.info(f"RESPONSE: Rate limiting IP {intrusion.source_ip}")
        
        elif action == ResponseAction.QUARANTINE_USER and intrusion.user_id:
            # Would integrate with user management system
            self.logger.info(f"RESPONSE: Quarantining user {intrusion.user_id}")
        
        elif action == ResponseAction.EMAIL_ADMIN:
            # Would send email to security team
            self.logger.info(f"RESPONSE: Sending admin alert for intrusion {intrusion.event_id}")
        
        elif action == ResponseAction.DISABLE_ACCOUNT and intrusion.user_id:
            # Would disable user account
            self.logger.info(f"RESPONSE: Disabling account {intrusion.user_id}")
        
        elif action == ResponseAction.INCREASE_MONITORING:
            # Would increase monitoring for source
            self.logger.info(f"RESPONSE: Increasing monitoring for {intrusion.source_ip or intrusion.user_id}")
        
        elif action == ResponseAction.CAPTURE_TRAFFIC and intrusion.source_ip:
            # Would start packet capture
            self.logger.info(f"RESPONSE: Starting traffic capture for {intrusion.source_ip}")
    
    def _event_to_text(self, event: Dict[str, Any]) -> str:
        """Convert event to text for pattern matching."""
        text_parts = [
            event.get('action', ''),
            event.get('resource', ''),
            str(event.get('details', {})),
            event.get('error_message', ''),
            event.get('user_agent', '')
        ]
        
        return ' '.join(filter(None, text_parts)).lower()
    
    def get_intrusion_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get intrusion detection summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_intrusions = [
            intrusion for intrusion in self.intrusion_events.values()
            if intrusion.timestamp >= cutoff_time
        ]
        
        # Count by type
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for intrusion in recent_intrusions:
            by_type[intrusion.intrusion_type.value] += 1
            by_severity[intrusion.severity.value] += 1
        
        # Top source IPs
        source_ips = defaultdict(int)
        for intrusion in recent_intrusions:
            if intrusion.source_ip:
                source_ips[intrusion.source_ip] += 1
        
        top_ips = sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_intrusions': len(recent_intrusions),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'top_source_ips': top_ips,
            'detection_types': {
                dt.value: sum(1 for i in recent_intrusions if i.detection_type == dt)
                for dt in DetectionType
            },
            'unresolved_count': sum(1 for i in recent_intrusions if not i.resolved),
            'ml_model_trained': self.is_trained,
            'user_profiles': len(self.user_profiles),
            'signatures_enabled': sum(1 for s in self.signatures.values() if s.enabled)
        }
    
    def add_signature(self, signature: IntrusionSignature):
        """Add custom intrusion signature."""
        self.signatures[signature.signature_id] = signature
        self.logger.info(f"Added intrusion signature: {signature.name}")
    
    def resolve_intrusion(self, intrusion_id: str, notes: str = ""):
        """Mark intrusion as resolved."""
        if intrusion_id in self.intrusion_events:
            self.intrusion_events[intrusion_id].resolved = True
            self.logger.info(f"Resolved intrusion: {intrusion_id}")


# Example usage and testing
if __name__ == "__main__":
    from audit_system import AuditConfig, AuditManager
    
    # Initialize components
    audit_config = AuditConfig()
    audit_manager = AuditManager(audit_config)
    
    ids_config = IDSConfig()
    ids = IntrusionDetectionSystem(ids_config, audit_manager)
    
    # Test IDS functionality
    print("=== Intrusion Detection System Test ===")
    
    # Generate test intrusion events
    audit_manager.log_authentication("user1", False, "192.168.1.100")
    audit_manager.log_authentication("user1", False, "192.168.1.100")
    audit_manager.log_authentication("user1", False, "192.168.1.100")
    audit_manager.log_authentication("user1", False, "192.168.1.100")
    audit_manager.log_authentication("user1", False, "192.168.1.100")
    
    audit_manager.log_security_violation("user2", "sql_injection", "192.168.1.200", 
                                        {"payload": "'; DROP TABLE users; --"})
    
    # Wait for processing
    time.sleep(3)
    
    # Get intrusion summary
    summary = ids.get_intrusion_summary()
    print(f"Total intrusions detected: {summary['total_intrusions']}")
    print(f"Intrusions by type: {summary['by_type']}")
    print(f"Intrusions by severity: {summary['by_severity']}")
    print(f"Top source IPs: {summary['top_source_ips']}")
    print(f"ML model trained: {summary['ml_model_trained']}")
    print(f"User profiles: {summary['user_profiles']}")
    
    # Stop components
    ids.stop_monitoring()
    audit_manager.stop_processing()
