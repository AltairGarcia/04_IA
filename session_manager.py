"""
Enhanced Session Management System

This module provides secure session storage, session hijacking protection,
concurrent session management, and session timeout handling.
"""

import os
import json
import secrets
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import ipaddress
from user_agents import parse as parse_user_agent
import jwt

logger = logging.getLogger(__name__)

@dataclass
class Session:
    """Session data structure."""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    device_fingerprint: str
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    session_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.session_data is None:
            self.session_data = {}

@dataclass
class SessionSecurity:
    """Session security information."""
    session_id: str
    security_level: str  # 'low', 'medium', 'high'
    risk_score: float
    anomalies: List[str]
    verification_required: bool = False
    mfa_verified: bool = False
    
@dataclass
class DeviceInfo:
    """Device information for fingerprinting."""
    browser: str
    os: str
    device_type: str
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None

class SessionManager:
    """Manages secure user sessions with advanced security features."""
    
    def __init__(self, db_path: str = "sessions.db", encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Session configuration
        self.session_timeout = timedelta(hours=24)
        self.idle_timeout = timedelta(hours=2)
        self.max_sessions_per_user = 5
        self.session_rotation_interval = timedelta(hours=4)
        
        # Security thresholds
        self.max_risk_score = 0.8
        self.location_change_threshold = 100  # km
        self.suspicious_activity_threshold = 5
        
        # JWT configuration for session tokens
        self.jwt_secret = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize session database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        ip_address TEXT NOT NULL,
                        user_agent TEXT NOT NULL,
                        device_fingerprint TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        session_data TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Session security table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_security (
                        session_id TEXT PRIMARY KEY,
                        security_level TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        anomalies TEXT,
                        verification_required BOOLEAN DEFAULT 0,
                        mfa_verified BOOLEAN DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Session events table for monitoring
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Device fingerprints table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS device_fingerprints (
                        fingerprint TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        device_info TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_trusted BOOLEAN DEFAULT 0,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events (session_id)")
                
                conn.commit()
                logger.info("Session database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize session database: {e}")
            raise
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str, 
                      device_info: Optional[DeviceInfo] = None) -> Optional[str]:
        """Create a new secure session."""
        try:
            # Generate secure session ID
            session_id = secrets.token_urlsafe(32)
            
            # Create device fingerprint
            device_fingerprint = self._create_device_fingerprint(user_agent, device_info)
            
            # Calculate session expiry
            expires_at = datetime.now() + self.session_timeout
            
            # Create session
            session = Session(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint,
                expires_at=expires_at
            )
            
            # Assess security risk
            security = self._assess_session_security(session, device_info)
            
            # Check concurrent sessions limit
            if not self._enforce_session_limit(user_id):
                logger.warning(f"Session limit exceeded for user {user_id}")
                return None
            
            # Save session
            self._save_session(session)
            self._save_session_security(security)
            self._save_device_fingerprint(user_id, device_fingerprint, device_info)
            
            # Log session creation
            self._log_session_event(session_id, "session_created", {
                "user_id": user_id,
                "security_level": security.security_level,
                "risk_score": security.risk_score
            }, ip_address, user_agent)
            
            logger.info(f"Session created for user {user_id}, security level: {security.security_level}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def validate_session(self, session_id: str, ip_address: str, user_agent: str) -> Optional[Session]:
        """Validate and update session."""
        try:
            session = self._get_session(session_id)
            if not session:
                return None
            
            # Check if session is expired or inactive
            if not session.is_active or datetime.now() > session.expires_at:
                self.invalidate_session(session_id)
                return None
            
            # Check for session hijacking
            if self._detect_session_hijacking(session, ip_address, user_agent):
                logger.warning(f"Session hijacking detected for session {session_id}")
                self.invalidate_session(session_id, reason="hijacking_detected")
                return None
            
            # Update last activity
            self._update_session_activity(session_id)
            
            # Check if session rotation is needed
            if self._should_rotate_session(session):
                new_session_id = self._rotate_session(session)
                if new_session_id:
                    session.session_id = new_session_id
            
            # Log session validation
            self._log_session_event(session_id, "session_validated", {
                "user_id": session.user_id
            }, ip_address, user_agent)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to validate session: {e}")
            return None
    
    def invalidate_session(self, session_id: str, reason: str = "user_logout") -> bool:
        """Invalidate a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET is_active = 0 
                    WHERE session_id = ?
                """, (session_id,))
                
                success = cursor.rowcount > 0
                
                if success:
                    # Log session invalidation
                    self._log_session_event(session_id, "session_invalidated", {
                        "reason": reason
                    })
                    logger.info(f"Session {session_id} invalidated: {reason}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
            return False
    
    def invalidate_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """Invalidate all sessions for a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if except_session:
                    cursor.execute("""
                        UPDATE sessions 
                        SET is_active = 0 
                        WHERE user_id = ? AND session_id != ? AND is_active = 1
                    """, (user_id, except_session))
                else:
                    cursor.execute("""
                        UPDATE sessions 
                        SET is_active = 0 
                        WHERE user_id = ? AND is_active = 1
                    """, (user_id,))
                
                invalidated_count = cursor.rowcount
                
                logger.info(f"Invalidated {invalidated_count} sessions for user {user_id}")
                return invalidated_count
                
        except Exception as e:
            logger.error(f"Failed to invalidate user sessions: {e}")
            return 0
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.session_id, s.ip_address, s.user_agent, s.created_at, 
                           s.last_activity, ss.security_level, ss.risk_score
                    FROM sessions s
                    LEFT JOIN session_security ss ON s.session_id = ss.session_id
                    WHERE s.user_id = ? AND s.is_active = 1 AND s.expires_at > ?
                    ORDER BY s.last_activity DESC
                """, (user_id, datetime.now()))
                
                sessions = []
                for row in cursor.fetchall():
                    # Parse user agent for display
                    ua = parse_user_agent(row[2])
                    
                    sessions.append({
                        'session_id': row[0],
                        'ip_address': row[1],
                        'browser': f"{ua.browser.family} {ua.browser.version_string}",
                        'os': f"{ua.os.family} {ua.os.version_string}",
                        'device': ua.device.family,
                        'created_at': row[3],
                        'last_activity': row[4],
                        'security_level': row[5] or 'medium',
                        'risk_score': row[6] or 0.0
                    })
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def update_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        try:
            # Encrypt session data
            encrypted_data = self.cipher_suite.encrypt(json.dumps(data).encode()).decode()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET session_data = ? 
                    WHERE session_id = ? AND is_active = 1
                """, (encrypted_data, session_id))
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update session data: {e}")
            return False
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_data FROM sessions 
                    WHERE session_id = ? AND is_active = 1
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row or not row[0]:
                    return {}
                
                # Decrypt session data
                decrypted_data = self.cipher_suite.decrypt(row[0].encode()).decode()
                return json.loads(decrypted_data)
                
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return {}
    
    def create_session_token(self, session_id: str, user_id: str) -> str:
        """Create JWT token for session."""
        payload = {
            'session_id': session_id,
            'user_id': user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + self.session_timeout
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_session_token(self, token: str) -> Optional[Dict[str, str]]:
        """Verify JWT session token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Verify session is still active
            session = self._get_session(payload['session_id'])
            if not session or not session.is_active:
                return None
            
            return {
                'session_id': payload['session_id'],
                'user_id': payload['user_id']
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Session token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid session token")
            return None
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Mark expired sessions as inactive
                cursor.execute("""
                    UPDATE sessions 
                    SET is_active = 0 
                    WHERE expires_at < ? AND is_active = 1
                """, (datetime.now(),))
                
                expired_count = cursor.rowcount
                
                # Clean up old session data (older than 30 days)
                cleanup_date = datetime.now() - timedelta(days=30)
                cursor.execute("""
                    DELETE FROM session_events 
                    WHERE timestamp < ?
                """, (cleanup_date,))
                
                cursor.execute("""
                    DELETE FROM session_security 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE expires_at < ? AND is_active = 0
                    )
                """, (cleanup_date,))
                
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE expires_at < ? AND is_active = 0
                """, (cleanup_date,))
                
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired sessions")
                
                return expired_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def _create_device_fingerprint(self, user_agent: str, device_info: Optional[DeviceInfo] = None) -> str:
        """Create device fingerprint for session tracking."""
        ua = parse_user_agent(user_agent)
        
        fingerprint_data = {
            'browser': ua.browser.family,
            'browser_version': ua.browser.version_string,
            'os': ua.os.family,
            'os_version': ua.os.version_string,
            'device': ua.device.family
        }
        
        if device_info:
            fingerprint_data.update({
                'screen_resolution': device_info.screen_resolution,
                'timezone': device_info.timezone,
                'language': device_info.language
            })
        
        # Create hash of fingerprint data
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def _assess_session_security(self, session: Session, device_info: Optional[DeviceInfo] = None) -> SessionSecurity:
        """Assess security risk for a session."""
        risk_score = 0.0
        anomalies = []
        security_level = "medium"
        
        # Check if device is known and trusted
        if not self._is_trusted_device(session.user_id, session.device_fingerprint):
            risk_score += 0.3
            anomalies.append("unknown_device")
        
        # Check IP address patterns
        if self._is_suspicious_ip(session.ip_address):
            risk_score += 0.4
            anomalies.append("suspicious_ip")
        
        # Check for concurrent sessions from different locations
        if self._has_location_anomaly(session.user_id, session.ip_address):
            risk_score += 0.2
            anomalies.append("location_anomaly")
        
        # Determine security level
        if risk_score >= 0.7:
            security_level = "high"
        elif risk_score >= 0.4:
            security_level = "medium"
        else:
            security_level = "low"
        
        verification_required = risk_score >= 0.5
        
        return SessionSecurity(
            session_id=session.session_id,
            security_level=security_level,
            risk_score=risk_score,
            anomalies=anomalies,
            verification_required=verification_required
        )
    
    def _detect_session_hijacking(self, session: Session, current_ip: str, current_ua: str) -> bool:
        """Detect potential session hijacking."""
        # Check for significant IP address change
        if session.ip_address != current_ip:
            # Allow for reasonable IP changes (same subnet/ISP)
            try:
                session_network = ipaddress.ip_network(f"{session.ip_address}/24", strict=False)
                current_ip_obj = ipaddress.ip_address(current_ip)
                if current_ip_obj not in session_network:
                    return True
            except:
                return True
        
        # Check for user agent changes (browser/device switching)
        session_ua = parse_user_agent(session.user_agent)
        current_ua_parsed = parse_user_agent(current_ua)
        
        if (session_ua.browser.family != current_ua_parsed.browser.family or 
            session_ua.os.family != current_ua_parsed.os.family):
            return True
        
        return False
    
    def _should_rotate_session(self, session: Session) -> bool:
        """Check if session should be rotated."""
        if not session.created_at:
            return False
        
        time_since_creation = datetime.now() - session.created_at
        return time_since_creation >= self.session_rotation_interval
    
    def _rotate_session(self, session: Session) -> Optional[str]:
        """Rotate session ID for security."""
        try:
            new_session_id = secrets.token_urlsafe(32)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update session ID
                cursor.execute("""
                    UPDATE sessions 
                    SET session_id = ? 
                    WHERE session_id = ?
                """, (new_session_id, session.session_id))
                
                # Update security record
                cursor.execute("""
                    UPDATE session_security 
                    SET session_id = ? 
                    WHERE session_id = ?
                """, (new_session_id, session.session_id))
                
                # Log rotation
                self._log_session_event(new_session_id, "session_rotated", {
                    "old_session_id": session.session_id
                })
                
                logger.info(f"Session rotated: {session.session_id} -> {new_session_id}")
                return new_session_id
                
        except Exception as e:
            logger.error(f"Failed to rotate session: {e}")
            return None
    
    def _enforce_session_limit(self, user_id: str) -> bool:
        """Enforce maximum sessions per user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM sessions 
                    WHERE user_id = ? AND is_active = 1 AND expires_at > ?
                """, (user_id, datetime.now()))
                
                current_sessions = cursor.fetchone()[0]
                
                if current_sessions >= self.max_sessions_per_user:
                    # Remove oldest session
                    cursor.execute("""
                        UPDATE sessions 
                        SET is_active = 0 
                        WHERE user_id = ? AND is_active = 1 
                        ORDER BY last_activity ASC 
                        LIMIT 1
                    """, (user_id,))
                    
                    logger.info(f"Removed oldest session for user {user_id} due to limit")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to enforce session limit: {e}")
            return False
    
    def _save_session(self, session: Session):
        """Save session to database."""
        encrypted_data = None
        if session.session_data:
            encrypted_data = self.cipher_suite.encrypt(json.dumps(session.session_data).encode()).decode()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions 
                (session_id, user_id, ip_address, user_agent, device_fingerprint, 
                 created_at, last_activity, expires_at, session_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.user_id, session.ip_address,
                session.user_agent, session.device_fingerprint,
                session.created_at, session.last_activity, session.expires_at,
                encrypted_data
            ))
    
    def _save_session_security(self, security: SessionSecurity):
        """Save session security information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_security 
                (session_id, security_level, risk_score, anomalies, verification_required, mfa_verified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                security.session_id, security.security_level, security.risk_score,
                json.dumps(security.anomalies), security.verification_required, security.mfa_verified
            ))
    
    def _save_device_fingerprint(self, user_id: str, fingerprint: str, device_info: Optional[DeviceInfo]):
        """Save device fingerprint."""
        device_info_json = json.dumps(asdict(device_info)) if device_info else "{}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO device_fingerprints 
                (fingerprint, user_id, device_info, last_seen)
                VALUES (?, ?, ?, ?)
            """, (fingerprint, user_id, device_info_json, datetime.now()))
    
    def _get_session(self, session_id: str) -> Optional[Session]:
        """Get session from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                session_data = {}
                if row[9]:  # session_data column
                    try:
                        decrypted_data = self.cipher_suite.decrypt(row[9].encode()).decode()
                        session_data = json.loads(decrypted_data)
                    except:
                        pass
                
                return Session(
                    session_id=row[0],
                    user_id=row[1],
                    ip_address=row[2],
                    user_agent=row[3],
                    device_fingerprint=row[4],
                    is_active=bool(row[5]),
                    created_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    last_activity=datetime.fromisoformat(row[7]) if row[7] else None,
                    expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    session_data=session_data
                )
                
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def _update_session_activity(self, session_id: str):
        """Update session last activity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET last_activity = ? 
                WHERE session_id = ?
            """, (datetime.now(), session_id))
    
    def _log_session_event(self, session_id: str, event_type: str, event_data: Dict[str, Any] = None, 
                          ip_address: str = None, user_agent: str = None):
        """Log session event."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_events 
                (session_id, event_type, event_data, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id, event_type, 
                json.dumps(event_data) if event_data else None,
                ip_address, user_agent
            ))
    
    def _is_trusted_device(self, user_id: str, fingerprint: str) -> bool:
        """Check if device is trusted."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT is_trusted FROM device_fingerprints 
                    WHERE user_id = ? AND fingerprint = ?
                """, (user_id, fingerprint))
                
                row = cursor.fetchone()
                return row and bool(row[0])
                
        except Exception as e:
            logger.error(f"Failed to check trusted device: {e}")
            return False
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious."""
        # Implement IP reputation checking
        # This could integrate with threat intelligence feeds
        suspicious_ranges = [
            "10.0.0.0/8",      # Private networks might be suspicious in some contexts
            "172.16.0.0/12",   # Private networks
            "192.168.0.0/16"   # Private networks
        ]
        
        try:
            ip = ipaddress.ip_address(ip_address)
            for range_str in suspicious_ranges:
                network = ipaddress.ip_network(range_str)
                if ip in network:
                    return True
        except:
            pass
        
        return False
    
    def _has_location_anomaly(self, user_id: str, ip_address: str) -> bool:
        """Check for location anomalies in user sessions."""
        # This would integrate with IP geolocation services
        # For now, implement basic check for concurrent sessions from different IPs
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT ip_address FROM sessions 
                    WHERE user_id = ? AND is_active = 1 AND ip_address != ?
                    AND last_activity > ?
                """, (user_id, ip_address, datetime.now() - timedelta(hours=1)))
                
                other_ips = cursor.fetchall()
                return len(other_ips) > 0
                
        except Exception as e:
            logger.error(f"Failed to check location anomaly: {e}")
            return False
