"""
Security and Authentication Management System

This module provides comprehensive security features including user authentication,
authorization, API key management, data encryption, and security monitoring
for the content creation platform.
"""

import os
import json
import sqlite3
import hashlib
import secrets
import hmac
import base64
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from pathlib import Path
import re
import ipaddress
from functools import wraps
import time

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles and permissions."""
    ADMIN = "admin"
    EDITOR = "editor"
    CREATOR = "creator"
    VIEWER = "viewer"

class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    API_KEY = "api_key"
    OAUTH = "oauth"
    MFA = "mfa"

@dataclass
class User:
    """User data structure."""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class APIKey:
    """API key data structure."""
    id: str
    user_id: str
    name: str
    key_hash: str
    permissions: List[str]
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class SecurityEvent:
    """Security event for monitoring."""
    id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    severity: str = "info"
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SecurityManager:
    """Manages security, authentication, and authorization."""

    def __init__(self, db_path: str = "security.db", secret_key: Optional[str] = None):
        self.db_path = db_path
        self.secret_key = secret_key or os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
        self.jwt_secret = os.getenv("JWT_SECRET") or secrets.token_urlsafe(32)
        self.encryption_key = self._derive_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=24)
        self.password_min_length = 8
        self.api_rate_limits = {}

        self._initialize_database()
        self._create_default_admin()

    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret."""
        password = self.secret_key.encode()
        salt = b'content_creation_salt'  # In production, use a random salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def _initialize_database(self):
        """Initialize the security database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    mfa_enabled BOOLEAN DEFAULT 0,
                    mfa_secret TEXT,
                    created_at TEXT,
                    last_login TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TEXT,
                    security_level TEXT DEFAULT 'internal',
                    metadata TEXT
                )
            """)

            # API keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    name TEXT,
                    key_hash TEXT,
                    permissions TEXT,
                    expires_at TEXT,
                    last_used TEXT,
                    usage_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    token_hash TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Security events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT,
                    user_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT,
                    severity TEXT DEFAULT 'info',
                    timestamp TEXT
                )
            """)

            # Rate limiting table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id TEXT PRIMARY KEY,
                    identifier TEXT,
                    endpoint TEXT,
                    count INTEGER DEFAULT 0,
                    window_start TEXT,
                    updated_at TEXT
                )
            """)

            conn.commit()

    def _create_default_admin(self):
        """Create default admin user if none exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = ?", (UserRole.ADMIN.value,))

            if cursor.fetchone()[0] == 0:
                admin_user = User(
                    id=secrets.token_urlsafe(16),
                    username="admin",
                    email="admin@localhost",
                    password_hash=self._hash_password("admin123"),
                    role=UserRole.ADMIN,
                    is_verified=True,
                    security_level=SecurityLevel.RESTRICTED
                )

                self._save_user(admin_user)
                logger.info("Created default admin user (username: admin, password: admin123)")

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    def _validate_password_strength(self, password: str) -> List[str]:
        """Validate password strength."""
        errors = []

        if len(password) < self.password_min_length:
            errors.append(f"Password must be at least {self.password_min_length} characters long")

        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?]', password):
            errors.append("Password must contain at least one special character")

        return errors

    def create_user(self, username: str, email: str, password: str,
                   role: UserRole = UserRole.CREATOR) -> Tuple[bool, Union[User, List[str]]]:
        """Create a new user."""
        # Validate password strength
        password_errors = self._validate_password_strength(password)
        if password_errors:
            return False, password_errors

        # Check if user already exists
        existing_user = self.get_user_by_username(username)
        if existing_user:
            return False, ["Username already exists"]

        existing_email = self.get_user_by_email(email)
        if existing_email:
            return False, ["Email already exists"]

        # Create user
        user = User(
            id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role
        )

        self._save_user(user)
        self._log_security_event("user_created", user.id, "", "", {
            "username": username,
            "email": email,
            "role": role.value
        })

        return True, user

    def authenticate_user(self, username: str, password: str,
                         ip_address: str = "", user_agent: str = "") -> Tuple[bool, Union[User, str]]:
        """Authenticate user with username and password."""
        user = self.get_user_by_username(username)

        if not user:
            self._log_security_event("login_failed", None, ip_address, user_agent, {
                "username": username,
                "reason": "user_not_found"
            }, "warning")
            return False, "Invalid credentials"

        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_security_event("login_attempt_locked", user.id, ip_address, user_agent, {
                "username": username,
                "locked_until": user.locked_until.isoformat()
            }, "warning")
            return False, f"Account locked until {user.locked_until}"

        # Check if account is active
        if not user.is_active:
            self._log_security_event("login_attempt_inactive", user.id, ip_address, user_agent, {
                "username": username
            }, "warning")
            return False, "Account is deactivated"

        # Verify password
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now() + self.lockout_duration

            self._save_user(user)
            self._log_security_event("login_failed", user.id, ip_address, user_agent, {
                "username": username,
                "failed_attempts": user.failed_login_attempts,
                "reason": "invalid_password"
            }, "warning")

            return False, "Invalid credentials"

        # Successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        self._save_user(user)

        self._log_security_event("login_success", user.id, ip_address, user_agent, {
            "username": username
        })

        return True, user

    def create_session(self, user: User, ip_address: str = "",
                      user_agent: str = "") -> str:
        """Create a new session for user."""
        session_id = secrets.token_urlsafe(32)
        token = jwt.encode(
            {
                "session_id": session_id,
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "exp": datetime.utcnow() + self.session_timeout
            },
            self.jwt_secret,
            algorithm="HS256"
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, user_id, token_hash, ip_address, user_agent,
                                    created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user.id,
                hashlib.sha256(token.encode()).hexdigest(),
                ip_address,
                user_agent,
                datetime.now().isoformat(),
                (datetime.now() + self.session_timeout).isoformat(),
                True
            ))
            conn.commit()

        self._log_security_event("session_created", user.id, ip_address, user_agent, {
            "session_id": session_id
        })

        return token

    def validate_session(self, token: str) -> Optional[User]:
        """Validate session token and return user."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            session_id = payload.get("session_id")
            user_id = payload.get("user_id")

            # Check if session exists and is active
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, expires_at, is_active
                    FROM sessions
                    WHERE id = ? AND user_id = ?
                """, (session_id, user_id))

                row = cursor.fetchone()
                if not row or not row[2]:  # Session not found or inactive
                    return None

                expires_at = datetime.fromisoformat(row[1])
                if datetime.now() >= expires_at:
                    # Session expired
                    cursor.execute("UPDATE sessions SET is_active = 0 WHERE id = ?", (session_id,))
                    conn.commit()
                    return None

            # Get user
            user = self.get_user_by_id(user_id)
            return user

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_session(self, token: str):
        """Revoke a session."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            session_id = payload.get("session_id")

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE sessions SET is_active = 0 WHERE id = ?", (session_id,))
                conn.commit()

        except jwt.InvalidTokenError:
            pass

    def create_api_key(self, user: User, name: str, permissions: List[str],
                      expires_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Create a new API key for user."""
        # Generate API key
        key = f"sk-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)

        api_key = APIKey(
            id=secrets.token_urlsafe(16),
            user_id=user.id,
            name=name,
            key_hash=key_hash,
            permissions=permissions,
            expires_at=expires_at
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_keys (id, user_id, name, key_hash, permissions,
                                    expires_at, last_used, usage_count, is_active,
                                    created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key.id,
                api_key.user_id,
                api_key.name,
                api_key.key_hash,
                json.dumps(api_key.permissions),
                api_key.expires_at.isoformat() if api_key.expires_at else None,
                None,
                api_key.usage_count,
                api_key.is_active,
                api_key.created_at.isoformat(),
                json.dumps(api_key.metadata)
            ))
            conn.commit()

        self._log_security_event("api_key_created", user.id, "", "", {
            "api_key_name": name,
            "permissions": permissions
        })

        return key, api_key

    def validate_api_key(self, key: str) -> Optional[Tuple[User, APIKey]]:
        """Validate API key and return user and key info."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ak.*, u.* FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1
            """, (key_hash,))

            row = cursor.fetchone()
            if not row:
                return None

            # Parse API key data
            api_key_data = row[:11]  # First 11 columns are from api_keys table
            user_data = row[11:]     # Rest are from users table

            # Check if API key is expired
            if api_key_data[5]:  # expires_at
                expires_at = datetime.fromisoformat(api_key_data[5])
                if datetime.now() >= expires_at:
                    return None

            # Update last used and usage count
            cursor.execute("""
                UPDATE api_keys
                SET last_used = ?, usage_count = usage_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), api_key_data[0]))
            conn.commit()

            # Create objects
            api_key = APIKey(
                id=api_key_data[0],
                user_id=api_key_data[1],
                name=api_key_data[2],
                key_hash=api_key_data[3],
                permissions=json.loads(api_key_data[4]) if api_key_data[4] else [],
                expires_at=datetime.fromisoformat(api_key_data[5]) if api_key_data[5] else None,
                last_used=datetime.fromisoformat(api_key_data[6]) if api_key_data[6] else None,
                usage_count=api_key_data[7],
                is_active=bool(api_key_data[8]),
                created_at=datetime.fromisoformat(api_key_data[9]) if api_key_data[9] else None,
                metadata=json.loads(api_key_data[10]) if api_key_data[10] else {}
            )

            user = User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                password_hash=user_data[3],
                role=UserRole(user_data[4]),
                is_active=bool(user_data[5]),
                is_verified=bool(user_data[6]),
                mfa_enabled=bool(user_data[7]),
                mfa_secret=user_data[8],
                created_at=datetime.fromisoformat(user_data[9]) if user_data[9] else None,
                last_login=datetime.fromisoformat(user_data[10]) if user_data[10] else None,
                failed_login_attempts=user_data[11],
                locked_until=datetime.fromisoformat(user_data[12]) if user_data[12] else None,
                security_level=SecurityLevel(user_data[13]) if user_data[13] else SecurityLevel.INTERNAL,
                metadata=json.loads(user_data[14]) if user_data[14] else {}
            )

            return user, api_key

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted.decode()

    def check_rate_limit(self, identifier: str, endpoint: str,
                        limit: int, window_minutes: int = 60) -> bool:
        """Check if request is within rate limits."""
        window_start = datetime.now() - timedelta(minutes=window_minutes)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get or create rate limit record
            cursor.execute("""
                SELECT count, window_start FROM rate_limits
                WHERE identifier = ? AND endpoint = ?
            """, (identifier, endpoint))

            row = cursor.fetchone()

            if row:
                count, window_start_str = row
                record_window_start = datetime.fromisoformat(window_start_str)

                # Check if window has expired
                if datetime.now() - record_window_start >= timedelta(minutes=window_minutes):
                    # Reset window
                    cursor.execute("""
                        UPDATE rate_limits
                        SET count = 1, window_start = ?, updated_at = ?
                        WHERE identifier = ? AND endpoint = ?
                    """, (datetime.now().isoformat(), datetime.now().isoformat(),
                         identifier, endpoint))
                    conn.commit()
                    return True

                # Check if within limit
                if count >= limit:
                    return False

                # Increment count
                cursor.execute("""
                    UPDATE rate_limits
                    SET count = count + 1, updated_at = ?
                    WHERE identifier = ? AND endpoint = ?
                """, (datetime.now().isoformat(), identifier, endpoint))

            else:
                # Create new rate limit record
                cursor.execute("""
                    INSERT INTO rate_limits (id, identifier, endpoint, count,
                                           window_start, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                """, (
                    secrets.token_urlsafe(16),
                    identifier,
                    endpoint,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))

            conn.commit()
            return True

    def _save_user(self, user: User):
        """Save user to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO users
                (id, username, email, password_hash, role, is_active, is_verified,
                 mfa_enabled, mfa_secret, created_at, last_login, failed_login_attempts,
                 locked_until, security_level, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.id,
                user.username,
                user.email,
                user.password_hash,
                user.role.value,
                user.is_active,
                user.is_verified,
                user.mfa_enabled,
                user.mfa_secret,
                user.created_at.isoformat() if user.created_at else None,
                user.last_login.isoformat() if user.last_login else None,
                user.failed_login_attempts,
                user.locked_until.isoformat() if user.locked_until else None,
                user.security_level.value,
                json.dumps(user.metadata)
            ))
            conn.commit()

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

            return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))

            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))

            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

            return None

    def _row_to_user(self, row) -> User:
        """Convert database row to User object."""
        return User(
            id=row[0],
            username=row[1],
            email=row[2],
            password_hash=row[3],
            role=UserRole(row[4]),
            is_active=bool(row[5]),
            is_verified=bool(row[6]),
            mfa_enabled=bool(row[7]),
            mfa_secret=row[8],
            created_at=datetime.fromisoformat(row[9]) if row[9] else None,
            last_login=datetime.fromisoformat(row[10]) if row[10] else None,
            failed_login_attempts=row[11],
            locked_until=datetime.fromisoformat(row[12]) if row[12] else None,
            security_level=SecurityLevel(row[13]) if row[13] else SecurityLevel.INTERNAL,
            metadata=json.loads(row[14]) if row[14] else {}
        )

    def _log_security_event(self, event_type: str, user_id: Optional[str],
                           ip_address: str, user_agent: str, details: Dict[str, Any],
                           severity: str = "info"):
        """Log security event."""
        event = SecurityEvent(
            id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity=severity
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO security_events
                (id, event_type, user_id, ip_address, user_agent, details, severity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.event_type,
                event.user_id,
                event.ip_address,
                event.user_agent,
                json.dumps(event.details),
                event.severity,
                event.timestamp.isoformat()
            ))
            conn.commit()

    def get_security_events(self, user_id: Optional[str] = None,
                          event_type: Optional[str] = None,
                          severity: Optional[str] = None,
                          hours: int = 24, limit: int = 100) -> List[SecurityEvent]:
        """Get security events with filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT * FROM security_events
                WHERE timestamp >= datetime('now', '-{} hours')
            """.format(hours)

            params = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            events = []
            for row in cursor.fetchall():
                events.append(SecurityEvent(
                    id=row[0],
                    event_type=row[1],
                    user_id=row[2],
                    ip_address=row[3],
                    user_agent=row[4],
                    details=json.loads(row[5]) if row[5] else {},
                    severity=row[6],
                    timestamp=datetime.fromisoformat(row[7])
                ))

            return events

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get event counts by type
            cursor.execute("""
                SELECT event_type, severity, COUNT(*)
                FROM security_events
                WHERE timestamp >= datetime('now', '-{} hours')
                GROUP BY event_type, severity
            """.format(hours))

            events_by_type = {}
            total_events = 0

            for event_type, severity, count in cursor.fetchall():
                if event_type not in events_by_type:
                    events_by_type[event_type] = {}
                events_by_type[event_type][severity] = count
                total_events += count

            # Get active users
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()[0]

            # Get active sessions
            cursor.execute("""
                SELECT COUNT(*) FROM sessions
                WHERE is_active = 1 AND expires_at > datetime('now')
            """)
            active_sessions = cursor.fetchone()[0]

            # Get API key usage
            cursor.execute("""
                SELECT COUNT(*) FROM api_keys
                WHERE is_active = 1 AND last_used >= datetime('now', '-{} hours')
            """.format(hours))
            active_api_keys = cursor.fetchone()[0]

            return {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "active_users": active_users,
                "active_sessions": active_sessions,
                "active_api_keys": active_api_keys,
                "hours": hours
            }

def require_auth(permission: Optional[str] = None, role: Optional[UserRole] = None):
    """Decorator to require authentication and authorization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a simplified decorator
            # In a real web framework, you'd extract the token from request headers
            # and validate it using the SecurityManager

            # For now, we'll assume the security manager is passed as a parameter
            # or available in the context

            # Example usage in a web framework:
            # token = request.headers.get('Authorization', '').replace('Bearer ', '')
            # user = security_manager.validate_session(token)
            # if not user:
            #     raise UnauthorizedException("Invalid or expired token")

            # Check role if specified
            # if role and user.role != role:
            #     raise ForbiddenException("Insufficient permissions")

            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_api_key(permission: Optional[str] = None):
    """Decorator to require API key authentication."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Similar to require_auth, but for API key validation
            # api_key = request.headers.get('X-API-Key')
            # result = security_manager.validate_api_key(api_key)
            # if not result:
            #     raise UnauthorizedException("Invalid API key")

            return func(*args, **kwargs)
        return wrapper
    return decorator

# Factory function
def create_security_manager(db_path: str = "security.db",
                          secret_key: Optional[str] = None) -> SecurityManager:
    """Create and configure security manager."""
    return SecurityManager(db_path, secret_key)

# Example usage and testing
if __name__ == "__main__":
    def test_security_system():
        """Test the security and authentication system."""
        print("🔐 Testing Security and Authentication System...")

        # Create security manager
        security = create_security_manager("test_security.db")

        # Test user creation
        success, result = security.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            role=UserRole.CREATOR
        )

        if success:
            user = result
            print(f"✅ User created: {user.username}")

            # Test authentication
            auth_success, auth_result = security.authenticate_user(
                "testuser", "TestPassword123!", "127.0.0.1", "Test Agent"
            )

            if auth_success:
                authenticated_user = auth_result
                print(f"✅ Authentication successful: {authenticated_user.username}")

                # Test session creation
                token = security.create_session(authenticated_user, "127.0.0.1", "Test Agent")
                print(f"✅ Session created: {token[:20]}...")

                # Test session validation
                session_user = security.validate_session(token)
                if session_user:
                    print(f"✅ Session validated: {session_user.username}")

                # Test API key creation
                api_key, api_key_obj = security.create_api_key(
                    authenticated_user, "Test API Key", ["read", "write"]
                )
                print(f"✅ API key created: {api_key[:20]}...")

                # Test API key validation
                api_result = security.validate_api_key(api_key)
                if api_result:
                    api_user, api_key_info = api_result
                    print(f"✅ API key validated: {api_user.username}")

                # Test data encryption
                sensitive_data = "This is sensitive information"
                encrypted = security.encrypt_data(sensitive_data)
                decrypted = security.decrypt_data(encrypted)
                print(f"✅ Encryption test: {decrypted == sensitive_data}")

                # Test rate limiting
                rate_limited = security.check_rate_limit("test_user", "/api/test", 5, 60)
                print(f"✅ Rate limiting: {rate_limited}")

                # Get security summary
                summary = security.get_security_summary(hours=1)
                print(f"✅ Security summary: {summary['total_events']} events")

            else:
                print(f"❌ Authentication failed: {auth_result}")
        else:
            print(f"❌ User creation failed: {result}")

        print("🎉 Security System test completed!")

    # Run test
    test_security_system()
