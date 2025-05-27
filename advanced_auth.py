"""
Advanced Authentication System with Multi-Factor Authentication (MFA)

This module provides comprehensive MFA capabilities including TOTP, SMS/Email verification,
backup codes, and QR code generation for authenticator apps.
"""

import os
import qrcode
import pyotp
import secrets
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart
from io import BytesIO
import base64
import sqlite3
import json
import hashlib
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class MFADevice:
    """Multi-factor authentication device."""
    id: str
    user_id: str
    device_type: str  # 'totp', 'sms', 'email'
    device_name: str
    secret_key: Optional[str] = None
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    backup_codes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class MFAChallenge:
    """MFA challenge for verification."""
    id: str
    user_id: str
    device_id: str
    challenge_type: str
    challenge_code: str
    expires_at: datetime
    is_used: bool = False
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class MFAManager:
    """Manages multi-factor authentication for enhanced security."""
    
    def __init__(self, db_path: str = "mfa.db", encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # MFA settings
        self.totp_issuer = "LangGraph101"
        self.challenge_expiry = timedelta(minutes=5)
        self.backup_codes_count = 10
        self.max_failed_attempts = 3
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize MFA database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # MFA devices table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mfa_devices (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        device_type TEXT NOT NULL,
                        device_name TEXT NOT NULL,
                        secret_key TEXT,
                        phone_number TEXT,
                        email_address TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        backup_codes TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # MFA challenges table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mfa_challenges (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        device_id TEXT NOT NULL,
                        challenge_type TEXT NOT NULL,
                        challenge_code TEXT NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        is_used BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id),
                        FOREIGN KEY (device_id) REFERENCES mfa_devices (id)
                    )
                """)
                
                # MFA attempts table for rate limiting
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mfa_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        device_id TEXT,
                        attempt_type TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.commit()
                logger.info("MFA database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize MFA database: {e}")
            raise
    
    def setup_totp_device(self, user_id: str, device_name: str, user_email: str) -> Tuple[str, str, str]:
        """
        Set up TOTP device for user.
        
        Returns:
            Tuple of (device_id, secret_key, qr_code_data_url)
        """
        try:
            # Generate device ID and secret
            device_id = secrets.token_urlsafe(16)
            secret_key = pyotp.random_base32()
            
            # Create TOTP URI for QR code
            totp_uri = pyotp.totp.TOTP(secret_key).provisioning_uri(
                name=user_email,
                issuer_name=self.totp_issuer
            )
            
            # Generate QR code
            qr_code_data_url = self._generate_qr_code(totp_uri)
            
            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            
            # Create MFA device
            device = MFADevice(
                id=device_id,
                user_id=user_id,
                device_type="totp",
                device_name=device_name,
                secret_key=secret_key,
                backup_codes=backup_codes
            )
            
            # Save to database
            self._save_mfa_device(device)
            
            logger.info(f"TOTP device set up for user {user_id}")
            return device_id, secret_key, qr_code_data_url
            
        except Exception as e:
            logger.error(f"Failed to setup TOTP device: {e}")
            raise
    
    def setup_sms_device(self, user_id: str, device_name: str, phone_number: str) -> str:
        """Set up SMS-based MFA device."""
        try:
            device_id = secrets.token_urlsafe(16)
            
            device = MFADevice(
                id=device_id,
                user_id=user_id,
                device_type="sms",
                device_name=device_name,
                phone_number=phone_number,
                backup_codes=self._generate_backup_codes()
            )
            
            self._save_mfa_device(device)
            logger.info(f"SMS device set up for user {user_id}")
            return device_id
            
        except Exception as e:
            logger.error(f"Failed to setup SMS device: {e}")
            raise
    
    def setup_email_device(self, user_id: str, device_name: str, email_address: str) -> str:
        """Set up email-based MFA device."""
        try:
            device_id = secrets.token_urlsafe(16)
            
            device = MFADevice(
                id=device_id,
                user_id=user_id,
                device_type="email",
                device_name=device_name,
                email_address=email_address,
                backup_codes=self._generate_backup_codes()
            )
            
            self._save_mfa_device(device)
            logger.info(f"Email device set up for user {user_id}")
            return device_id
            
        except Exception as e:
            logger.error(f"Failed to setup email device: {e}")
            raise
    
    def verify_totp_code(self, user_id: str, device_id: str, code: str) -> bool:
        """Verify TOTP code from authenticator app."""
        try:
            device = self._get_mfa_device(device_id)
            if not device or device.user_id != user_id or device.device_type != "totp":
                return False
            
            # Check if it's a backup code
            if self._verify_backup_code(device, code):
                self._record_mfa_attempt(user_id, device_id, "backup_code", True)
                return True
            
            # Verify TOTP code
            totp = pyotp.TOTP(device.secret_key)
            is_valid = totp.verify(code, valid_window=1)  # Allow 30s window
            
            # Record attempt
            self._record_mfa_attempt(user_id, device_id, "totp", is_valid)
            
            if is_valid:
                self._update_device_last_used(device_id)
                logger.info(f"TOTP verification successful for user {user_id}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    def send_sms_challenge(self, user_id: str, device_id: str) -> Optional[str]:
        """Send SMS challenge code."""
        try:
            device = self._get_mfa_device(device_id)
            if not device or device.user_id != user_id or device.device_type != "sms":
                return None
            
            # Generate challenge
            challenge_id = secrets.token_urlsafe(16)
            challenge_code = str(secrets.randbelow(900000) + 100000)  # 6-digit code
            
            challenge = MFAChallenge(
                id=challenge_id,
                user_id=user_id,
                device_id=device_id,
                challenge_type="sms",
                challenge_code=challenge_code,
                expires_at=datetime.now() + self.challenge_expiry
            )
            
            # Save challenge
            self._save_mfa_challenge(challenge)
            
            # Send SMS (mock implementation)
            self._send_sms_message(device.phone_number, challenge_code)
            
            logger.info(f"SMS challenge sent to user {user_id}")
            return challenge_id
            
        except Exception as e:
            logger.error(f"Failed to send SMS challenge: {e}")
            return None
    
    def send_email_challenge(self, user_id: str, device_id: str) -> Optional[str]:
        """Send email challenge code."""
        try:
            device = self._get_mfa_device(device_id)
            if not device or device.user_id != user_id or device.device_type != "email":
                return None
            
            # Generate challenge
            challenge_id = secrets.token_urlsafe(16)
            challenge_code = str(secrets.randbelow(900000) + 100000)  # 6-digit code
            
            challenge = MFAChallenge(
                id=challenge_id,
                user_id=user_id,
                device_id=device_id,
                challenge_type="email",
                challenge_code=challenge_code,
                expires_at=datetime.now() + self.challenge_expiry
            )
            
            # Save challenge
            self._save_mfa_challenge(challenge)
            
            # Send email
            self._send_email_message(device.email_address, challenge_code)
            
            logger.info(f"Email challenge sent to user {user_id}")
            return challenge_id
            
        except Exception as e:
            logger.error(f"Failed to send email challenge: {e}")
            return None
    
    def verify_challenge_code(self, challenge_id: str, code: str) -> bool:
        """Verify challenge code for SMS/email."""
        try:
            challenge = self._get_mfa_challenge(challenge_id)
            if not challenge or challenge.is_used or challenge.expires_at < datetime.now():
                return False
            
            is_valid = challenge.challenge_code == code
            
            if is_valid:
                # Mark challenge as used
                self._mark_challenge_used(challenge_id)
                self._update_device_last_used(challenge.device_id)
                self._record_mfa_attempt(challenge.user_id, challenge.device_id, challenge.challenge_type, True)
                logger.info(f"Challenge verification successful for challenge {challenge_id}")
            else:
                self._record_mfa_attempt(challenge.user_id, challenge.device_id, challenge.challenge_type, False)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Challenge verification failed: {e}")
            return False
    
    def get_user_devices(self, user_id: str) -> List[Dict]:
        """Get all MFA devices for a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, device_type, device_name, is_active, created_at, last_used
                    FROM mfa_devices 
                    WHERE user_id = ? AND is_active = 1
                    ORDER BY created_at DESC
                """, (user_id,))
                
                devices = []
                for row in cursor.fetchall():
                    devices.append({
                        'id': row[0],
                        'device_type': row[1],
                        'device_name': row[2],
                        'is_active': bool(row[3]),
                        'created_at': row[4],
                        'last_used': row[5]
                    })
                
                return devices
                
        except Exception as e:
            logger.error(f"Failed to get user devices: {e}")
            return []
    
    def disable_device(self, user_id: str, device_id: str) -> bool:
        """Disable an MFA device."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE mfa_devices 
                    SET is_active = 0 
                    WHERE id = ? AND user_id = ?
                """, (device_id, user_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"MFA device {device_id} disabled for user {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to disable device: {e}")
            return False
    
    def regenerate_backup_codes(self, user_id: str, device_id: str) -> Optional[List[str]]:
        """Regenerate backup codes for a device."""
        try:
            device = self._get_mfa_device(device_id)
            if not device or device.user_id != user_id:
                return None
            
            new_backup_codes = self._generate_backup_codes()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE mfa_devices 
                    SET backup_codes = ? 
                    WHERE id = ?
                """, (json.dumps(new_backup_codes), device_id))
            
            logger.info(f"Backup codes regenerated for device {device_id}")
            return new_backup_codes
            
        except Exception as e:
            logger.error(f"Failed to regenerate backup codes: {e}")
            return None
    
    def _generate_qr_code(self, totp_uri: str) -> str:
        """Generate QR code data URL for TOTP setup."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to data URL
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_data}"
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA device."""
        codes = []
        for _ in range(self.backup_codes_count):
            code = secrets.token_hex(4).upper()  # 8-character hex code
            codes.append(code)
        return codes
    
    def _verify_backup_code(self, device: MFADevice, code: str) -> bool:
        """Verify and consume a backup code."""
        if not device.backup_codes:
            return False
        
        code = code.upper().strip()
        if code in device.backup_codes:
            # Remove used backup code
            device.backup_codes.remove(code)
            
            # Update in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE mfa_devices 
                    SET backup_codes = ? 
                    WHERE id = ?
                """, (json.dumps(device.backup_codes), device.id))
            
            return True
        
        return False
    
    def _save_mfa_device(self, device: MFADevice):
        """Save MFA device to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Encrypt secret key if present
            secret_key = device.secret_key
            if secret_key:
                secret_key = self.cipher_suite.encrypt(secret_key.encode()).decode()
            
            cursor.execute("""
                INSERT INTO mfa_devices 
                (id, user_id, device_type, device_name, secret_key, phone_number, 
                 email_address, backup_codes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                device.id, device.user_id, device.device_type, device.device_name,
                secret_key, device.phone_number, device.email_address,
                json.dumps(device.backup_codes) if device.backup_codes else None,
                device.created_at
            ))
    
    def _get_mfa_device(self, device_id: str) -> Optional[MFADevice]:
        """Get MFA device by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM mfa_devices WHERE id = ? AND is_active = 1
                """, (device_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Decrypt secret key if present
                secret_key = row[4]
                if secret_key:
                    secret_key = self.cipher_suite.decrypt(secret_key.encode()).decode()
                
                backup_codes = json.loads(row[10]) if row[10] else None
                
                return MFADevice(
                    id=row[0],
                    user_id=row[1],
                    device_type=row[2],
                    device_name=row[3],
                    secret_key=secret_key,
                    phone_number=row[5],
                    email_address=row[6],
                    is_active=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    last_used=datetime.fromisoformat(row[9]) if row[9] else None,
                    backup_codes=backup_codes
                )
                
        except Exception as e:
            logger.error(f"Failed to get MFA device: {e}")
            return None
    
    def _save_mfa_challenge(self, challenge: MFAChallenge):
        """Save MFA challenge to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO mfa_challenges 
                (id, user_id, device_id, challenge_type, challenge_code, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                challenge.id, challenge.user_id, challenge.device_id,
                challenge.challenge_type, challenge.challenge_code,
                challenge.expires_at, challenge.created_at
            ))
    
    def _get_mfa_challenge(self, challenge_id: str) -> Optional[MFAChallenge]:
        """Get MFA challenge by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM mfa_challenges WHERE id = ?
                """, (challenge_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return MFAChallenge(
                    id=row[0],
                    user_id=row[1],
                    device_id=row[2],
                    challenge_type=row[3],
                    challenge_code=row[4],
                    expires_at=datetime.fromisoformat(row[5]),
                    is_used=bool(row[6]),
                    created_at=datetime.fromisoformat(row[7]) if row[7] else None
                )
                
        except Exception as e:
            logger.error(f"Failed to get MFA challenge: {e}")
            return None
    
    def _mark_challenge_used(self, challenge_id: str):
        """Mark challenge as used."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE mfa_challenges SET is_used = 1 WHERE id = ?
            """, (challenge_id,))
    
    def _update_device_last_used(self, device_id: str):
        """Update device last used timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE mfa_devices SET last_used = ? WHERE id = ?
            """, (datetime.now(), device_id))
    
    def _record_mfa_attempt(self, user_id: str, device_id: str, attempt_type: str, success: bool, ip_address: str = None, user_agent: str = None):
        """Record MFA attempt for monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO mfa_attempts 
                (user_id, device_id, attempt_type, success, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, device_id, attempt_type, success, ip_address, user_agent))
    
    def _send_sms_message(self, phone_number: str, code: str):
        """Send SMS message (mock implementation)."""
        # In a real implementation, integrate with SMS service like Twilio
        message = f"Your LangGraph101 verification code is: {code}. Valid for 5 minutes."
        logger.info(f"SMS sent to {phone_number}: {message}")
    
    def _send_email_message(self, email_address: str, code: str):
        """Send email verification code."""
        try:
            if not all([self.smtp_username, self.smtp_password]):
                logger.warning("SMTP credentials not configured, skipping email")
                return
            
            msg = MimeMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = email_address
            msg['Subject'] = "LangGraph101 Verification Code"
            
            body = f"""
            Your LangGraph101 verification code is: {code}
            
            This code is valid for 5 minutes.
            
            If you didn't request this code, please ignore this email.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email verification sent to {email_address}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
