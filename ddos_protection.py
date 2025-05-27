"""
Advanced DDoS Protection and Rate Limiting System

This module provides comprehensive DDoS protection including sliding window rate limiting,
IP-based blocking, behavioral analysis, challenge-response system, and automatic threat detection.
"""

import os
import time
import json
import sqlite3
import logging
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import ipaddress
from threading import Lock
import redis
from functools import wraps

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BlockType(Enum):
    """Types of IP blocks."""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    CHALLENGE = "challenge"

@dataclass
class RateLimit:
    """Rate limit configuration."""
    name: str
    max_requests: int
    window_seconds: int
    burst_allowance: int = 0
    
@dataclass
class ThreatDetection:
    """Threat detection result."""
    ip_address: str
    threat_level: ThreatLevel
    threat_type: str
    confidence: float
    evidence: List[str]
    timestamp: datetime
    
@dataclass
class IPBlock:
    """IP block record."""
    ip_address: str
    block_type: BlockType
    reason: str
    blocked_at: datetime
    expires_at: Optional[datetime] = None
    block_count: int = 1
    
@dataclass
class Challenge:
    """Challenge-response record."""
    challenge_id: str
    ip_address: str
    challenge_type: str  # 'captcha', 'proof_of_work', 'rate_limit'
    challenge_data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3

class DDoSProtection:
    """Advanced DDoS protection and rate limiting system."""
    
    def __init__(self, db_path: str = "ddos_protection.db", redis_url: Optional[str] = None):
        self.db_path = db_path
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        # Rate limiting configuration
        self.rate_limits = {
            'global': RateLimit('global', 1000, 60, 100),  # 1000 req/min with 100 burst
            'per_ip': RateLimit('per_ip', 60, 60, 10),     # 60 req/min per IP with 10 burst
            'auth': RateLimit('auth', 5, 300, 0),          # 5 auth attempts per 5 min
            'api': RateLimit('api', 100, 60, 20),          # 100 API calls/min with 20 burst
        }
        
        # DDoS protection settings
        self.suspicious_request_threshold = 50  # requests in detection window
        self.detection_window = 60  # seconds
        self.auto_block_threshold = 100  # requests to trigger auto-block
        self.challenge_threshold = 30   # requests to trigger challenge
        
        # Block durations
        self.temp_block_duration = timedelta(minutes=15)
        self.escalation_block_duration = timedelta(hours=1)
        self.permanent_block_duration = timedelta(days=30)
        
        # In-memory tracking
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.sliding_windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.active_challenges: Dict[str, Challenge] = {}
        self.blocked_ips: Dict[str, IPBlock] = {}
        
        # Thread safety
        self.lock = Lock()
        
        self._initialize_database()
        self._load_blocked_ips()
        
        # Start background cleanup task
        self._start_cleanup_task()
    
    def _initialize_database(self):
        """Initialize DDoS protection database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Rate limit violations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limit_violations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ip_address TEXT NOT NULL,
                        limit_type TEXT NOT NULL,
                        requests_count INTEGER NOT NULL,
                        window_start TIMESTAMP NOT NULL,
                        violation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_agent TEXT,
                        endpoint TEXT
                    )
                """)
                
                # IP blocks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ip_blocks (
                        ip_address TEXT PRIMARY KEY,
                        block_type TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        block_count INTEGER DEFAULT 1,
                        last_activity TIMESTAMP
                    )
                """)
                
                # Threat detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS threat_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ip_address TEXT NOT NULL,
                        threat_level TEXT NOT NULL,
                        threat_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        evidence TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        action_taken TEXT
                    )
                """)
                
                # Challenge responses table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS challenge_responses (
                        challenge_id TEXT PRIMARY KEY,
                        ip_address TEXT NOT NULL,
                        challenge_type TEXT NOT NULL,
                        challenge_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        attempts INTEGER DEFAULT 0,
                        completed BOOLEAN DEFAULT 0,
                        success BOOLEAN DEFAULT 0
                    )
                """)
                
                # Request patterns table for analysis
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS request_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ip_address TEXT NOT NULL,
                        user_agent TEXT,
                        endpoint TEXT,
                        method TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        response_code INTEGER,
                        response_time REAL,
                        request_size INTEGER,
                        anomaly_score REAL
                    )
                """)
                
                # Indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_ip_time ON rate_limit_violations (ip_address, violation_time)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocks_expires ON ip_blocks (expires_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_threats_ip_time ON threat_detections (ip_address, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_ip_time ON request_patterns (ip_address, timestamp)")
                
                conn.commit()
                logger.info("DDoS protection database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize DDoS protection database: {e}")
            raise
    
    def check_rate_limit(self, ip_address: str, limit_type: str = 'per_ip', 
                        endpoint: str = None, user_agent: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        try:
            with self.lock:
                # Check if IP is blocked
                if self._is_ip_blocked(ip_address):
                    return False, {'blocked': True, 'reason': 'IP blocked'}
                
                # Get rate limit configuration
                rate_limit = self.rate_limits.get(limit_type)
                if not rate_limit:
                    return True, {}
                
                current_time = time.time()
                window_key = f"{ip_address}:{limit_type}"
                
                # Use Redis for distributed rate limiting if available
                if self.redis_client:
                    return self._check_redis_rate_limit(ip_address, rate_limit, current_time)
                
                # Use in-memory sliding window
                window = self.sliding_windows[window_key][rate_limit.name]
                
                # Remove old requests outside the window
                cutoff_time = current_time - rate_limit.window_seconds
                while window and window[0] <= cutoff_time:
                    window.popleft()
                
                # Check if within limit
                current_requests = len(window)
                
                # Apply burst allowance
                effective_limit = rate_limit.max_requests + rate_limit.burst_allowance
                
                if current_requests >= effective_limit:
                    # Record violation
                    self._record_rate_limit_violation(ip_address, limit_type, current_requests, 
                                                    datetime.fromtimestamp(current_time - rate_limit.window_seconds),
                                                    user_agent, endpoint)
                    
                    # Check for threat patterns
                    threat = self._analyze_threat_patterns(ip_address, current_requests)
                    if threat:
                        self._handle_threat_detection(threat)
                    
                    return False, {
                        'rate_limited': True,
                        'limit_type': limit_type,
                        'current_requests': current_requests,
                        'max_requests': effective_limit,
                        'window_seconds': rate_limit.window_seconds,
                        'retry_after': rate_limit.window_seconds
                    }
                
                # Record successful request
                window.append(current_time)
                self._record_request_pattern(ip_address, user_agent, endpoint, current_time)
                
                return True, {
                    'current_requests': current_requests + 1,
                    'max_requests': effective_limit,
                    'window_seconds': rate_limit.window_seconds,
                    'remaining_requests': effective_limit - current_requests - 1
                }
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, {}  # Fail open for availability
    
    def _check_redis_rate_limit(self, ip_address: str, rate_limit: RateLimit, current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis sliding window."""
        try:
            key = f"rate_limit:{ip_address}:{rate_limit.name}"
            window_start = current_time - rate_limit.window_seconds
            
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, rate_limit.window_seconds)
            
            results = pipe.execute()
            current_requests = results[1]
            
            effective_limit = rate_limit.max_requests + rate_limit.burst_allowance
            
            if current_requests >= effective_limit:
                # Remove the request we just added since it's rejected
                self.redis_client.zrem(key, str(current_time))
                
                return False, {
                    'rate_limited': True,
                    'current_requests': current_requests,
                    'max_requests': effective_limit,
                    'window_seconds': rate_limit.window_seconds,
                    'retry_after': rate_limit.window_seconds
                }
            
            return True, {
                'current_requests': current_requests + 1,
                'max_requests': effective_limit,
                'window_seconds': rate_limit.window_seconds,
                'remaining_requests': effective_limit - current_requests - 1
            }
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True, {}
    
    def block_ip(self, ip_address: str, block_type: BlockType, reason: str, 
                duration: Optional[timedelta] = None) -> bool:
        """Block an IP address."""
        try:
            with self.lock:
                expires_at = None
                if duration:
                    expires_at = datetime.now() + duration
                elif block_type == BlockType.TEMPORARY:
                    expires_at = datetime.now() + self.temp_block_duration
                
                # Check if IP is already blocked and escalate
                existing_block = self.blocked_ips.get(ip_address)
                block_count = 1
                if existing_block:
                    block_count = existing_block.block_count + 1
                    # Escalate block duration
                    if block_count >= 3:
                        block_type = BlockType.PERMANENT
                        expires_at = datetime.now() + self.permanent_block_duration
                    elif block_count >= 2:
                        expires_at = datetime.now() + self.escalation_block_duration
                
                block = IPBlock(
                    ip_address=ip_address,
                    block_type=block_type,
                    reason=reason,
                    blocked_at=datetime.now(),
                    expires_at=expires_at,
                    block_count=block_count
                )
                
                self.blocked_ips[ip_address] = block
                
                # Save to database
                self._save_ip_block(block)
                
                logger.warning(f"IP {ip_address} blocked: {reason} (type: {block_type.value}, count: {block_count})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to block IP: {e}")
            return False
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address."""
        try:
            with self.lock:
                if ip_address in self.blocked_ips:
                    del self.blocked_ips[ip_address]
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM ip_blocks WHERE ip_address = ?", (ip_address,))
                    
                    success = cursor.rowcount > 0
                    if success:
                        logger.info(f"IP {ip_address} unblocked")
                    
                    return success
                    
        except Exception as e:
            logger.error(f"Failed to unblock IP: {e}")
            return False
    
    def create_challenge(self, ip_address: str, challenge_type: str = 'proof_of_work') -> Optional[str]:
        """Create a challenge for suspicious IP."""
        try:
            challenge_id = secrets.token_urlsafe(16)
            
            challenge_data = {}
            if challenge_type == 'proof_of_work':
                challenge_data = self._create_proof_of_work_challenge()
            elif challenge_type == 'rate_limit':
                challenge_data = {'delay': 5}  # 5 second delay
            
            challenge = Challenge(
                challenge_id=challenge_id,
                ip_address=ip_address,
                challenge_type=challenge_type,
                challenge_data=challenge_data,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=10)
            )
            
            self.active_challenges[challenge_id] = challenge
            self._save_challenge(challenge)
            
            logger.info(f"Challenge created for IP {ip_address}: {challenge_type}")
            return challenge_id
            
        except Exception as e:
            logger.error(f"Failed to create challenge: {e}")
            return None
    
    def verify_challenge(self, challenge_id: str, response: str) -> bool:
        """Verify challenge response."""
        try:
            challenge = self.active_challenges.get(challenge_id)
            if not challenge or datetime.now() > challenge.expires_at:
                return False
            
            challenge.attempts += 1
            
            if challenge.attempts > challenge.max_attempts:
                # Too many attempts, block IP
                self.block_ip(challenge.ip_address, BlockType.TEMPORARY, 
                             "Too many challenge attempts")
                del self.active_challenges[challenge_id]
                return False
            
            is_valid = False
            if challenge.challenge_type == 'proof_of_work':
                is_valid = self._verify_proof_of_work(challenge.challenge_data, response)
            elif challenge.challenge_type == 'rate_limit':
                # For rate limit challenges, any response is valid after delay
                is_valid = True
            
            if is_valid:
                # Mark as completed
                challenge.attempts = -1  # Mark as successful
                self._update_challenge_status(challenge_id, True, True)
                logger.info(f"Challenge {challenge_id} completed successfully")
            else:
                self._update_challenge_status(challenge_id, False, False)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Challenge verification failed: {e}")
            return False
    
    def get_threat_analysis(self, ip_address: str) -> Dict[str, Any]:
        """Get threat analysis for an IP address."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent violations
                cursor.execute("""
                    SELECT COUNT(*), MAX(violation_time)
                    FROM rate_limit_violations 
                    WHERE ip_address = ? AND violation_time > ?
                """, (ip_address, datetime.now() - timedelta(hours=24)))
                
                violation_data = cursor.fetchone()
                violations_24h = violation_data[0] if violation_data else 0
                last_violation = violation_data[1] if violation_data and violation_data[1] else None
                
                # Get threat detections
                cursor.execute("""
                    SELECT threat_level, threat_type, confidence, timestamp
                    FROM threat_detections 
                    WHERE ip_address = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """, (ip_address,))
                
                threats = []
                for row in cursor.fetchall():
                    threats.append({
                        'level': row[0],
                        'type': row[1],
                        'confidence': row[2],
                        'timestamp': row[3]
                    })
                
                # Get block history
                cursor.execute("""
                    SELECT block_type, reason, blocked_at, expires_at
                    FROM ip_blocks 
                    WHERE ip_address = ?
                """, (ip_address,))
                
                block_info = cursor.fetchone()
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(violations_24h, threats, block_info)
                
                return {
                    'ip_address': ip_address,
                    'risk_score': risk_score,
                    'violations_24h': violations_24h,
                    'last_violation': last_violation,
                    'threats': threats,
                    'is_blocked': ip_address in self.blocked_ips,
                    'block_info': {
                        'type': block_info[0] if block_info else None,
                        'reason': block_info[1] if block_info else None,
                        'blocked_at': block_info[2] if block_info else None,
                        'expires_at': block_info[3] if block_info else None
                    } if block_info else None
                }
                
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return {'ip_address': ip_address, 'risk_score': 0.0}
    
    def get_protection_stats(self) -> Dict[str, Any]:
        """Get DDoS protection statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get stats for last 24 hours
                since = datetime.now() - timedelta(hours=24)
                
                # Rate limit violations
                cursor.execute("""
                    SELECT COUNT(*) FROM rate_limit_violations 
                    WHERE violation_time > ?
                """, (since,))
                violations_24h = cursor.fetchone()[0]
                
                # Threat detections
                cursor.execute("""
                    SELECT threat_level, COUNT(*) FROM threat_detections 
                    WHERE timestamp > ? 
                    GROUP BY threat_level
                """, (since,))
                threat_counts = dict(cursor.fetchall())
                
                # Active blocks
                cursor.execute("""
                    SELECT block_type, COUNT(*) FROM ip_blocks 
                    WHERE (expires_at IS NULL OR expires_at > ?) 
                    GROUP BY block_type
                """, (datetime.now(),))
                block_counts = dict(cursor.fetchall())
                
                # Challenge statistics
                cursor.execute("""
                    SELECT challenge_type, COUNT(*), 
                           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
                    FROM challenge_responses 
                    WHERE created_at > ? 
                    GROUP BY challenge_type
                """, (since,))
                
                challenge_stats = {}
                for row in cursor.fetchall():
                    challenge_stats[row[0]] = {
                        'total': row[1],
                        'successful': row[2],
                        'success_rate': row[2] / row[1] if row[1] > 0 else 0
                    }
                
                return {
                    'violations_24h': violations_24h,
                    'threat_counts': threat_counts,
                    'active_blocks': block_counts,
                    'challenge_stats': challenge_stats,
                    'total_blocked_ips': len(self.blocked_ips),
                    'active_challenges': len(self.active_challenges)
                }
                
        except Exception as e:
            logger.error(f"Failed to get protection stats: {e}")
            return {}
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is currently blocked."""
        block = self.blocked_ips.get(ip_address)
        if not block:
            return False
        
        # Check if temporary block has expired
        if block.expires_at and datetime.now() > block.expires_at:
            del self.blocked_ips[ip_address]
            self._remove_expired_block(ip_address)
            return False
        
        return True
    
    def _analyze_threat_patterns(self, ip_address: str, current_requests: int) -> Optional[ThreatDetection]:
        """Analyze request patterns for threats."""
        try:
            # Simple heuristic-based threat detection
            threat_level = ThreatLevel.LOW
            confidence = 0.0
            evidence = []
            threat_type = "rate_limit_violation"
            
            if current_requests > self.auto_block_threshold:
                threat_level = ThreatLevel.CRITICAL
                confidence = 0.95
                evidence.append(f"Excessive requests: {current_requests}")
                threat_type = "ddos_attack"
            elif current_requests > self.suspicious_request_threshold:
                threat_level = ThreatLevel.HIGH
                confidence = 0.8
                evidence.append(f"Suspicious request volume: {current_requests}")
                threat_type = "potential_attack"
            elif current_requests > self.challenge_threshold:
                threat_level = ThreatLevel.MEDIUM
                confidence = 0.6
                evidence.append(f"Elevated request rate: {current_requests}")
                threat_type = "rate_abuse"
            
            if threat_level != ThreatLevel.LOW:
                return ThreatDetection(
                    ip_address=ip_address,
                    threat_level=threat_level,
                    threat_type=threat_type,
                    confidence=confidence,
                    evidence=evidence,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return None
    
    def _handle_threat_detection(self, threat: ThreatDetection):
        """Handle detected threat."""
        try:
            # Record threat detection
            self._save_threat_detection(threat)
            
            # Take action based on threat level
            action_taken = "none"
            
            if threat.threat_level == ThreatLevel.CRITICAL:
                # Immediately block IP
                self.block_ip(threat.ip_address, BlockType.TEMPORARY, 
                             f"Critical threat: {threat.threat_type}")
                action_taken = "immediate_block"
            elif threat.threat_level == ThreatLevel.HIGH:
                # Block IP with escalation
                duration = self.escalation_block_duration
                if threat.confidence > 0.9:
                    duration = self.permanent_block_duration
                
                self.block_ip(threat.ip_address, BlockType.TEMPORARY, 
                             f"High threat: {threat.threat_type}", duration)
                action_taken = "escalated_block"
            elif threat.threat_level == ThreatLevel.MEDIUM:
                # Issue challenge
                challenge_id = self.create_challenge(threat.ip_address, 'proof_of_work')
                action_taken = f"challenge_issued:{challenge_id}" if challenge_id else "challenge_failed"
            
            logger.warning(f"Threat handled: {threat.ip_address} - {threat.threat_type} "
                          f"(level: {threat.threat_level.value}, action: {action_taken})")
            
        except Exception as e:
            logger.error(f"Failed to handle threat: {e}")
    
    def _create_proof_of_work_challenge(self) -> Dict[str, Any]:
        """Create proof-of-work challenge."""
        # Simple proof-of-work: find nonce that makes hash start with zeros
        target = "0000"  # 4 leading zeros
        salt = secrets.token_hex(16)
        
        return {
            'target': target,
            'salt': salt,
            'algorithm': 'sha256'
        }
    
    def _verify_proof_of_work(self, challenge_data: Dict[str, Any], response: str) -> bool:
        """Verify proof-of-work response."""
        try:
            target = challenge_data['target']
            salt = challenge_data['salt']
            
            # Verify that hash(salt + response) starts with target
            hash_input = salt + response
            hash_result = hashlib.sha256(hash_input.encode()).hexdigest()
            
            return hash_result.startswith(target)
            
        except Exception as e:
            logger.error(f"Proof-of-work verification failed: {e}")
            return False
    
    def _calculate_risk_score(self, violations: int, threats: List[Dict], block_info: Any) -> float:
        """Calculate risk score for an IP address."""
        score = 0.0
        
        # Violations contribution
        if violations > 0:
            score += min(violations * 0.1, 0.5)  # Max 0.5 from violations
        
        # Threats contribution
        for threat in threats[-5:]:  # Last 5 threats
            if threat['level'] == 'critical':
                score += 0.3
            elif threat['level'] == 'high':
                score += 0.2
            elif threat['level'] == 'medium':
                score += 0.1
        
        # Block history contribution
        if block_info:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _record_rate_limit_violation(self, ip_address: str, limit_type: str, requests_count: int,
                                   window_start: datetime, user_agent: str = None, endpoint: str = None):
        """Record rate limit violation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO rate_limit_violations 
                    (ip_address, limit_type, requests_count, window_start, user_agent, endpoint)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ip_address, limit_type, requests_count, window_start, user_agent, endpoint))
                
        except Exception as e:
            logger.error(f"Failed to record rate limit violation: {e}")
    
    def _record_request_pattern(self, ip_address: str, user_agent: str, endpoint: str, timestamp: float):
        """Record request pattern for analysis."""
        try:
            # Sample requests for pattern analysis (store 1 in 100)
            if int(timestamp * 1000) % 100 == 0:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO request_patterns 
                        (ip_address, user_agent, endpoint, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (ip_address, user_agent, endpoint, datetime.fromtimestamp(timestamp)))
                    
        except Exception as e:
            logger.error(f"Failed to record request pattern: {e}")
    
    def _save_ip_block(self, block: IPBlock):
        """Save IP block to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ip_blocks 
                (ip_address, block_type, reason, blocked_at, expires_at, block_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                block.ip_address, block.block_type.value, block.reason,
                block.blocked_at, block.expires_at, block.block_count
            ))
    
    def _save_threat_detection(self, threat: ThreatDetection):
        """Save threat detection to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO threat_detections 
                (ip_address, threat_level, threat_type, confidence, evidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                threat.ip_address, threat.threat_level.value, threat.threat_type,
                threat.confidence, json.dumps(threat.evidence), threat.timestamp
            ))
    
    def _save_challenge(self, challenge: Challenge):
        """Save challenge to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO challenge_responses 
                (challenge_id, ip_address, challenge_type, challenge_data, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                challenge.challenge_id, challenge.ip_address, challenge.challenge_type,
                json.dumps(challenge.challenge_data), challenge.created_at, challenge.expires_at
            ))
    
    def _update_challenge_status(self, challenge_id: str, completed: bool, success: bool):
        """Update challenge completion status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE challenge_responses 
                SET completed = ?, success = ?, attempts = (
                    SELECT attempts FROM challenge_responses WHERE challenge_id = ?
                ) + 1
                WHERE challenge_id = ?
            """, (completed, success, challenge_id, challenge_id))
    
    def _load_blocked_ips(self):
        """Load blocked IPs from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ip_address, block_type, reason, blocked_at, expires_at, block_count
                    FROM ip_blocks 
                    WHERE expires_at IS NULL OR expires_at > ?
                """, (datetime.now(),))
                
                for row in cursor.fetchall():
                    block = IPBlock(
                        ip_address=row[0],
                        block_type=BlockType(row[1]),
                        reason=row[2],
                        blocked_at=datetime.fromisoformat(row[3]),
                        expires_at=datetime.fromisoformat(row[4]) if row[4] else None,
                        block_count=row[5]
                    )
                    self.blocked_ips[row[0]] = block
                
                logger.info(f"Loaded {len(self.blocked_ips)} blocked IPs from database")
                
        except Exception as e:
            logger.error(f"Failed to load blocked IPs: {e}")
    
    def _remove_expired_block(self, ip_address: str):
        """Remove expired block from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM ip_blocks 
                    WHERE ip_address = ? AND expires_at <= ?
                """, (ip_address, datetime.now()))
                
        except Exception as e:
            logger.error(f"Failed to remove expired block: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        def cleanup_worker():
            while True:
                try:
                    # Clean up expired blocks
                    expired_ips = []
                    with self.lock:
                        for ip, block in list(self.blocked_ips.items()):
                            if block.expires_at and datetime.now() > block.expires_at:
                                expired_ips.append(ip)
                        
                        for ip in expired_ips:
                            del self.blocked_ips[ip]
                            self._remove_expired_block(ip)
                    
                    # Clean up expired challenges
                    expired_challenges = []
                    for challenge_id, challenge in list(self.active_challenges.items()):
                        if datetime.now() > challenge.expires_at:
                            expired_challenges.append(challenge_id)
                    
                    for challenge_id in expired_challenges:
                        del self.active_challenges[challenge_id]
                    
                    if expired_ips or expired_challenges:
                        logger.info(f"Cleaned up {len(expired_ips)} expired blocks and {len(expired_challenges)} expired challenges")
                    
                    time.sleep(60)  # Run every minute
                    
                except Exception as e:
                    logger.error(f"Cleanup task failed: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()


def ddos_protection_middleware(ddos_protection: DDoSProtection):
    """Decorator for DDoS protection middleware."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request information (implementation depends on framework)
            ip_address = kwargs.get('ip_address', '127.0.0.1')
            endpoint = kwargs.get('endpoint', 'unknown')
            user_agent = kwargs.get('user_agent', '')
            
            # Check rate limits
            allowed, limit_info = ddos_protection.check_rate_limit(
                ip_address=ip_address,
                limit_type='per_ip',
                endpoint=endpoint,
                user_agent=user_agent
            )
            
            if not allowed:
                if limit_info.get('blocked'):
                    return {'error': 'IP blocked', 'code': 403}
                else:
                    return {
                        'error': 'Rate limit exceeded',
                        'code': 429,
                        'retry_after': limit_info.get('retry_after', 60)
                    }
            
            # Proceed with original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
