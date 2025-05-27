"""
Production Features Module

This module provides rate limiting, input validation, and other production-ready features
for the LangGraph 101 AI Agent Platform.
"""

import time
import re
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps
import logging
import hashlib

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting implementation with per-user and global limits."""
    
    def __init__(self):
        self.user_requests: Dict[str, List[float]] = {}
        self.global_requests: List[float] = []
        self.max_requests_per_user = 100  # requests per hour
        self.max_global_requests = 1000   # requests per hour
        self.window_size = 3600  # 1 hour in seconds
        
    def is_rate_limited(self, user_id: str) -> bool:
        """Check if user or global rate limit is exceeded."""
        current_time = time.time()
        
        # Clean old requests (older than window_size)
        self._clean_old_requests(current_time)
        
        # Check user-specific rate limit
        user_request_count = len(self.user_requests.get(user_id, []))
        if user_request_count >= self.max_requests_per_user:
            logger.warning(f"User {user_id} exceeded rate limit: {user_request_count} requests")
            return True
            
        # Check global rate limit
        global_request_count = len(self.global_requests)
        if global_request_count >= self.max_global_requests:
            logger.warning(f"Global rate limit exceeded: {global_request_count} requests")
            return True
            
        return False
    
    def record_request(self, user_id: str):
        """Record a new request for the user."""
        current_time = time.time()
        
        # Record user request
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(current_time)
        
        # Record global request
        self.global_requests.append(current_time)
        
        # Clean old requests
        self._clean_old_requests(current_time)
    
    def _clean_old_requests(self, current_time: float):
        """Remove requests older than the window size."""
        cutoff_time = current_time - self.window_size
        
        # Clean user requests
        for user_id in list(self.user_requests.keys()):
            self.user_requests[user_id] = [
                req_time for req_time in self.user_requests[user_id] 
                if req_time > cutoff_time
            ]
            # Remove empty user entries
            if not self.user_requests[user_id]:
                del self.user_requests[user_id]
        
        # Clean global requests
        self.global_requests = [
            req_time for req_time in self.global_requests 
            if req_time > cutoff_time
        ]
    
    def get_remaining_requests(self, user_id: str) -> Dict[str, int]:
        """Get remaining requests for user and global limits."""
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        user_used = len(self.user_requests.get(user_id, []))
        global_used = len(self.global_requests)
        
        return {
            'user_remaining': max(0, self.max_requests_per_user - user_used),
            'global_remaining': max(0, self.max_global_requests - global_used),
            'user_limit': self.max_requests_per_user,
            'global_limit': self.max_global_requests
        }


class InputValidator:
    """Input validation and sanitization for user inputs."""
    
    # Dangerous patterns to block
    BLOCKED_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',              # JavaScript URLs
        r'on\w+\s*=',               # Event handlers
        r'eval\s*\(',               # eval() calls
        r'exec\s*\(',               # exec() calls
        r'import\s+os',             # OS imports
        r'__import__',              # Dynamic imports
        r'subprocess',              # Subprocess calls
        r'system\s*\(',             # System calls
    ]
    
    # Maximum input lengths
    MAX_LENGTHS = {
        'message': 4000,
        'persona_name': 100,
        'username': 50,
        'email': 254,
        'filename': 255,
        'general': 1000
    }
    
    @classmethod
    def validate_input(cls, input_text: str, input_type: str = 'general') -> Dict[str, Any]:
        """
        Validate and sanitize user input.
        
        Args:
            input_text: The text to validate
            input_type: Type of input for length validation
            
        Returns:
            Dict with validation results and sanitized text
        """
        result = {
            'is_valid': True,
            'sanitized_text': input_text,
            'warnings': [],
            'errors': []
        }
        
        if not input_text:
            result['errors'].append("Input cannot be empty")
            result['is_valid'] = False
            return result
        
        # Check length limits
        max_length = cls.MAX_LENGTHS.get(input_type, cls.MAX_LENGTHS['general'])
        if len(input_text) > max_length:
            result['errors'].append(f"Input too long. Maximum {max_length} characters allowed.")
            result['is_valid'] = False
            return result
        
        # Check for dangerous patterns
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                result['errors'].append("Input contains potentially dangerous content")
                result['is_valid'] = False
                logger.warning(f"Blocked dangerous pattern: {pattern} in input: {input_text[:100]}...")
                return result
        
        # Sanitize the input
        sanitized = cls._sanitize_text(input_text)
        result['sanitized_text'] = sanitized
        
        if sanitized != input_text:
            result['warnings'].append("Input was sanitized for safety")
        
        return result
    
    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        """Sanitize text by removing or escaping dangerous characters."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters except newlines, tabs, and carriage returns
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
        # Limit consecutive whitespace
        text = re.sub(r'\s{10,}', ' ' * 10, text)
        
        return text.strip()
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @classmethod
    def validate_filename(cls, filename: str) -> bool:
        """Validate filename for safety."""
        # Block dangerous filenames
        blocked_names = ['con', 'prn', 'aux', 'nul'] + [f'com{i}' for i in range(1, 10)] + [f'lpt{i}' for i in range(1, 10)]
        
        if filename.lower() in blocked_names:
            return False
        
        # Block dangerous characters
        dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
        if re.search(dangerous_chars, filename):
            return False
        
        return True


def rate_limit_check(func):
    """Decorator to apply rate limiting to functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize rate limiter if not exists
        if 'rate_limiter' not in st.session_state:
            st.session_state.rate_limiter = RateLimiter()
        
        # Get current user
        auth_manager = st.session_state.get('auth_manager')
        if auth_manager and auth_manager.is_authenticated():
            current_user = auth_manager.get_current_user()
            user_id = current_user['username']
        else:
            user_id = 'anonymous'
        
        # Check rate limit
        if st.session_state.rate_limiter.is_rate_limited(user_id):
            st.error("⚠️ Rate limit exceeded. Please wait before making more requests.")
            remaining = st.session_state.rate_limiter.get_remaining_requests(user_id)
            st.info(f"Rate limit: {remaining['user_remaining']} user requests remaining, "
                   f"{remaining['global_remaining']} global requests remaining")
            return None
        
        # Record the request
        st.session_state.rate_limiter.record_request(user_id)
        
        # Execute the function
        return func(*args, **kwargs)
    
    return wrapper


def display_rate_limit_info():
    """Display current rate limit status to users."""
    if 'rate_limiter' not in st.session_state:
        return
    
    # Get current user
    auth_manager = st.session_state.get('auth_manager')
    if auth_manager and auth_manager.is_authenticated():
        current_user = auth_manager.get_current_user()
        user_id = current_user['username']
    else:
        user_id = 'anonymous'
    
    remaining = st.session_state.rate_limiter.get_remaining_requests(user_id)
    
    # Show rate limit info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🚦 Rate Limits")
        
        # User rate limit
        user_pct = (remaining['user_remaining'] / remaining['user_limit']) * 100
        if user_pct > 50:
            color = "green"
        elif user_pct > 20:
            color = "orange"
        else:
            color = "red"
        
        st.markdown(f"**User Requests:** <span style='color:{color}'>{remaining['user_remaining']}/{remaining['user_limit']}</span>", 
                   unsafe_allow_html=True)
        
        # Global rate limit
        global_pct = (remaining['global_remaining'] / remaining['global_limit']) * 100
        if global_pct > 50:
            color = "green"
        elif global_pct > 20:
            color = "orange"
        else:
            color = "red"
        
        st.markdown(f"**Global Requests:** <span style='color:{color}'>{remaining['global_remaining']}/{remaining['global_limit']}</span>", 
                   unsafe_allow_html=True)
        
        st.caption("Limits reset every hour")


class SecurityManager:
    """Additional security features for production deployment."""
    
    def __init__(self):
        self.failed_login_attempts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
    
    def record_failed_login(self, username: str):
        """Record a failed login attempt."""
        current_time = time.time()
        
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = []
        
        self.failed_login_attempts[username].append(current_time)
        
        # Clean old attempts (older than lockout duration)
        cutoff_time = current_time - self.lockout_duration
        self.failed_login_attempts[username] = [
            attempt_time for attempt_time in self.failed_login_attempts[username]
            if attempt_time > cutoff_time
        ]
    
    def is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_login_attempts:
            return False
        
        current_time = time.time()
        cutoff_time = current_time - self.lockout_duration
        
        # Count recent failed attempts
        recent_attempts = [
            attempt_time for attempt_time in self.failed_login_attempts[username]
            if attempt_time > cutoff_time
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def clear_failed_attempts(self, username: str):
        """Clear failed login attempts for successful login."""
        if username in self.failed_login_attempts:
            self.failed_login_attempts[username] = []
    
    def get_lockout_remaining(self, username: str) -> int:
        """Get remaining lockout time in seconds."""
        if not self.is_account_locked(username):
            return 0
        
        if username not in self.failed_login_attempts:
            return 0
        
        # Find the most recent failed attempt
        recent_attempts = self.failed_login_attempts[username]
        if not recent_attempts:
            return 0
        
        last_attempt = max(recent_attempts)
        elapsed = time.time() - last_attempt
        remaining = max(0, self.lockout_duration - elapsed)
        
        return int(remaining)


# Initialize global instances
if 'security_manager' not in st.session_state:
    st.session_state.security_manager = SecurityManager()
