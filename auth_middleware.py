"""
Authentication Middleware for LangGraph 101 Application
Provides secure authentication and authorization for production deployment.
"""

import os
import jwt
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps
from dataclasses import dataclass
import streamlit as st
from core.config import get_config

# Import production features for security
try:
    from production_features import SecurityManager
except ImportError:
    SecurityManager = None

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User data structure for authentication."""
    id: str
    username: str
    email: str
    role: str = "user"
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None


class AuthManager:
    """Manages authentication and session handling for Streamlit app."""
    
    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.security.secret_key
        self.jwt_secret = self.config.security.jwt_secret
        self.session_timeout = self.config.security.session_timeout
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'auth_timestamp' not in st.session_state:
            st.session_state.auth_timestamp = None
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = self.secret_key[:16]  # Use part of secret key as salt
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password) == hashed
    
    def create_token(self, user_id: str, username: str) -> str:
        """Create JWT token for user."""
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': datetime.utcnow() + timedelta(seconds=self.session_timeout),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        # Check security manager for account lockout
        if SecurityManager and hasattr(st.session_state, 'security_manager'):
            security_mgr = st.session_state.security_manager
            if security_mgr.is_account_locked(username):
                remaining = security_mgr.get_lockout_remaining(username)
                logger.warning(f"Account {username} is locked. Remaining: {remaining}s")
                return None
        
        # For demo purposes, using hardcoded admin user
        # In production, this would connect to a user database
        admin_user = {
            'id': '1',
            'username': 'admin',
            'password_hash': self.hash_password('admin123'),  # Default password
            'email': 'admin@langgraph101.com',
            'role': 'admin'
        }
        
        demo_user = {
            'id': '2',
            'username': 'demo',
            'password_hash': self.hash_password('demo123'),  # Default password
            'email': 'demo@langgraph101.com',
            'role': 'user'
        }
        
        users = [admin_user, demo_user]
        
        for user_data in users:
            if user_data['username'] == username:
                if self.verify_password(password, user_data['password_hash']):
                    # Clear failed attempts on successful login
                    if SecurityManager and hasattr(st.session_state, 'security_manager'):
                        st.session_state.security_manager.clear_failed_attempts(username)
                    
                    return User(
                        id=user_data['id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        role=user_data['role'],
                        last_login=datetime.now()
                    )
        
        # Record failed login attempt
        if SecurityManager and hasattr(st.session_state, 'security_manager'):
            st.session_state.security_manager.record_failed_login(username)
        
        return None
    
    def login(self, username: str, password: str) -> bool:
        """Handle user login."""
        user = self.authenticate_user(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.user = user
            st.session_state.auth_timestamp = time.time()
            
            # Create token for session
            token = self.create_token(user.id, user.username)
            st.session_state.auth_token = token
            
            logger.info(f"User {username} authenticated successfully")
            return True
        
        logger.warning(f"Failed authentication attempt for {username}")
        return False
    
    def logout(self):
        """Handle user logout."""
        if st.session_state.user:
            logger.info(f"User {st.session_state.user.username} logged out")
        
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.auth_timestamp = None
        if 'auth_token' in st.session_state:
            del st.session_state.auth_token
    
    def is_authenticated(self) -> bool:
        """Check if current session is authenticated."""
        if not st.session_state.authenticated:
            return False
        
        # Check session timeout
        if st.session_state.auth_timestamp:
            if time.time() - st.session_state.auth_timestamp > self.session_timeout:
                self.logout()
                return False
        
        # Verify token if present
        if 'auth_token' in st.session_state:
            payload = self.verify_token(st.session_state.auth_token)
            if not payload:
                self.logout()
                return False
        
        return True
    
    def get_current_user(self) -> Optional[User]:
        """Get current authenticated user."""
        if self.is_authenticated():
            return st.session_state.user
        return None
    
    def require_role(self, required_role: str) -> bool:
        """Check if current user has required role."""
        user = self.get_current_user()
        if not user:
            return False
        
        role_hierarchy = {'admin': 2, 'user': 1}
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def render_login_form(self) -> bool:
        """Render login form in Streamlit."""
        st.title("🔐 LangGraph 101 Authentication")
        st.markdown("Please log in to access the application.")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if self.login(username, password):
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        
        # Demo credentials info
        with st.expander("Demo Credentials"):
            st.markdown("""
            **Admin User:**
            - Username: `admin`
            - Password: `admin123`
            
            **Demo User:**
            - Username: `demo`  
            - Password: `demo123`
            """)
        
        return False


def require_authentication(func: Callable) -> Callable:
    """Decorator to require authentication for Streamlit pages."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_manager = AuthManager()
        
        if not auth_manager.is_authenticated():
            auth_manager.render_login_form()
            return None
        
        return func(*args, **kwargs)
    
    return wrapper


def require_role(role: str) -> Callable:
    """Decorator to require specific role for Streamlit pages."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth_manager = AuthManager()
            
            if not auth_manager.is_authenticated():
                auth_manager.render_login_form()
                return None
            
            if not auth_manager.require_role(role):
                st.error(f"Access denied. Required role: {role}")
                return None
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Global auth manager instance
auth_manager = AuthManager()


# Example usage functions
def render_user_info():
    """Render current user information."""
    user = auth_manager.get_current_user()
    if user:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Info")
            st.write(f"**Username:** {user.username}")
            st.write(f"**Role:** {user.role}")
            st.write(f"**Email:** {user.email}")
            
            if st.button("Logout"):
                auth_manager.logout()
                st.experimental_rerun()


def render_admin_panel():
    """Render admin panel (admin only)."""
    if auth_manager.require_role('admin'):
        st.markdown("### 🛠️ Admin Panel")
        st.info("Admin-only features would go here")
        return True
    return False


if __name__ == "__main__":
    # Test authentication system
    auth = AuthManager()
    
    # Test password hashing
    password = "test123"
    hashed = auth.hash_password(password)
    print(f"Password: {password}")
    print(f"Hashed: {hashed}")
    print(f"Verified: {auth.verify_password(password, hashed)}")
    
    # Test token creation
    token = auth.create_token("user1", "testuser")
    print(f"Token: {token}")
    
    payload = auth.verify_token(token)
    print(f"Payload: {payload}")
