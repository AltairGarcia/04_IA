"""
OAuth2 Provider and Integration System

This module provides OAuth2 server implementation with support for external providers
like Google, GitHub, and Microsoft, along with secure token management.
"""

import os
import json
import secrets
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlencode, parse_qs, urlparse
import jwt
import sqlite3
from cryptography.fernet import Fernet
import hashlib
import base64

logger = logging.getLogger(__name__)

@dataclass
class OAuthProvider:
    """OAuth provider configuration."""
    name: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    user_info_url: str
    scope: str
    redirect_uri: str
    
@dataclass
class OAuthToken:
    """OAuth token data structure."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    scope: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_in or not self.created_at:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.expires_in)

@dataclass
class OAuthSession:
    """OAuth session for state management."""
    state: str
    user_id: Optional[str]
    provider: str
    redirect_uri: str
    scopes: List[str]
    created_at: datetime
    expires_at: datetime
    code_verifier: Optional[str] = None  # For PKCE
    
class OAuth2Manager:
    """Manages OAuth2 authentication and authorization."""
    
    def __init__(self, db_path: str = "oauth2.db", encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # OAuth2 settings
        self.session_timeout = timedelta(minutes=10)  # State session timeout
        self.token_refresh_buffer = timedelta(minutes=5)  # Refresh before expiry
        
        # Provider configurations
        self.providers = self._load_provider_configs()
        
        self._initialize_database()
    
    def _load_provider_configs(self) -> Dict[str, OAuthProvider]:
        """Load OAuth provider configurations."""
        providers = {}
        
        # Google OAuth2
        if all([os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET")]):
            providers["google"] = OAuthProvider(
                name="google",
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
                authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
                token_url="https://oauth2.googleapis.com/token",
                user_info_url="https://www.googleapis.com/oauth2/v2/userinfo",
                scope="openid email profile",
                redirect_uri=os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/auth/google/callback")
            )
        
        # GitHub OAuth2
        if all([os.getenv("GITHUB_CLIENT_ID"), os.getenv("GITHUB_CLIENT_SECRET")]):
            providers["github"] = OAuthProvider(
                name="github",
                client_id=os.getenv("GITHUB_CLIENT_ID"),
                client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
                authorize_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                user_info_url="https://api.github.com/user",
                scope="user:email",
                redirect_uri=os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8501/auth/github/callback")
            )
        
        # Microsoft OAuth2
        if all([os.getenv("MICROSOFT_CLIENT_ID"), os.getenv("MICROSOFT_CLIENT_SECRET")]):
            providers["microsoft"] = OAuthProvider(
                name="microsoft",
                client_id=os.getenv("MICROSOFT_CLIENT_ID"),
                client_secret=os.getenv("MICROSOFT_CLIENT_SECRET"),
                authorize_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
                user_info_url="https://graph.microsoft.com/v1.0/me",
                scope="openid profile email",
                redirect_uri=os.getenv("MICROSOFT_REDIRECT_URI", "http://localhost:8501/auth/microsoft/callback")
            )
        
        return providers
    
    def _initialize_database(self):
        """Initialize OAuth2 database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # OAuth sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS oauth_sessions (
                        state TEXT PRIMARY KEY,
                        user_id TEXT,
                        provider TEXT NOT NULL,
                        redirect_uri TEXT NOT NULL,
                        scopes TEXT NOT NULL,
                        code_verifier TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL
                    )
                """)
                
                # OAuth tokens table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS oauth_tokens (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        access_token TEXT NOT NULL,
                        refresh_token TEXT,
                        token_type TEXT DEFAULT 'Bearer',
                        expires_in INTEGER,
                        scope TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, provider),
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # OAuth user mappings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS oauth_user_mappings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        provider_user_id TEXT NOT NULL,
                        provider_username TEXT,
                        provider_email TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(provider, provider_user_id),
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.commit()
                logger.info("OAuth2 database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize OAuth2 database: {e}")
            raise
    
    def get_authorization_url(self, provider_name: str, user_id: Optional[str] = None, 
                            scopes: Optional[List[str]] = None) -> Optional[Tuple[str, str]]:
        """
        Generate OAuth2 authorization URL.
        
        Returns:
            Tuple of (authorization_url, state) or None if provider not found
        """
        try:
            provider = self.providers.get(provider_name)
            if not provider:
                logger.error(f"Provider {provider_name} not configured")
                return None
            
            # Generate state for CSRF protection
            state = secrets.token_urlsafe(32)
            
            # Use provided scopes or default
            request_scopes = scopes or provider.scope.split()
            
            # Create session
            session = OAuthSession(
                state=state,
                user_id=user_id,
                provider=provider_name,
                redirect_uri=provider.redirect_uri,
                scopes=request_scopes,
                created_at=datetime.now(),
                expires_at=datetime.now() + self.session_timeout
            )
            
            # Generate PKCE code verifier for enhanced security
            code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode('utf-8')).digest()
            ).decode('utf-8').rstrip('=')
            
            session.code_verifier = code_verifier
            
            # Save session
            self._save_oauth_session(session)
            
            # Build authorization URL
            params = {
                'client_id': provider.client_id,
                'redirect_uri': provider.redirect_uri,
                'scope': ' '.join(request_scopes),
                'response_type': 'code',
                'state': state,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256'
            }
            
            auth_url = f"{provider.authorize_url}?{urlencode(params)}"
            
            logger.info(f"Generated authorization URL for provider {provider_name}")
            return auth_url, state
            
        except Exception as e:
            logger.error(f"Failed to generate authorization URL: {e}")
            return None
    
    def handle_callback(self, provider_name: str, code: str, state: str) -> Optional[Dict[str, Any]]:
        """
        Handle OAuth2 callback and exchange code for tokens.
        
        Returns:
            Dictionary with user info and tokens or None on failure
        """
        try:
            # Verify state and get session
            session = self._get_oauth_session(state)
            if not session or session.provider != provider_name:
                logger.error("Invalid or expired OAuth state")
                return None
            
            provider = self.providers.get(provider_name)
            if not provider:
                logger.error(f"Provider {provider_name} not configured")
                return None
            
            # Exchange code for tokens
            token_data = self._exchange_code_for_tokens(provider, code, session.code_verifier)
            if not token_data:
                return None
            
            # Get user info from provider
            user_info = self._get_user_info(provider, token_data.access_token)
            if not user_info:
                return None
            
            # Clean up session
            self._delete_oauth_session(state)
            
            result = {
                'provider': provider_name,
                'tokens': asdict(token_data),
                'user_info': user_info,
                'session_user_id': session.user_id
            }
            
            logger.info(f"OAuth callback successful for provider {provider_name}")
            return result
            
        except Exception as e:
            logger.error(f"OAuth callback failed: {e}")
            return None
    
    def save_user_tokens(self, user_id: str, provider_name: str, tokens: OAuthToken, 
                        provider_user_info: Dict[str, Any]) -> bool:
        """Save OAuth tokens and user mapping."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Encrypt tokens
                encrypted_access_token = self.cipher_suite.encrypt(tokens.access_token.encode()).decode()
                encrypted_refresh_token = None
                if tokens.refresh_token:
                    encrypted_refresh_token = self.cipher_suite.encrypt(tokens.refresh_token.encode()).decode()
                
                # Save/update tokens
                cursor.execute("""
                    INSERT OR REPLACE INTO oauth_tokens 
                    (user_id, provider, access_token, refresh_token, token_type, expires_in, scope, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, provider_name, encrypted_access_token, encrypted_refresh_token,
                    tokens.token_type, tokens.expires_in, tokens.scope, tokens.created_at
                ))
                
                # Save/update user mapping
                provider_user_id = str(provider_user_info.get('id', provider_user_info.get('sub', '')))
                provider_username = provider_user_info.get('login', provider_user_info.get('name', ''))
                provider_email = provider_user_info.get('email', '')
                
                cursor.execute("""
                    INSERT OR REPLACE INTO oauth_user_mappings 
                    (user_id, provider, provider_user_id, provider_username, provider_email, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id, provider_name, provider_user_id, provider_username, 
                    provider_email, datetime.now()
                ))
                
                conn.commit()
                logger.info(f"OAuth tokens saved for user {user_id}, provider {provider_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save OAuth tokens: {e}")
            return False
    
    def get_user_tokens(self, user_id: str, provider_name: str) -> Optional[OAuthToken]:
        """Get OAuth tokens for user and provider."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT access_token, refresh_token, token_type, expires_in, scope, created_at
                    FROM oauth_tokens 
                    WHERE user_id = ? AND provider = ?
                """, (user_id, provider_name))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Decrypt tokens
                access_token = self.cipher_suite.decrypt(row[0].encode()).decode()
                refresh_token = None
                if row[1]:
                    refresh_token = self.cipher_suite.decrypt(row[1].encode()).decode()
                
                return OAuthToken(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_type=row[2],
                    expires_in=row[3],
                    scope=row[4],
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None
                )
                
        except Exception as e:
            logger.error(f"Failed to get OAuth tokens: {e}")
            return None
    
    def refresh_tokens(self, user_id: str, provider_name: str) -> Optional[OAuthToken]:
        """Refresh OAuth tokens if needed."""
        try:
            current_tokens = self.get_user_tokens(user_id, provider_name)
            if not current_tokens or not current_tokens.refresh_token:
                return None
            
            # Check if refresh is needed
            if not current_tokens.is_expired:
                # Check if we're within refresh buffer
                if current_tokens.created_at and current_tokens.expires_in:
                    expires_at = current_tokens.created_at + timedelta(seconds=current_tokens.expires_in)
                    if datetime.now() < expires_at - self.token_refresh_buffer:
                        return current_tokens
            
            provider = self.providers.get(provider_name)
            if not provider:
                return None
            
            # Refresh tokens
            new_tokens = self._refresh_access_token(provider, current_tokens.refresh_token)
            if new_tokens:
                # Save new tokens
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    encrypted_access_token = self.cipher_suite.encrypt(new_tokens.access_token.encode()).decode()
                    encrypted_refresh_token = None
                    if new_tokens.refresh_token:
                        encrypted_refresh_token = self.cipher_suite.encrypt(new_tokens.refresh_token.encode()).decode()
                    
                    cursor.execute("""
                        UPDATE oauth_tokens 
                        SET access_token = ?, refresh_token = ?, expires_in = ?, created_at = ?
                        WHERE user_id = ? AND provider = ?
                    """, (
                        encrypted_access_token, encrypted_refresh_token,
                        new_tokens.expires_in, new_tokens.created_at,
                        user_id, provider_name
                    ))
                
                logger.info(f"Tokens refreshed for user {user_id}, provider {provider_name}")
                return new_tokens
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to refresh tokens: {e}")
            return None
    
    def revoke_tokens(self, user_id: str, provider_name: str) -> bool:
        """Revoke OAuth tokens for user and provider."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM oauth_tokens 
                    WHERE user_id = ? AND provider = ?
                """, (user_id, provider_name))
                
                success = cursor.rowcount > 0
                if success:
                    logger.info(f"OAuth tokens revoked for user {user_id}, provider {provider_name}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to revoke OAuth tokens: {e}")
            return False
    
    def get_user_providers(self, user_id: str) -> List[Dict[str, Any]]:
        """Get connected OAuth providers for user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT provider, provider_username, provider_email, created_at
                    FROM oauth_user_mappings 
                    WHERE user_id = ?
                """, (user_id,))
                
                providers = []
                for row in cursor.fetchall():
                    providers.append({
                        'provider': row[0],
                        'username': row[1],
                        'email': row[2],
                        'connected_at': row[3]
                    })
                
                return providers
                
        except Exception as e:
            logger.error(f"Failed to get user providers: {e}")
            return []
    
    def _exchange_code_for_tokens(self, provider: OAuthProvider, code: str, code_verifier: str) -> Optional[OAuthToken]:
        """Exchange authorization code for access tokens."""
        try:
            data = {
                'client_id': provider.client_id,
                'client_secret': provider.client_secret,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': provider.redirect_uri,
                'code_verifier': code_verifier
            }
            
            headers = {'Accept': 'application/json'}
            
            response = requests.post(provider.token_url, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            
            return OAuthToken(
                access_token=token_data['access_token'],
                refresh_token=token_data.get('refresh_token'),
                token_type=token_data.get('token_type', 'Bearer'),
                expires_in=token_data.get('expires_in'),
                scope=token_data.get('scope')
            )
            
        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            return None
    
    def _refresh_access_token(self, provider: OAuthProvider, refresh_token: str) -> Optional[OAuthToken]:
        """Refresh access token using refresh token."""
        try:
            data = {
                'client_id': provider.client_id,
                'client_secret': provider.client_secret,
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'
            }
            
            headers = {'Accept': 'application/json'}
            
            response = requests.post(provider.token_url, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            
            return OAuthToken(
                access_token=token_data['access_token'],
                refresh_token=token_data.get('refresh_token', refresh_token),  # Some providers don't return new refresh token
                token_type=token_data.get('token_type', 'Bearer'),
                expires_in=token_data.get('expires_in'),
                scope=token_data.get('scope')
            )
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return None
    
    def _get_user_info(self, provider: OAuthProvider, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from OAuth provider."""
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(provider.user_info_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    def _save_oauth_session(self, session: OAuthSession):
        """Save OAuth session to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO oauth_sessions 
                (state, user_id, provider, redirect_uri, scopes, code_verifier, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.state, session.user_id, session.provider, session.redirect_uri,
                json.dumps(session.scopes), session.code_verifier,
                session.created_at, session.expires_at
            ))
    
    def _get_oauth_session(self, state: str) -> Optional[OAuthSession]:
        """Get OAuth session by state."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM oauth_sessions WHERE state = ? AND expires_at > ?
                """, (state, datetime.now()))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return OAuthSession(
                    state=row[0],
                    user_id=row[1],
                    provider=row[2],
                    redirect_uri=row[3],
                    scopes=json.loads(row[4]),
                    created_at=datetime.fromisoformat(row[6]),
                    expires_at=datetime.fromisoformat(row[7]),
                    code_verifier=row[5]
                )
                
        except Exception as e:
            logger.error(f"Failed to get OAuth session: {e}")
            return None
    
    def _delete_oauth_session(self, state: str):
        """Delete OAuth session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM oauth_sessions WHERE state = ?", (state,))
    
    def cleanup_expired_sessions(self):
        """Clean up expired OAuth sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM oauth_sessions WHERE expires_at < ?", (datetime.now(),))
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired OAuth sessions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
