"""
Security Headers & CORS Management System
Implements comprehensive security headers and CORS configuration for LangGraph 101.

Features:
- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- X-Frame-Options
- X-Content-Type-Options
- Referrer Policy
- Advanced CORS configuration
- Security header validation
- Dynamic policy management
"""

import logging
import re
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib
import base64
import secrets


class SecurityHeaderType(Enum):
    """Security header types."""
    CSP = "Content-Security-Policy"
    HSTS = "Strict-Transport-Security"
    FRAME_OPTIONS = "X-Frame-Options"
    CONTENT_TYPE_OPTIONS = "X-Content-Type-Options"
    REFERRER_POLICY = "Referrer-Policy"
    XSS_PROTECTION = "X-XSS-Protection"
    PERMISSIONS_POLICY = "Permissions-Policy"


class CSPDirective(Enum):
    """Content Security Policy directives."""
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    CONNECT_SRC = "connect-src"
    FONT_SRC = "font-src"
    OBJECT_SRC = "object-src"
    MEDIA_SRC = "media-src"
    FRAME_SRC = "frame-src"
    CHILD_SRC = "child-src"
    WORKER_SRC = "worker-src"
    MANIFEST_SRC = "manifest-src"
    BASE_URI = "base-uri"
    FORM_ACTION = "form-action"
    FRAME_ANCESTORS = "frame-ancestors"
    UPGRADE_INSECURE_REQUESTS = "upgrade-insecure-requests"


@dataclass
class CSPPolicy:
    """Content Security Policy configuration."""
    directives: Dict[CSPDirective, List[str]] = field(default_factory=dict)
    report_uri: Optional[str] = None
    report_only: bool = False
    nonce_enabled: bool = True
    strict_dynamic: bool = False
    
    def __post_init__(self):
        """Initialize default CSP policy."""
        if not self.directives:
            self.directives = {
                CSPDirective.DEFAULT_SRC: ["'self'"],
                CSPDirective.SCRIPT_SRC: ["'self'", "'unsafe-inline'"],
                CSPDirective.STYLE_SRC: ["'self'", "'unsafe-inline'"],
                CSPDirective.IMG_SRC: ["'self'", "data:", "https:"],
                CSPDirective.CONNECT_SRC: ["'self'"],
                CSPDirective.FONT_SRC: ["'self'"],
                CSPDirective.OBJECT_SRC: ["'none'"],
                CSPDirective.FRAME_ANCESTORS: ["'none'"],
                CSPDirective.BASE_URI: ["'self'"],
                CSPDirective.FORM_ACTION: ["'self'"]
            }


@dataclass
class CORSPolicy:
    """CORS policy configuration."""
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    exposed_headers: List[str] = field(default_factory=list)
    allow_credentials: bool = False
    max_age: int = 86400  # 24 hours
    preflight_continue: bool = False
    
    def __post_init__(self):
        """Validate CORS configuration."""
        if self.allow_credentials and "*" in self.allowed_origins:
            raise ValueError("Cannot use wildcard origin with credentials")


@dataclass
class SecurityHeadersConfig:
    """Security headers configuration."""
    csp_policy: CSPPolicy = field(default_factory=CSPPolicy)
    cors_policy: CORSPolicy = field(default_factory=CORSPolicy)
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = True
    frame_options: str = "DENY"
    content_type_options: bool = True
    xss_protection: bool = True
    referrer_policy: str = "strict-origin-when-cross-origin"
    permissions_policy: Dict[str, List[str]] = field(default_factory=dict)


class SecurityHeadersManager:
    """
    Manages security headers and CORS configuration.
    
    Provides comprehensive security headers management including CSP,
    HSTS, frame options, and advanced CORS configuration.
    """
    
    def __init__(self, config: Optional[SecurityHeadersConfig] = None):
        """Initialize security headers manager."""
        self.config = config or SecurityHeadersConfig()
        self.logger = logging.getLogger(__name__)
        self._nonce_cache: Dict[str, Tuple[str, datetime]] = {}
        self._setup_default_permissions_policy()
    
    def _setup_default_permissions_policy(self):
        """Setup default permissions policy."""
        if not self.config.permissions_policy:
            self.config.permissions_policy = {
                "geolocation": ["'none'"],
                "microphone": ["'none'"],
                "camera": ["'none'"],
                "payment": ["'none'"],
                "usb": ["'none'"],
                "accelerometer": ["'self'"],
                "gyroscope": ["'self'"]
            }
    
    def generate_nonce(self, session_id: str) -> str:
        """
        Generate CSP nonce for inline scripts/styles.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Base64 encoded nonce
        """
        # Clean up expired nonces
        self._cleanup_expired_nonces()
        
        # Generate new nonce
        nonce = base64.b64encode(secrets.token_bytes(16)).decode('ascii')
        self._nonce_cache[session_id] = (nonce, datetime.now())
        
        self.logger.debug(f"Generated CSP nonce for session: {session_id}")
        return nonce
    
    def _cleanup_expired_nonces(self):
        """Clean up expired nonces."""
        cutoff = datetime.now() - timedelta(hours=1)
        expired_sessions = [
            session_id for session_id, (_, timestamp) in self._nonce_cache.items()
            if timestamp < cutoff
        ]
        
        for session_id in expired_sessions:
            del self._nonce_cache[session_id]
    
    def build_csp_header(self, session_id: Optional[str] = None) -> str:
        """
        Build Content Security Policy header.
        
        Args:
            session_id: Session ID for nonce generation
            
        Returns:
            CSP header value
        """
        policy_parts = []
        
        for directive, sources in self.config.csp_policy.directives.items():
            sources_list = sources.copy()
            
            # Add nonce for script and style sources if enabled
            if (self.config.csp_policy.nonce_enabled and session_id and
                directive in [CSPDirective.SCRIPT_SRC, CSPDirective.STYLE_SRC]):
                nonce = self.generate_nonce(session_id)
                sources_list.append(f"'nonce-{nonce}'")
            
            # Add strict-dynamic for script-src if enabled
            if (self.config.csp_policy.strict_dynamic and 
                directive == CSPDirective.SCRIPT_SRC):
                sources_list.append("'strict-dynamic'")
            
            sources_str = " ".join(sources_list)
            policy_parts.append(f"{directive.value} {sources_str}")
        
        # Add upgrade-insecure-requests if enabled
        if CSPDirective.UPGRADE_INSECURE_REQUESTS in self.config.csp_policy.directives:
            policy_parts.append("upgrade-insecure-requests")
        
        # Add report URI if configured
        if self.config.csp_policy.report_uri:
            policy_parts.append(f"report-uri {self.config.csp_policy.report_uri}")
        
        return "; ".join(policy_parts)
    
    def build_hsts_header(self) -> str:
        """
        Build HTTP Strict Transport Security header.
        
        Returns:
            HSTS header value
        """
        hsts_parts = [f"max-age={self.config.hsts_max_age}"]
        
        if self.config.hsts_include_subdomains:
            hsts_parts.append("includeSubDomains")
        
        if self.config.hsts_preload:
            hsts_parts.append("preload")
        
        return "; ".join(hsts_parts)
    
    def build_permissions_policy_header(self) -> str:
        """
        Build Permissions Policy header.
        
        Returns:
            Permissions Policy header value
        """
        policy_parts = []
        
        for feature, allowlist in self.config.permissions_policy.items():
            allowlist_str = " ".join(allowlist)
            policy_parts.append(f"{feature}=({allowlist_str})")
        
        return ", ".join(policy_parts)
    
    def get_security_headers(self, session_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get all security headers.
        
        Args:
            session_id: Session ID for nonce generation
            
        Returns:
            Dictionary of security headers
        """
        headers = {}
        
        # Content Security Policy
        csp_header_name = (
            "Content-Security-Policy-Report-Only" 
            if self.config.csp_policy.report_only 
            else "Content-Security-Policy"
        )
        headers[csp_header_name] = self.build_csp_header(session_id)
        
        # HTTP Strict Transport Security
        headers["Strict-Transport-Security"] = self.build_hsts_header()
        
        # X-Frame-Options
        headers["X-Frame-Options"] = self.config.frame_options
        
        # X-Content-Type-Options
        if self.config.content_type_options:
            headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection
        if self.config.xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer Policy
        headers["Referrer-Policy"] = self.config.referrer_policy
        
        # Permissions Policy
        if self.config.permissions_policy:
            headers["Permissions-Policy"] = self.build_permissions_policy_header()
        
        return headers
    
    def get_cors_headers(self, origin: Optional[str] = None, 
                        method: Optional[str] = None) -> Dict[str, str]:
        """
        Get CORS headers for a request.
        
        Args:
            origin: Request origin
            method: Request method
            
        Returns:
            Dictionary of CORS headers
        """
        headers = {}
        
        # Check if origin is allowed
        if origin and self._is_origin_allowed(origin):
            headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.config.cors_policy.allowed_origins:
            headers["Access-Control-Allow-Origin"] = "*"
        
        # Allow credentials
        if self.config.cors_policy.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        # Allowed methods
        headers["Access-Control-Allow-Methods"] = ", ".join(
            self.config.cors_policy.allowed_methods
        )
        
        # Allowed headers
        if "*" in self.config.cors_policy.allowed_headers:
            headers["Access-Control-Allow-Headers"] = "*"
        else:
            headers["Access-Control-Allow-Headers"] = ", ".join(
                self.config.cors_policy.allowed_headers
            )
        
        # Exposed headers
        if self.config.cors_policy.exposed_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(
                self.config.cors_policy.exposed_headers
            )
        
        # Max age for preflight
        headers["Access-Control-Max-Age"] = str(self.config.cors_policy.max_age)
        
        return headers
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed.
        
        Args:
            origin: Request origin
            
        Returns:
            True if origin is allowed
        """
        if "*" in self.config.cors_policy.allowed_origins:
            return True
        
        for allowed_origin in self.config.cors_policy.allowed_origins:
            if self._match_origin(origin, allowed_origin):
                return True
        
        return False
    
    def _match_origin(self, origin: str, pattern: str) -> bool:
        """
        Match origin against pattern.
        
        Args:
            origin: Request origin
            pattern: Allowed origin pattern
            
        Returns:
            True if origin matches pattern
        """
        if pattern == origin:
            return True
        
        # Support wildcard subdomains
        if pattern.startswith("*."):
            domain = pattern[2:]
            parsed_origin = urlparse(origin)
            return parsed_origin.netloc.endswith(f".{domain}") or parsed_origin.netloc == domain
        
        return False
    
    def validate_csp_compliance(self, content: str) -> List[str]:
        """
        Validate content against CSP policy.
        
        Args:
            content: HTML content to validate
            
        Returns:
            List of CSP violations
        """
        violations = []
        
        # Check for inline scripts
        if re.search(r'<script(?![^>]*src=)[^>]*>', content, re.IGNORECASE):
            if ("'unsafe-inline'" not in 
                self.config.csp_policy.directives.get(CSPDirective.SCRIPT_SRC, [])):
                violations.append("Inline script without unsafe-inline or nonce")
        
        # Check for inline styles
        if re.search(r'<style[^>]*>', content, re.IGNORECASE) or re.search(r'style=', content, re.IGNORECASE):
            if ("'unsafe-inline'" not in 
                self.config.csp_policy.directives.get(CSPDirective.STYLE_SRC, [])):
                violations.append("Inline style without unsafe-inline or nonce")
        
        # Check for eval usage
        if re.search(r'\beval\s*\(', content, re.IGNORECASE):
            if ("'unsafe-eval'" not in 
                self.config.csp_policy.directives.get(CSPDirective.SCRIPT_SRC, [])):
                violations.append("eval() usage without unsafe-eval")
        
        return violations
    
    def update_csp_policy(self, directive: CSPDirective, sources: List[str]):
        """
        Update CSP policy directive.
        
        Args:
            directive: CSP directive to update
            sources: New sources for directive
        """
        self.config.csp_policy.directives[directive] = sources
        self.logger.info(f"Updated CSP directive {directive.value}: {sources}")
    
    def add_trusted_source(self, directive: CSPDirective, source: str):
        """
        Add trusted source to CSP directive.
        
        Args:
            directive: CSP directive
            source: Source to add
        """
        if directive not in self.config.csp_policy.directives:
            self.config.csp_policy.directives[directive] = []
        
        if source not in self.config.csp_policy.directives[directive]:
            self.config.csp_policy.directives[directive].append(source)
            self.logger.info(f"Added trusted source to {directive.value}: {source}")
    
    def get_nonce_for_session(self, session_id: str) -> Optional[str]:
        """
        Get current nonce for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Current nonce or None
        """
        if session_id in self._nonce_cache:
            nonce, timestamp = self._nonce_cache[session_id]
            # Check if nonce is still valid (1 hour)
            if datetime.now() - timestamp < timedelta(hours=1):
                return nonce
            else:
                del self._nonce_cache[session_id]
        
        return None
    
    def generate_csp_report_handler(self):
        """
        Generate CSP violation report handler.
        
        Returns:
            Report handler function
        """
        def handle_csp_report(report_data: Dict[str, Any]):
            """Handle CSP violation report."""
            try:
                csp_report = report_data.get('csp-report', {})
                
                violation_info = {
                    'timestamp': datetime.now().isoformat(),
                    'document_uri': csp_report.get('document-uri'),
                    'violated_directive': csp_report.get('violated-directive'),
                    'blocked_uri': csp_report.get('blocked-uri'),
                    'source_file': csp_report.get('source-file'),
                    'line_number': csp_report.get('line-number'),
                    'column_number': csp_report.get('column-number')
                }
                
                self.logger.warning(f"CSP violation reported: {violation_info}")
                
                # Here you could store violations in database or send alerts
                
            except Exception as e:
                self.logger.error(f"Error handling CSP report: {e}")
        
        return handle_csp_report


def create_secure_headers_middleware():
    """
    Create middleware for applying security headers.
    
    Returns:
        Middleware function
    """
    headers_manager = SecurityHeadersManager()
    
    def security_headers_middleware(request, response, session_id: Optional[str] = None):
        """Apply security headers to response."""
        try:
            # Get security headers
            security_headers = headers_manager.get_security_headers(session_id)
            
            # Get CORS headers if needed
            origin = request.headers.get('Origin')
            cors_headers = headers_manager.get_cors_headers(origin)
            
            # Apply headers to response
            for header_name, header_value in {**security_headers, **cors_headers}.items():
                response.headers[header_name] = header_value
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error applying security headers: {e}")
    
    return security_headers_middleware


# Example usage and testing
if __name__ == "__main__":
    # Initialize security headers manager
    config = SecurityHeadersConfig()
    headers_manager = SecurityHeadersManager(config)
    
    # Test CSP header generation
    print("=== Content Security Policy ===")
    csp_header = headers_manager.build_csp_header("test-session")
    print(f"CSP: {csp_header}")
    
    # Test HSTS header
    print("\n=== HSTS Header ===")
    hsts_header = headers_manager.build_hsts_header()
    print(f"HSTS: {hsts_header}")
    
    # Test all security headers
    print("\n=== All Security Headers ===")
    all_headers = headers_manager.get_security_headers("test-session")
    for name, value in all_headers.items():
        print(f"{name}: {value}")
    
    # Test CORS headers
    print("\n=== CORS Headers ===")
    cors_headers = headers_manager.get_cors_headers("https://example.com")
    for name, value in cors_headers.items():
        print(f"{name}: {value}")
    
    # Test CSP compliance validation
    print("\n=== CSP Compliance Validation ===")
    test_html = """
    <html>
        <head>
            <script>alert('test');</script>
            <style>body { color: red; }</style>
        </head>
        <body style="background: blue;">
            <div onclick="eval('alert()')">Click me</div>
        </body>
    </html>
    """
    violations = headers_manager.validate_csp_compliance(test_html)
    print(f"CSP Violations: {violations}")
