"""
Input Security & Validation System
Comprehensive input validation and sanitization for LangGraph 101.

Features:
- SQL injection prevention
- XSS protection
- Command injection protection
- File upload security
- Schema validation
- Data sanitization
- Input filtering
- Malicious payload detection
"""

import logging
import re
import html
import json
import hashlib
import mimetypes
import tempfile
import os
from typing import Dict, List, Optional, Union, Any, Callable, Pattern, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import urllib.parse
import base64
try:
    import magic  # python-magic for file type detection
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available, using basic file type detection")
from pathlib import Path
import bleach
from sqlalchemy import text
from jsonschema import validate, ValidationError


class ValidationLevel(Enum):
    """Input validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class InputType(Enum):
    """Input data types."""
    TEXT = "text"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    NUMBER = "number"
    DATE = "date"
    JSON = "json"
    HTML = "html"
    SQL = "sql"
    FILE = "file"
    PATH = "path"
    COMMAND = "command"


@dataclass
class ValidationRule:
    """Input validation rule definition."""
    input_type: InputType
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[Pattern] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    sanitizer: Optional[Callable] = None
    error_message: Optional[str] = None


@dataclass
class FileValidationConfig:
    """File upload validation configuration."""
    allowed_extensions: List[str] = field(default_factory=lambda: ['.txt', '.pdf', '.jpg', '.png'])
    allowed_mime_types: List[str] = field(default_factory=lambda: ['text/plain', 'application/pdf', 'image/jpeg', 'image/png'])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    scan_for_malware: bool = True
    quarantine_suspicious: bool = True
    allowed_magic_numbers: Dict[str, bytes] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize magic numbers for file type detection."""
        if not self.allowed_magic_numbers:
            self.allowed_magic_numbers = {
                'pdf': b'%PDF-',
                'jpg': b'\xff\xd8\xff',
                'png': b'\x89PNG\r\n\x1a\n',
                'txt': b'',  # Text files don't have magic numbers
            }


@dataclass
class InputSecurityConfig:
    """Input security configuration."""
    validation_level: ValidationLevel = ValidationLevel.STRICT
    file_config: FileValidationConfig = field(default_factory=FileValidationConfig)
    xss_protection: bool = True
    sql_injection_protection: bool = True
    command_injection_protection: bool = True
    path_traversal_protection: bool = True
    sanitize_html: bool = True
    allowed_html_tags: List[str] = field(default_factory=lambda: ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li'])
    allowed_html_attributes: Dict[str, List[str]] = field(default_factory=dict)
    max_input_length: int = 10000
    enable_logging: bool = True


class InputSecurityManager:
    """
    Comprehensive input security and validation manager.
    
    Provides protection against common input-based attacks including
    XSS, SQL injection, command injection, and file upload attacks.
    """
    
    # Common attack patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
        re.compile(r"(\b(OR|AND)\s+\d+\s*=\s*\d+)", re.IGNORECASE),
        re.compile(r"(\bOR\s+'\w+'\s*=\s*'\w+')", re.IGNORECASE),
        re.compile(r"(--|/\*|\*/|;|'|\")", re.IGNORECASE),
        re.compile(r"(\bhex\(|\bchar\(|\bconcat\()", re.IGNORECASE),
    ]
    
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<object[^>]*>", re.IGNORECASE),
        re.compile(r"<embed[^>]*>", re.IGNORECASE),
        re.compile(r"vbscript:", re.IGNORECASE),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r"[;&|`$(){}]"),
        re.compile(r"\b(rm|del|format|sudo|su|passwd|chmod|chown)\b", re.IGNORECASE),
        re.compile(r"(>|<|>>|<<|\|)"),
        re.compile(r"\$\([^)]*\)"),
        re.compile(r"`[^`]*`"),
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\.[\\/]"),
        re.compile(r"[\\/]\.\."),
        re.compile(r"\%2e\%2e[\\/]", re.IGNORECASE),
        re.compile(r"[\\/]\%2e\%2e", re.IGNORECASE),
    ]
    
    def __init__(self, config: Optional[InputSecurityConfig] = None):
        """Initialize input security manager."""
        self.config = config or InputSecurityConfig()
        self.logger = logging.getLogger(__name__)
        self.validation_rules: Dict[str, ValidationRule] = {}
        self._setup_default_rules()
        self._setup_bleach_config()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        self.validation_rules.update({
            'email': ValidationRule(
                input_type=InputType.EMAIL,
                pattern=re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
                max_length=254,
                error_message="Invalid email format"
            ),
            'url': ValidationRule(
                input_type=InputType.URL,
                pattern=re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
                max_length=2048,
                error_message="Invalid URL format"
            ),
            'phone': ValidationRule(
                input_type=InputType.PHONE,
                pattern=re.compile(r'^\+?[\d\s\-\(\)]{10,}$'),
                error_message="Invalid phone number format"
            ),
            'username': ValidationRule(
                input_type=InputType.TEXT,
                pattern=re.compile(r'^[a-zA-Z0-9_-]{3,50}$'),
                min_length=3,
                max_length=50,
                error_message="Username must be 3-50 characters, alphanumeric, underscore or dash only"
            ),
            'password': ValidationRule(
                input_type=InputType.TEXT,
                min_length=8,
                max_length=128,
                pattern=re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'),
                error_message="Password must be 8+ characters with uppercase, lowercase, number and special character"
            )
        })
    
    def _setup_bleach_config(self):
        """Setup HTML sanitization configuration."""
        if not self.config.allowed_html_attributes:
            self.config.allowed_html_attributes = {
                'a': ['href', 'title'],
                'img': ['src', 'alt', 'title'],
                'p': ['class'],
                'div': ['class'],
            }
    
    def add_validation_rule(self, name: str, rule: ValidationRule):
        """
        Add custom validation rule.
        
        Args:
            name: Rule name
            rule: Validation rule
        """
        self.validation_rules[name] = rule
        self.logger.debug(f"Added validation rule: {name}")
    
    def validate_input(self, data: Any, rule_name: str) -> Tuple[bool, Optional[str], Any]:
        """
        Validate input against specified rule.
        
        Args:
            data: Input data to validate
            rule_name: Name of validation rule
            
        Returns:
            Tuple of (is_valid, error_message, sanitized_data)
        """
        if rule_name not in self.validation_rules:
            return False, f"Unknown validation rule: {rule_name}", None
        
        rule = self.validation_rules[rule_name]
        
        try:
            # Check if required
            if rule.required and (data is None or data == ""):
                return False, "Field is required", None
            
            # Skip validation for empty optional fields
            if not rule.required and (data is None or data == ""):
                return True, None, data
            
            # Convert to string for validation
            str_data = str(data) if data is not None else ""
            
            # Length validation
            if rule.min_length and len(str_data) < rule.min_length:
                return False, rule.error_message or f"Minimum length is {rule.min_length}", None
            
            if rule.max_length and len(str_data) > rule.max_length:
                return False, rule.error_message or f"Maximum length is {rule.max_length}", None
            
            # Pattern validation
            if rule.pattern and not rule.pattern.match(str_data):
                return False, rule.error_message or "Invalid format", None
            
            # Allowed values validation
            if rule.allowed_values and data not in rule.allowed_values:
                return False, rule.error_message or f"Value must be one of: {rule.allowed_values}", None
            
            # Custom validator
            if rule.custom_validator:
                is_valid, error_msg = rule.custom_validator(data)
                if not is_valid:
                    return False, error_msg, None
            
            # Security checks based on input type
            sanitized_data = self._apply_security_checks(data, rule.input_type)
            
            # Apply sanitizer if configured
            if rule.sanitizer:
                sanitized_data = rule.sanitizer(sanitized_data)
            
            return True, None, sanitized_data
            
        except Exception as e:
            self.logger.error(f"Validation error for rule {rule_name}: {e}")
            return False, "Validation failed", None
    
    def _apply_security_checks(self, data: Any, input_type: InputType) -> Any:
        """
        Apply security checks based on input type.
        
        Args:
            data: Input data
            input_type: Type of input
            
        Returns:
            Sanitized data
        """
        str_data = str(data) if data is not None else ""
        
        # General length check
        if len(str_data) > self.config.max_input_length:
            raise ValueError("Input too long")
        
        # SQL injection protection
        if self.config.sql_injection_protection:
            if self._detect_sql_injection(str_data):
                raise ValueError("Potential SQL injection detected")
        
        # XSS protection
        if self.config.xss_protection and input_type in [InputType.TEXT, InputType.HTML]:
            if self._detect_xss(str_data):
                if self.config.sanitize_html:
                    str_data = self._sanitize_html(str_data)
                else:
                    raise ValueError("Potential XSS attack detected")
        
        # Command injection protection
        if self.config.command_injection_protection and input_type == InputType.COMMAND:
            if self._detect_command_injection(str_data):
                raise ValueError("Potential command injection detected")
        
        # Path traversal protection
        if self.config.path_traversal_protection and input_type == InputType.PATH:
            if self._detect_path_traversal(str_data):
                raise ValueError("Potential path traversal detected")
        
        return str_data if isinstance(data, str) else data
    
    def _detect_sql_injection(self, data: str) -> bool:
        """
        Detect SQL injection patterns.
        
        Args:
            data: Input data to check
            
        Returns:
            True if SQL injection detected
        """
        for pattern in self.SQL_INJECTION_PATTERNS:
            if pattern.search(data):
                self.logger.warning(f"SQL injection pattern detected: {pattern.pattern}")
                return True
        return False
    
    def _detect_xss(self, data: str) -> bool:
        """
        Detect XSS patterns.
        
        Args:
            data: Input data to check
            
        Returns:
            True if XSS detected
        """
        for pattern in self.XSS_PATTERNS:
            if pattern.search(data):
                self.logger.warning(f"XSS pattern detected: {pattern.pattern}")
                return True
        return False
    
    def _detect_command_injection(self, data: str) -> bool:
        """
        Detect command injection patterns.
        
        Args:
            data: Input data to check
            
        Returns:
            True if command injection detected
        """
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if pattern.search(data):
                self.logger.warning(f"Command injection pattern detected: {pattern.pattern}")
                return True
        return False
    
    def _detect_path_traversal(self, data: str) -> bool:
        """
        Detect path traversal patterns.
        
        Args:
            data: Input data to check
            
        Returns:
            True if path traversal detected
        """
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(data):
                self.logger.warning(f"Path traversal pattern detected: {pattern.pattern}")
                return True
        return False
    
    def _sanitize_html(self, html_content: str) -> str:
        """
        Sanitize HTML content.
        
        Args:
            html_content: HTML to sanitize
            
        Returns:
            Sanitized HTML
        """
        return bleach.clean(
            html_content,
            tags=self.config.allowed_html_tags,
            attributes=self.config.allowed_html_attributes,
            strip=True
        )
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # HTML escape
        sanitized = html.escape(text)
        
        # URL decode to prevent double encoding attacks
        sanitized = urllib.parse.unquote(sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize unicode
        sanitized = sanitized.encode('utf-8', 'ignore').decode('utf-8')
        
        return sanitized
    
    def validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON data against schema.
        
        Args:
            data: JSON data to validate
            schema: JSON schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    def validate_file_upload(self, file_path: str, original_filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file upload security.
        
        Args:
            file_path: Path to uploaded file
            original_filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File not found"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.config.file_config.max_file_size:
                return False, f"File too large. Maximum size: {self.config.file_config.max_file_size} bytes"
            
            # Check file extension
            file_ext = Path(original_filename).suffix.lower()
            if file_ext not in self.config.file_config.allowed_extensions:
                return False, f"File extension not allowed: {file_ext}"
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(original_filename)
            if mime_type and mime_type not in self.config.file_config.allowed_mime_types:
                return False, f"MIME type not allowed: {mime_type}"
            
            # Verify file type using magic numbers
            if not self._verify_file_type(file_path, file_ext):
                return False, "File type verification failed"
            
            # Scan for malware (placeholder - would integrate with antivirus)
            if self.config.file_config.scan_for_malware:
                if self._scan_for_malware(file_path):
                    return False, "Malware detected in file"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False, "File validation failed"
    
    def _verify_file_type(self, file_path: str, expected_ext: str) -> bool:
        """
        Verify file type using magic numbers.
        
        Args:
            file_path: Path to file
            expected_ext: Expected extension
            
        Returns:
            True if file type matches extension
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)  # Read first 32 bytes
            
            # Get expected magic number
            expected_magic = self.config.file_config.allowed_magic_numbers.get(
                expected_ext.lstrip('.'), b''
            )
            
            # For text files, just check if it's valid UTF-8
            if expected_ext in ['.txt', '.csv', '.json']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read(1024)  # Try to read first 1KB as UTF-8
                    return True
                except UnicodeDecodeError:
                    return False
            
            # For other files, check magic numbers
            if expected_magic and not header.startswith(expected_magic):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"File type verification error: {e}")
            return False
    
    def _scan_for_malware(self, file_path: str) -> bool:
        """
        Scan file for malware (placeholder implementation).
        
        Args:
            file_path: Path to file
            
        Returns:
            True if malware detected
        """
        # This is a placeholder - in production, you would integrate
        # with antivirus software like ClamAV
        
        # Basic checks for suspicious patterns
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024 * 1024)  # Read first 1MB
            
            # Check for executable headers
            suspicious_headers = [
                b'MZ',  # PE executable
                b'\x7fELF',  # ELF executable
                b'\xca\xfe\xba\xbe',  # Mach-O executable
            ]
            
            for header in suspicious_headers:
                if content.startswith(header):
                    self.logger.warning(f"Suspicious executable header detected: {header}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Malware scan error: {e}")
            return True  # Fail secure
    
    def batch_validate(self, data_dict: Dict[str, Any], 
                      rules_dict: Dict[str, str]) -> Dict[str, Tuple[bool, Optional[str], Any]]:
        """
        Validate multiple inputs at once.
        
        Args:
            data_dict: Dictionary of field names to values
            rules_dict: Dictionary of field names to rule names
            
        Returns:
            Dictionary of field names to validation results
        """
        results = {}
        
        for field_name, value in data_dict.items():
            if field_name in rules_dict:
                rule_name = rules_dict[field_name]
                results[field_name] = self.validate_input(value, rule_name)
            else:
                results[field_name] = (False, f"No validation rule for field: {field_name}", None)
        
        return results
    
    def create_form_validator(self, form_schema: Dict[str, str]):
        """
        Create a form validator function.
        
        Args:
            form_schema: Dictionary mapping field names to rule names
            
        Returns:
            Validator function
        """
        def validate_form(form_data: Dict[str, Any]) -> Tuple[bool, Dict[str, str], Dict[str, Any]]:
            """
            Validate form data.
            
            Returns:
                Tuple of (is_valid, errors_dict, sanitized_data)
            """
            results = self.batch_validate(form_data, form_schema)
            
            is_valid = all(result[0] for result in results.values())
            errors = {field: result[1] for field, result in results.items() if not result[0]}
            sanitized = {field: result[2] for field, result in results.items() if result[0]}
            
            return is_valid, errors, sanitized
        
        return validate_form


# Example usage and testing
if __name__ == "__main__":
    # Initialize input security manager
    config = InputSecurityConfig()
    security_manager = InputSecurityManager(config)
    
    # Test input validation
    print("=== Input Validation Tests ===")
    
    test_cases = [
        ("test@example.com", "email"),
        ("invalid-email", "email"),
        ("https://example.com", "url"),
        ("javascript:alert(1)", "url"),
        ("user123", "username"),
        ("<script>alert('xss')</script>", "username"),
        ("ValidPass123!", "password"),
        ("weak", "password"),
    ]
    
    for test_input, rule_name in test_cases:
        is_valid, error, sanitized = security_manager.validate_input(test_input, rule_name)
        status = "✓" if is_valid else "✗"
        print(f"{status} {rule_name}: '{test_input}' -> Valid: {is_valid}")
        if error:
            print(f"   Error: {error}")
        if sanitized != test_input:
            print(f"   Sanitized: '{sanitized}'")
    
    # Test XSS detection
    print("\n=== XSS Detection Tests ===")
    xss_tests = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "Normal text content",
    ]
    
    for test_input in xss_tests:
        detected = security_manager._detect_xss(test_input)
        status = "✗ DETECTED" if detected else "✓ CLEAN"
        print(f"{status}: '{test_input}'")
    
    # Test SQL injection detection
    print("\n=== SQL Injection Detection Tests ===")
    sql_tests = [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "UNION SELECT * FROM passwords",
        "normal search term",
    ]
    
    for test_input in sql_tests:
        detected = security_manager._detect_sql_injection(test_input)
        status = "✗ DETECTED" if detected else "✓ CLEAN"
        print(f"{status}: '{test_input}'")
    
    # Test form validation
    print("\n=== Form Validation Test ===")
    form_schema = {
        'email': 'email',
        'username': 'username',
        'password': 'password'
    }
    
    form_validator = security_manager.create_form_validator(form_schema)
    
    form_data = {
        'email': 'user@example.com',
        'username': 'testuser',
        'password': 'SecurePass123!'
    }
    
    is_valid, errors, sanitized = form_validator(form_data)
    print(f"Form valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    print(f"Sanitized data: {sanitized}")


class InputValidator:
    """Input validation system for security"""
    
    def __init__(self, max_length: int = 10000, 
                 allowed_file_types: Optional[List[str]] = None):
        self.max_length = max_length
        self.allowed_file_types = allowed_file_types or [
            '.txt', '.py', '.json', '.yaml', '.yml', '.md', '.html', '.css', '.js'
        ]
          # SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"][^'\"]*['\"]\\s*=\\s*['\"][^'\"]*['\"])"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>"
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e\\",
            r"\\\\[\w\$]+\\[\w\$]+"
        ]
    
    def validate_input(self, input_data: str, input_type: str = "text") -> Dict[str, Any]:
        """Validate input data for security threats"""
        result = {
            'valid': True,
            'threats_detected': [],
            'sanitized_input': input_data,
            'risk_level': 'low'
        }
        
        if not input_data:
            return result
        
        # Length validation
        if len(input_data) > self.max_length:
            result['valid'] = False
            result['threats_detected'].append('input_too_long')
            result['risk_level'] = 'medium'
        
        # SQL injection detection
        import re
        for pattern in self.sql_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                result['valid'] = False
                result['threats_detected'].append('sql_injection')
                result['risk_level'] = 'high'
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                result['valid'] = False
                result['threats_detected'].append('xss_attempt')
                result['risk_level'] = 'high'
        
        # Path traversal detection
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                result['valid'] = False
                result['threats_detected'].append('path_traversal')
                result['risk_level'] = 'high'
        
        # Sanitize input if needed
        if not result['valid']:
            result['sanitized_input'] = self._sanitize_input(input_data)
        
        return result
    
    def validate_file_upload(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Validate file uploads for security"""
        result = {
            'valid': True,
            'threats_detected': [],
            'risk_level': 'low'
        }
        
        # File extension validation
        import os
        _, ext = os.path.splitext(filename.lower())
        if ext not in self.allowed_file_types:
            result['valid'] = False
            result['threats_detected'].append('invalid_file_type')
            result['risk_level'] = 'medium'
        
        # File size validation (10MB limit)
        if len(file_content) > 10 * 1024 * 1024:
            result['valid'] = False
            result['threats_detected'].append('file_too_large')
            result['risk_level'] = 'medium'
        
        return result
    
    def _sanitize_input(self, input_data: str) -> str:
        """Sanitize potentially dangerous input"""
        import re
        import html
        
        # HTML escape
        sanitized = html.escape(input_data)
        
        # Remove potentially dangerous patterns
        for pattern in self.sql_patterns + self.xss_patterns + self.path_traversal_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
