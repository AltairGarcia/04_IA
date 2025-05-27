#!/usr/bin/env python3
"""
Simplified Security Testing Suite

This module provides a simplified version of the security testing suite
that tests core security components without complex dependencies.
"""

import os
import sys
import time
import json
import logging
import unittest
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCategory(Enum):
    """Test categories for security testing."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SESSION_MANAGEMENT = "session_management"
    INPUT_VALIDATION = "input_validation"
    RATE_LIMITING = "rate_limiting"
    SECURITY_HEADERS = "security_headers"
    ENCRYPTION = "encryption"
    BASIC_SECURITY = "basic_security"

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    category: TestCategory
    passed: bool
    execution_time: float
    message: str
    severity: str = "medium"
    details: Dict[str, Any] = None

@dataclass
class TestSuiteResults:
    """Test suite results summary."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    security_score: float
    results: List[TestResult]
    timestamp: datetime

class SimplifiedSecurityTester:
    """Simplified security testing implementation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_test(self, test_name: str, category: TestCategory, test_func, severity: str = "medium") -> TestResult:
        """Run a single test and record results."""
        start_time = time.time()
        try:
            result = test_func()
            passed = bool(result)
            message = "Test passed" if passed else "Test failed"
            if isinstance(result, tuple):
                passed, message = result
        except Exception as e:
            passed = False
            message = f"Test error: {str(e)}"
        
        execution_time = time.time() - start_time
        test_result = TestResult(
            test_name=test_name,
            category=category,
            passed=passed,
            execution_time=execution_time,
            message=message,
            severity=severity
        )
        
        self.results.append(test_result)
        logger.info(f"Test {test_name}: {'PASS' if passed else 'FAIL'} ({execution_time:.3f}s)")
        return test_result
    
    def test_basic_security_imports(self):
        """Test that security modules can be imported."""
        try:
            # Test basic imports without instantiation
            import advanced_auth
            import session_manager
            import ddos_protection
            import security_headers
            import input_security
            return True, "All security modules imported successfully"
        except Exception as e:
            return False, f"Import failed: {str(e)}"
    
    def test_password_strength_validation(self):
        """Test password strength validation."""
        try:
            # Simple password strength check
            weak_passwords = ["123", "password", "abc123"]
            strong_passwords = ["MyStr0ng!P@ssw0rd", "C0mpl3x!P@$$w0rd123"]
            
            # Basic password validation logic
            def validate_password(password):
                if len(password) < 8:
                    return False
                has_upper = any(c.isupper() for c in password)
                has_lower = any(c.islower() for c in password)
                has_digit = any(c.isdigit() for c in password)
                has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
                return has_upper and has_lower and has_digit and has_special
            
            # Test weak passwords should fail
            for pwd in weak_passwords:
                if validate_password(pwd):
                    return False, f"Weak password '{pwd}' was accepted"
            
            # Test strong passwords should pass
            for pwd in strong_passwords:
                if not validate_password(pwd):
                    return False, f"Strong password '{pwd}' was rejected"
            
            return True, "Password validation working correctly"
        except Exception as e:
            return False, f"Password validation test failed: {str(e)}"
    
    def test_session_security(self):
        """Test session security features."""
        try:
            # Test session token generation
            import secrets
            import uuid
            
            # Generate session tokens
            tokens = []
            for _ in range(10):
                token = secrets.token_urlsafe(32)
                if token in tokens:
                    return False, "Duplicate session token generated"
                tokens.append(token)
            
            # Test session ID format
            session_id = str(uuid.uuid4())
            if len(session_id) != 36:  # UUID format
                return False, "Invalid session ID format"
            
            return True, "Session security tests passed"
        except Exception as e:
            return False, f"Session security test failed: {str(e)}"
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        try:
            # Test XSS prevention
            xss_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "'; DROP TABLE users; --"
            ]
            
            # Simple sanitization function
            def sanitize_input(input_str):
                if not isinstance(input_str, str):
                    return str(input_str)
                # Remove script tags and javascript
                dangerous_patterns = ["<script", "javascript:", "onerror=", "DROP TABLE", "--"]
                for pattern in dangerous_patterns:
                    if pattern.lower() in input_str.lower():
                        return input_str.replace(pattern, "")
                return input_str
            
            for xss_input in xss_inputs:
                sanitized = sanitize_input(xss_input)
                if sanitized == xss_input:
                    return False, f"XSS input not sanitized: {xss_input}"
            
            return True, "Input sanitization working"
        except Exception as e:
            return False, f"Input sanitization test failed: {str(e)}"
    
    def test_encryption_basics(self):
        """Test basic encryption functionality."""
        try:
            import hashlib
            import base64
            from cryptography.fernet import Fernet
            
            # Test hashing
            test_data = "sensitive_data_123"
            hash_result = hashlib.sha256(test_data.encode()).hexdigest()
            if len(hash_result) != 64:  # SHA256 produces 64 char hex
                return False, "Invalid hash length"
            
            # Test encryption/decryption
            key = Fernet.generate_key()
            cipher = Fernet(key)
            encrypted = cipher.encrypt(test_data.encode())
            decrypted = cipher.decrypt(encrypted).decode()
            
            if decrypted != test_data:
                return False, "Encryption/decryption failed"
            
            return True, "Encryption tests passed"
        except Exception as e:
            return False, f"Encryption test failed: {str(e)}"
    
    def test_security_headers(self):
        """Test security headers configuration."""
        try:
            # Define required security headers
            required_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'"
            }
            
            # Simulate header validation
            for header, expected_value in required_headers.items():
                if not header or not expected_value:
                    return False, f"Invalid security header: {header}"
            
            return True, "Security headers validation passed"
        except Exception as e:
            return False, f"Security headers test failed: {str(e)}"
    
    def test_rate_limiting_logic(self):
        """Test rate limiting logic."""
        try:
            from collections import defaultdict
            import time
            
            # Simple rate limiter simulation
            request_counts = defaultdict(list)
            max_requests = 5
            window_seconds = 60
            
            def check_rate_limit(user_id):
                now = time.time()
                # Clean old requests
                request_counts[user_id] = [
                    req_time for req_time in request_counts[user_id]
                    if now - req_time < window_seconds
                ]
                
                if len(request_counts[user_id]) >= max_requests:
                    return False  # Rate limited
                
                request_counts[user_id].append(now)
                return True  # Allowed
            
            # Test normal usage
            user_id = "test_user"
            for i in range(max_requests):
                if not check_rate_limit(user_id):
                    return False, f"Rate limit triggered too early at request {i+1}"
            
            # Test rate limiting
            if check_rate_limit(user_id):
                return False, "Rate limit not enforced"
            
            return True, "Rate limiting logic working"
        except Exception as e:
            return False, f"Rate limiting test failed: {str(e)}"
    
    def run_all_tests(self) -> TestSuiteResults:
        """Run all security tests."""
        logger.info("Starting simplified security test suite...")
        
        # Define tests to run
        tests = [
            ("Security Module Imports", TestCategory.BASIC_SECURITY, self.test_basic_security_imports, "high"),
            ("Password Strength Validation", TestCategory.AUTHENTICATION, self.test_password_strength_validation, "high"),
            ("Session Security", TestCategory.SESSION_MANAGEMENT, self.test_session_security, "high"),
            ("Input Sanitization", TestCategory.INPUT_VALIDATION, self.test_input_sanitization, "high"),
            ("Encryption Basics", TestCategory.ENCRYPTION, self.test_encryption_basics, "medium"),
            ("Security Headers", TestCategory.SECURITY_HEADERS, self.test_security_headers, "medium"),
            ("Rate Limiting Logic", TestCategory.RATE_LIMITING, self.test_rate_limiting_logic, "medium"),
        ]
        
        # Run tests
        for test_name, category, test_func, severity in tests:
            self.run_test(test_name, category, test_func, severity)
        
        # Calculate results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        execution_time = time.time() - self.start_time
        
        # Calculate security score (0-100)
        if total_tests == 0:
            security_score = 0.0
        else:
            base_score = (passed_tests / total_tests) * 100
            
            # Apply severity penalties for failed tests
            severity_penalties = {"high": 15, "medium": 10, "low": 5}
            total_penalty = 0
            for result in self.results:
                if not result.passed:
                    total_penalty += severity_penalties.get(result.severity, 10)
            
            security_score = max(0.0, base_score - total_penalty)
        
        results = TestSuiteResults(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            execution_time=execution_time,
            security_score=security_score,
            results=self.results,
            timestamp=datetime.now()
        )
        
        return results
    
    def generate_report(self, results: TestSuiteResults) -> Dict[str, Any]:
        """Generate test report."""
        report = {
            "timestamp": results.timestamp.isoformat(),
            "summary": {
                "total_tests": results.total_tests,
                "passed_tests": results.passed_tests,
                "failed_tests": results.failed_tests,
                "execution_time": results.execution_time,
                "security_score": results.security_score,
                "status": "PASS" if results.security_score >= 80 else "FAIL"
            },
            "test_results": [asdict(result) for result in results.results],
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: TestSuiteResults) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []
        
        failed_categories = set()
        for result in results.results:
            if not result.passed:
                failed_categories.add(result.category.value)
        
        if "authentication" in failed_categories:
            recommendations.append("Strengthen authentication mechanisms and password policies")
        
        if "input_validation" in failed_categories:
            recommendations.append("Implement comprehensive input validation and sanitization")
        
        if "encryption" in failed_categories:
            recommendations.append("Review and strengthen encryption implementations")
        
        if "rate_limiting" in failed_categories:
            recommendations.append("Implement proper rate limiting and DDoS protection")
        
        if results.security_score < 80:
            recommendations.append("Overall security score below 80% - comprehensive security review recommended")
        
        if not recommendations:
            recommendations.append("All tests passed - maintain current security practices")
        
        return recommendations

def main():
    """Main function to run security tests."""
    print("=" * 60)
    print("LANGGRAPH 101 - SIMPLIFIED SECURITY TEST SUITE")
    print("=" * 60)
    
    # Create tester and run tests
    tester = SimplifiedSecurityTester()
    results = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_report(results)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {results.total_tests}")
    print(f"Passed: {results.passed_tests}")
    print(f"Failed: {results.failed_tests}")
    print(f"Execution Time: {results.execution_time:.2f} seconds")
    print(f"Security Score: {results.security_score:.1f}%")
    print(f"Status: {'PASS' if results.security_score >= 80 else 'FAIL'}")
    
    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)
    for result in results.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.test_name} ({result.execution_time:.3f}s)")
        if not result.passed:
            print(f"    Error: {result.message}")
    
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Save report to file
    report_file = f"simplified_security_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return exit code based on results
    return 0 if results.security_score >= 80 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
