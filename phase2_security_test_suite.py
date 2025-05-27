#!/usr/bin/env python3
"""
Phase 2 Advanced Security Test Suite
====================================

Comprehensive testing suite for all Phase 2 advanced security components.
Tests authentication, encryption, audit logging, intrusion detection, and more.

Usage:
    python phase2_security_test_suite.py
"""

import asyncio
import os
import sys
import tempfile
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from phase2_advanced_security import (
        AdvancedSecurityManager,
        JWTAuthenticationManager,
        EncryptionManager,
        SecurityAuditLogger,
        IntrusionDetectionSystem,
        SecureSessionManager,
        SecurityVulnerabilityScanner,
        SECURITY_CONFIG,
        SecurityUser,
        SecurityThreat
    )
    print("✅ Successfully imported all Phase 2 security components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class Phase2SecurityTestSuite:
    """Comprehensive test suite for Phase 2 security components."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp(prefix="security_test_")
        self.test_user = SecurityUser(
            user_id="test_user_123",
            username="testuser",
            email="test@example.com",
            password_hash="$2b$12$test_hash_for_testing_purposes_only",
            roles=["user", "tester"]
        )
        
    async def run_all_tests(self):
        """Run all security component tests."""
        print("\n🔐 Phase 2 Advanced Security Test Suite")
        print("=" * 50)
        
        tests = [
            ("Authentication Manager", self.test_authentication),
            ("Encryption Manager", self.test_encryption),
            ("Audit Logger", self.test_audit_logging),
            ("Intrusion Detection", self.test_intrusion_detection),
            ("Session Manager", self.test_session_management),
            ("Vulnerability Scanner", self.test_vulnerability_scanner),
            ("Advanced Security Manager", self.test_security_manager),
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            try:
                result = await test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAILED")
                self.test_results[test_name] = result
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results[test_name] = False
        
        # Print summary
        print(f"\n📊 Test Summary")
        print("=" * 30)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Phase 2 security system is ready for integration.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review and fix issues before proceeding.")
        
        return passed_tests == total_tests

    async def test_authentication(self):
        """Test JWT Authentication Manager."""
        try:
            auth_manager = JWTAuthenticationManager()
            
            # Test user creation
            user = await auth_manager.create_user("testuser", "test@example.com", "SecurePass123!")
            if not user:
                print("  ❌ User creation failed")
                return False
            
            # Test password verification
            is_valid = await auth_manager.verify_password("SecurePass123!", user.password_hash)
            if not is_valid:
                print("  ❌ Password verification failed")
                return False
            
            # Test JWT token creation
            token = await auth_manager.create_access_token(user)
            if not token:
                print("  ❌ JWT token generation failed")
                return False
            
            # Test JWT token verification
            verified_user = await auth_manager.verify_token(token)
            if not verified_user or verified_user.user_id != user.user_id:
                print("  ❌ JWT token verification failed")
                return False
            
            # Test refresh token generation
            refresh_token = await auth_manager.create_refresh_token(user)
            if not refresh_token:
                print("  ❌ Refresh token generation failed")
                return False
            
            print("  ✅ All authentication tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Authentication test error: {e}")
            return False
    
    async def test_encryption(self):
        """Test Encryption Manager."""
        try:
            encryption_manager = EncryptionManager()
            
            # Test data encryption
            test_data = "This is sensitive test data that needs encryption"
            # Use encrypt instead of encrypt_data
            encrypted_data = await encryption_manager.encrypt(test_data)
            if not encrypted_data or encrypted_data == test_data:
                print("  ❌ Data encryption failed")
                return False
            
            # Test data decryption
            # Use decrypt instead of decrypt_data
            decrypted_data = await encryption_manager.decrypt(encrypted_data)
            if decrypted_data != test_data:
                print("  ❌ Data decryption failed")
                return False
            
            # Test file encryption
            test_file_path = os.path.join(self.temp_dir, "test_file.txt")
            original_file_content = "Test file content for encryption"
            with open(test_file_path, "w") as f:
                f.write(original_file_content)
            
            # Encrypt file content
            encrypted_file_content = await encryption_manager.encrypt(original_file_content)
            encrypted_file_path = os.path.join(self.temp_dir, "test_file.encrypted.txt")
            with open(encrypted_file_path, "w") as f:
                f.write(encrypted_file_content)

            if not os.path.exists(encrypted_file_path):
                print("  ❌ Encrypted file not created")
                return False
            
            # Test file decryption
            # Read encrypted content and decrypt
            with open(encrypted_file_path, "r") as f:
                content_to_decrypt = f.read()
            
            decrypted_file_content = await encryption_manager.decrypt(content_to_decrypt)
            decrypted_file_path = os.path.join(self.temp_dir, "test_file.decrypted.txt")
            with open(decrypted_file_path, "w") as f:
                f.write(decrypted_file_content)

            if not os.path.exists(decrypted_file_path):
                print("  ❌ Decrypted file not created")
                return False
            
            with open(decrypted_file_path, "r") as f:
                final_content = f.read()
            if final_content != original_file_content:
                print("  ❌ File decryption content mismatch")
                return False
            
            # Test key rotation
            old_key_id = encryption_manager.current_key_id
            # Assuming rotate_keys is synchronous as per phase2_advanced_security.py
            encryption_manager.rotate_keys()
            new_key_id = encryption_manager.current_key_id
            if new_key_id == old_key_id:
                print("  ❌ Key rotation failed")
                return False
            
            # Test encryption with new key
            test_data_new_key = "Data with new key"
            encrypted_new = await encryption_manager.encrypt(test_data_new_key)
            decrypted_new = await encryption_manager.decrypt(encrypted_new)
            if decrypted_new != test_data_new_key:
                print("  ❌ Encryption/decryption with new key failed")
                return False

            # Test decryption of old data with old key (still possible due to key_id prefix)
            decrypted_old_data_after_rotation = await encryption_manager.decrypt(encrypted_data)
            if decrypted_old_data_after_rotation != test_data:
                print("  ❌ Decryption of old data after key rotation failed")
                return False

            print("  ✅ All encryption tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Encryption test error: {e}")
            return False
    
    async def test_audit_logging(self):
        """Test Security Audit Logger."""
        try:
            audit_db = os.path.join(self.temp_dir, "audit_test.db")
            audit_logger = SecurityAuditLogger(audit_db)
            
            # Test audit log creation
            # Corrected method name from log_security_event to log_event
            # Changed 'details' to 'additional_data'
            await audit_logger.log_event(
                user_id="test_user",
                event_type="user_login",
                event_category="authentication", 
                description="User login attempt", 
                ip_address="192.168.1.100",
                user_agent="test-suite-agent", 
                success=True,
                risk_level="low", 
                additional_data={"ip_address": "192.168.1.100", "success": True} 
            )
            
            # Test audit log retrieval
            # Corrected: get_recent_events returns a list, not an async generator, and is not awaitable
            logs_list = audit_logger.get_recent_events(hours=1) 
            
            if not logs_list or len(logs_list) == 0:
                print("  ❌ Audit log retrieval failed")
                return False
            
            # Verify the logged event is present and correct
            logged_event = next((log for log in logs_list if log.get('event_type') == 'user_login' and log.get('user_id') == 'test_user'), None)
            if not logged_event:
                print("  ❌ Logged event not found in retrieved logs")
                return False
            if logged_event.get('description') != "User login attempt":
                print("  ❌ Logged event description mismatch")
                return False

            # Test user activity logs
            user_logs = [log for log in logs_list if log.get('user_id') == 'test_user']
            if not user_logs or len(user_logs) == 0:
                print("  ❌ User activity log retrieval failed (filtered from recent)")
                return False
            
            # Test security alerts
            await audit_logger.log_event(
                user_id="attacker",
                event_type="failed_login_attempt",
                event_category="authentication",
                description="Multiple failed login attempts",
                ip_address="10.0.0.1",
                user_agent="attacker-agent",
                success=False,
                risk_level="high",
                additional_data={"ip_address": "10.0.0.1", "attempts": 5}
            )
            
            # Corrected: get_recent_events returns a list, not an async generator, and is not awaitable
            all_recent_logs = audit_logger.get_recent_events(hours=1) 
            alerts_list = [log for log in all_recent_logs if log.get('risk_level') in ['high', 'critical']]
            
            if not alerts_list or not any(alert['event_type'] == 'failed_login_attempt' for alert in alerts_list):
                print("  ❌ Security alerts retrieval failed (filtered from recent)")
                return False
            
            print("  ✅ All audit logging tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Audit logging test error: {e}")
            return False
    
    async def test_intrusion_detection(self):
        """Test Intrusion Detection System."""
        try:
            ids = IntrusionDetectionSystem()

            # Mock FastAPI Request object
            class MockRequest:
                def __init__(self, client_host, query_params=None, headers=None, body=None, path_params=None, url_path="/"):
                    self.client = MockClient(client_host) # Ensure client has a host attribute
                    self.query_params = query_params or {}
                    self.headers = headers or {}
                    self._body = body
                    self.path_params = path_params or {}
                    self.url = MockURL(path_params.get("path", url_path) if path_params else url_path)


                async def body(self):
                    return self._body

            class MockClient: # Added MockClient to nest host attribute
                def __init__(self, host):
                    self.host = host

            class MockURL:
                def __init__(self, path):
                    self.path = path

                def __str__(self):
                    return self.path

            # Test SQL injection detection
            sql_attacks = [
                "SELECT * FROM users WHERE id = 1 OR 1=1",
                "'; DROP TABLE users; --",
                "admin' OR '1'='1",
            ]
            
            for attack in sql_attacks:
                # Simulate a request with query parameters for SQLi
                mock_request = MockRequest(client_host="192.168.1.100", query_params={"q": attack})
                threat = await ids.analyze_request(mock_request)
                if not threat or threat.threat_type != "sql_injection":
                    print(f"  ❌ SQL injection not detected: {attack}")
                    return False
            
            # Test XSS detection
            xss_attacks = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
            ]
            
            for attack in xss_attacks:
                # Simulate a request with a body for XSS
                mock_request = MockRequest(client_host="192.168.1.100", body=attack.encode(), url_path="/legit_path_for_xss_test") # Added a neutral url_path
                threat = await ids.analyze_request(mock_request)
                # Check if 'xss_attacks' is part of the threat_type string, to allow for multiple detections
                if not threat or "xss_attacks" not in threat.threat_type.split(', '):
                    print(f"  ❌ XSS attack not detected or not primary: {attack} (Threat: {threat.threat_type if threat else 'None'})")
                    return False
            
            # Test path traversal detection
            path_attacks = [
                "../../../etc/passwd",
                "..\\\\..\\\\..\\\\windows\\\\system32", # Adjusted for Windows path if necessary, though IDS logic might normalize
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            ]
            
            for attack in path_attacks:
                # Simulate a request with path parameters for path traversal
                # Ensure the MockRequest's url.path is correctly reflecting the attack string for _analyze_url
                mock_request = MockRequest(client_host="192.168.1.100", path_params={"path": attack}, url_path=attack) 
                threat = await ids.analyze_request(mock_request)
                if not threat or "path_traversal" not in threat.threat_type.split(', '):
                    print(f"  ❌ Path traversal not detected or not primary: {attack} (Threat: {threat.threat_type if threat else 'None'})")
                    return False
            
            # Test legitimate requests (should not trigger alerts)
            legitimate_requests_data = [
                {"type": "query", "data": {"q": "SELECT name FROM products WHERE category = 'electronics'"}},
                {"type": "body", "data": "Hello, this is a normal message"},
                {"type": "path", "data": "/api/v1/users/profile"},
            ]
            
            for req_data in legitimate_requests_data:
                if req_data["type"] == "query":
                    mock_request = MockRequest(client_host="127.0.0.1", query_params=req_data["data"])
                elif req_data["type"] == "body":
                    mock_request = MockRequest(client_host="127.0.0.1", body=req_data["data"].encode())
                else: # path
                    mock_request = MockRequest(client_host="127.0.0.1", path_params={"path": req_data["data"]})
                
                threat = await ids.analyze_request(mock_request)
                if threat:
                    print(f"  ❌ False positive detected: {req_data}")
                    return False
            
            print("  ✅ All intrusion detection tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Intrusion detection test error: {e}")
            return False
    
    async def test_session_management(self):
        """Test Secure Session Manager."""
        try:
            session_manager = SecureSessionManager()
              # Test session creation
            session = await session_manager.create_session(
                self.test_user.user_id,
                "192.168.1.100",
                "test-user-agent"
            )
            if not session:
                print("  ❌ Session creation failed")
                return False
              # Test session validation
            validated_session = await session_manager.validate_session(session.session_id)
            if not validated_session or validated_session.user_id != self.test_user.user_id:
                print("  ❌ Session validation failed")
                return False
            
            # Test CSRF token generation
            csrf_token = await session_manager.generate_csrf_token(session.session_id)
            if not csrf_token:
                print("  ❌ CSRF token generation failed")
                return False
            
            # Test CSRF token validation
            is_valid = await session_manager.validate_csrf_token(session.session_id, csrf_token)
            if not is_valid:
                print("  ❌ CSRF token validation failed")
                return False
            
            # Test session update
            success = await session_manager.update_session_activity(session.session_id)
            if not success:
                print("  ❌ Session update failed")
                return False
            
            # Test session termination
            success = await session_manager.terminate_session(session.session_id)
            if not success:
                print("  ❌ Session termination failed")
                return False
            
            # Verify session is terminated
            terminated_session = await session_manager.validate_session(session.session_id)
            if terminated_session:
                print("  ❌ Session not properly terminated")
                return False
            
            print("  ✅ All session management tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Session management test error: {e}")
            return False
    
    async def test_vulnerability_scanner(self):
        """Test Security Vulnerability Scanner."""
        try:
            scanner = SecurityVulnerabilityScanner()
            
            # Test file permission scanning
            test_file = os.path.join(self.temp_dir, "test_permissions.txt")
            with open(test_file, "w") as f:
                f.write("Test content")
            
            # Make file readable by all (potential vulnerability)
            os.chmod(test_file, 0o777)
            
            vulnerabilities = await scanner.scan_file_permissions(self.temp_dir)
            found_vuln = any(vuln.file_path == test_file for vuln in vulnerabilities)
            if not found_vuln:
                print("  ❌ File permission vulnerability not detected")
                return False
            
            # Test credential scanning
            test_file_with_creds = os.path.join(self.temp_dir, "config_with_creds.py")
            with open(test_file_with_creds, "w") as f:
                f.write("API_KEY = 'sk-1234567890abcdef'\nPASSWORD = 'admin123'\n")
            
            vulnerabilities = await scanner.scan_for_exposed_credentials(self.temp_dir)
            found_cred_vuln = any(vuln.file_path == test_file_with_creds for vuln in vulnerabilities)
            if not found_cred_vuln:
                print("  ❌ Credential vulnerability not detected")
                return False
            
            # Test dependency scanning
            vulnerabilities = await scanner.scan_dependencies()
            # This should return results (might be empty if no vulnerabilities found)
            if vulnerabilities is None:
                print("  ❌ Dependency scanning failed")
                return False
            
            print("  ✅ All vulnerability scanner tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Vulnerability scanner test error: {e}")
            return False
    
    async def test_security_manager(self):
        """Test Advanced Security Manager (integration test)."""
        try:
            security_manager = AdvancedSecurityManager()
            
            # Test initialization
            await security_manager.initialize()
            
            # Test user registration through security manager
            success = await security_manager.register_user("integration_test", "test@integration.com", "SecurePass123!")
            if not success:
                print("  ❌ Security manager user registration failed")
                return False
            
            # Test user authentication through security manager
            user = await security_manager.authenticate_user("integration_test", "SecurePass123!")
            if not user:
                print("  ❌ Security manager authentication failed")
                return False
            
            # Test security event logging
            await security_manager.log_security_event(
                "test_event", 
                user.user_id, 
                {"test": "integration"}
            )
            
            # Test threat analysis
            threat = await security_manager.analyze_threat({
                "query": "'; DROP TABLE users; --",
                "ip_address": "192.168.1.100"
            })
            if not threat:
                print("  ❌ Security manager threat analysis failed")
                return False
            
            # Test security status
            status = await security_manager.get_security_status()
            if not status:
                print("  ❌ Security manager status retrieval failed")
                return False
            
            print("  ✅ All security manager integration tests passed")
            return True
            
        except Exception as e:
            print(f"  ❌ Security manager test error: {e}")
            return False
    
    def cleanup(self):
        """Clean up test files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Failed to clean up test directory: {e}")

async def main():
    """Main test execution function."""
    test_suite = Phase2SecurityTestSuite()
    
    try:
        success = await test_suite.run_all_tests()
        
        # Generate test report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"phase2_security_test_report_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "total_tests": len(test_suite.test_results),
            "passed_tests": sum(test_suite.test_results.values()),
            "success_rate": f"{(sum(test_suite.test_results.values())/len(test_suite.test_results))*100:.1f}%",
            "results": test_suite.test_results
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        if success:
            print("\n🚀 Phase 2 Advanced Security System is ready for production integration!")
            return 0
        else:
            print("\n🔧 Some tests failed. Please review and fix issues before proceeding.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        return 1
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
