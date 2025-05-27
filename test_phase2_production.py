"""
Phase 2 Production Features Test Suite
Tests authentication, rate limiting, input validation, and monitoring features.
"""

import os
import sys
import time
import requests
import json
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_application_startup():
    """Test that the application starts correctly and responds to health checks."""
    try:
        # Test main application endpoint
        response = requests.get("http://localhost:8501", timeout=10)
        print(f"✓ Application startup: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Application startup failed: {str(e)}")
        return False

def test_health_endpoints():
    """Test health monitoring endpoints."""
    try:
        # Test health server endpoint
        response = requests.get("http://localhost:8502/health", timeout=5)
        print(f"✓ Health endpoint: Status {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"  - Overall status: {health_data.get('overall_status', 'unknown')}")
            return True
        return False
    except Exception as e:
        print(f"✗ Health endpoint test failed: {str(e)}")
        return False

def test_authentication_integration():
    """Test authentication system integration."""
    try:
        from auth_middleware import AuthManager
        
        # Test authentication manager initialization
        auth_manager = AuthManager()
        print("✓ Authentication manager initialized")
        
        # Test password hashing
        password = "test123"
        hashed = auth_manager.hash_password(password)
        verified = auth_manager.verify_password(password, hashed)
        
        if verified:
            print("✓ Password hashing and verification working")
            return True
        else:
            print("✗ Password verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Authentication integration test failed: {str(e)}")
        return False

def test_production_features():
    """Test production features like rate limiting and input validation."""
    try:
        from production_features import RateLimiter, InputValidator, SecurityManager
        
        # Test rate limiter
        rate_limiter = RateLimiter()
        user_id = "test_user"
        
        # Should not be rate limited initially
        if not rate_limiter.is_rate_limited(user_id):
            print("✓ Rate limiter initialization working")
        else:
            print("✗ Rate limiter incorrectly showing as limited")
            return False
        
        # Test input validator
        test_input = "Hello, this is a test message!"
        validation_result = InputValidator.validate_input(test_input, 'message')
        
        if validation_result['is_valid']:
            print("✓ Input validation working for valid input")
        else:
            print("✗ Input validation rejecting valid input")
            return False
        
        # Test malicious input detection
        malicious_input = "<script>alert('xss')</script>"
        malicious_result = InputValidator.validate_input(malicious_input, 'message')
        
        if not malicious_result['is_valid']:
            print("✓ Input validation blocking malicious input")
        else:
            print("✗ Input validation allowing malicious input")
            return False
        
        # Test security manager
        security_manager = SecurityManager()
        print("✓ Security manager initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Production features test failed: {str(e)}")
        return False

def test_monitoring_dashboard():
    """Test monitoring dashboard functionality."""
    try:
        from monitoring_dashboard import MonitoringDashboard
        
        # Test dashboard initialization
        dashboard = MonitoringDashboard()
        print("✓ Monitoring dashboard initialized")
          # Test metrics collection
        metrics = dashboard.collect_system_metrics()
        if metrics and 'cpu_usage' in metrics:
            print(f"✓ System metrics collection working (CPU: {metrics['cpu_usage']:.1f}%)")
            return True
        else:
            print("✗ System metrics collection failed")
            return False
            
    except Exception as e:
        print(f"✗ Monitoring dashboard test failed: {str(e)}")
        return False

def test_database_connections():
    """Test database connectivity for monitoring and security features."""
    try:
        import sqlite3
        
        # Test monitoring database
        monitoring_db = "monitoring.db"
        conn = sqlite3.connect(monitoring_db)
        cursor = conn.cursor()
        
        # Check if metrics table exists or create it
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL
            )
        """)
        
        # Test insert
        cursor.execute("INSERT INTO metrics (metric_name, metric_value) VALUES (?, ?)", 
                      ("test_metric", 100.0))
        conn.commit()
        
        # Test select
        cursor.execute("SELECT COUNT(*) FROM metrics WHERE metric_name = 'test_metric'")
        count = cursor.fetchone()[0]
        
        conn.close()
        
        if count > 0:
            print("✓ Database connectivity working")
            return True
        else:
            print("✗ Database test failed")
            return False
            
    except Exception as e:
        print(f"✗ Database connectivity test failed: {str(e)}")
        return False

def test_configuration_loading():
    """Test robust configuration loading."""
    try:
        from config_robust import load_config_robust
        
        config = load_config_robust()
        if config:
            print("✓ Robust configuration loading working")
            return True
        else:
            print("✗ Configuration loading failed")
            return False
            
    except Exception as e:
        print(f"✗ Configuration loading test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling and graceful degradation."""
    try:
        from error_handling import ErrorHandler
        from error_integration import GlobalErrorHandler
        
        # Test error handler initialization
        error_handler = ErrorHandler()
        global_handler = GlobalErrorHandler()
        
        print("✓ Error handling systems initialized")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        return False

def run_phase2_tests():
    """Run all Phase 2 production readiness tests."""
    print("=" * 60)
    print("PHASE 2 PRODUCTION READINESS TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    tests = [
        ("Application Startup", test_application_startup),
        ("Health Endpoints", test_health_endpoints),
        ("Authentication Integration", test_authentication_integration),
        ("Production Features", test_production_features),
        ("Monitoring Dashboard", test_monitoring_dashboard),
        ("Database Connections", test_database_connections),
        ("Configuration Loading", test_configuration_loading),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            print("")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            print("")
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Phase 2 production features ready!")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed - Phase 2 mostly ready with minor issues")
    else:
        print("❌ Multiple test failures - Phase 2 needs attention")
    
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 2 - Production Readiness",
        "total_tests": total,
        "passed_tests": passed,
        "success_rate": (passed/total)*100,
        "status": "PASSED" if passed == total else "PARTIAL" if passed >= total * 0.8 else "FAILED"
    }
    
    # Save report
    with open("phase2_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: phase2_test_report.json")
    
    return passed == total

if __name__ == "__main__":
    success = run_phase2_tests()
    sys.exit(0 if success else 1)
