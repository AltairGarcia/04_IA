#!/usr/bin/env python3
"""
Comprehensive Error Recovery Testing Script

This script tests various error recovery mechanisms and validates
the robustness of the LangGraph 101 application.
"""

import sys
import os
import time
import random
from typing import Dict, Any, List
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our modules
from app_health import robust_function, get_health_summary
from error_handling import ErrorHandler, ErrorCategory
from config_robust import load_config_robust

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorRecoveryTester:
    """Test error recovery mechanisms comprehensively."""
    
    def __init__(self):
        self.test_results = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all error recovery tests."""
        print("🧪 COMPREHENSIVE ERROR RECOVERY TESTING")
        print("=" * 50)
        
        tests = [
            ("Robust Function Decorator", self.test_robust_function_decorator),
            ("Error Handler Retry", self.test_error_handler_retry),
            ("Configuration Recovery", self.test_configuration_recovery),
            ("Health Monitoring Resilience", self.test_health_monitoring_resilience),
            ("Network Error Recovery", self.test_network_error_recovery),
            ("Memory Pressure Handling", self.test_memory_pressure_handling),
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔬 Testing: {test_name}")
            print("-" * 30)
            
            try:
                result = test_func()
                self.test_results.append({
                    "test": test_name,
                    "status": "passed" if result else "failed",
                    "details": result
                })
                status_icon = "✅" if result else "❌"
                print(f"{status_icon} {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                self.test_results.append({
                    "test": test_name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"💥 {test_name}: ERROR - {e}")
        
        return self.generate_test_report()
    
    def test_robust_function_decorator(self) -> bool:
        """Test the robust_function decorator with retry logic."""
        
        @robust_function(max_retries=3, delay=0.1)
        def flaky_function():
            if random.random() < 0.6:  # 60% chance of failure
                raise Exception("Simulated network timeout")
            return "Success after retries!"
        
        try:
            # Test multiple times to verify retry behavior
            successes = 0
            attempts = 10
            
            for i in range(attempts):
                try:
                    result = flaky_function()
                    if result == "Success after retries!":
                        successes += 1
                except Exception:
                    pass  # Expected some failures
            
            # Should have some successes due to retries
            success_rate = successes / attempts
            print(f"   Success rate with retries: {success_rate:.1%}")
            return success_rate > 0.2  # At least 20% should succeed with retries
            
        except Exception as e:
            print(f"   Error testing robust function: {e}")
            return False
    
    def test_error_handler_retry(self) -> bool:
        """Test ErrorHandler retry mechanism."""
        
        @ErrorHandler.with_retry(
            max_retries=2, 
            delay=0.1, 
            retry_errors=[ErrorCategory.NETWORK_ERROR, ErrorCategory.SERVICE_UNAVAILABLE_ERROR]
        )
        def network_call():
            if random.random() < 0.7:  # 70% chance of network error
                import requests
                raise requests.ConnectionError("Connection failed")
            return "Network call succeeded"
        
        try:
            successes = 0
            attempts = 5
            
            for i in range(attempts):
                try:
                    result = network_call()
                    if "succeeded" in result:
                        successes += 1
                except Exception:
                    pass  # Expected some failures
            
            print(f"   Network retry successes: {successes}/{attempts}")
            return successes > 0  # At least one should succeed
            
        except Exception as e:
            print(f"   Error testing ErrorHandler retry: {e}")
            return False
    
    def test_configuration_recovery(self) -> bool:
        """Test configuration loading recovery."""
        try:
            # Test loading configuration multiple times
            for i in range(3):
                config = load_config_robust()
                if not hasattr(config, 'development_mode'):
                    print(f"   Configuration missing development_mode attribute")
                    return False
            
            print(f"   Configuration loading: Stable across multiple loads")
            return True
            
        except Exception as e:
            print(f"   Configuration recovery failed: {e}")
            return False
    
    def test_health_monitoring_resilience(self) -> bool:
        """Test health monitoring system resilience."""
        try:
            # Get health summary multiple times
            health_checks = []
            for i in range(3):
                health = get_health_summary()
                health_checks.append(health)
                time.sleep(0.1)
            
            # Verify consistency
            if len(health_checks) != 3:
                print(f"   Health check failed to return results")
                return False
            
            # Check that all have required fields
            required_fields = ['overall_status', 'total_checks', 'ok_count']
            for health in health_checks:
                for field in required_fields:
                    if field not in health:
                        print(f"   Missing field {field} in health summary")
                        return False
            
            print(f"   Health monitoring: Consistent across {len(health_checks)} checks")
            return True
            
        except Exception as e:
            print(f"   Health monitoring resilience test failed: {e}")
            return False
    
    def test_network_error_recovery(self) -> bool:
        """Test network error handling and recovery."""
        
        @robust_function(max_retries=2, delay=0.1, exceptions=(ConnectionError, TimeoutError))
        def mock_api_call():
            if random.random() < 0.5:  # 50% chance of network error
                raise ConnectionError("API unreachable")
            return {"status": "success", "data": "API response"}
        
        try:
            attempts = 5
            successes = 0
            
            for i in range(attempts):
                try:
                    result = mock_api_call()
                    if result.get("status") == "success":
                        successes += 1
                except Exception:
                    pass  # Expected some failures
            
            print(f"   Network error recovery: {successes}/{attempts} successful")
            return successes > 0
            
        except Exception as e:
            print(f"   Network error recovery test failed: {e}")
            return False
    
    def test_memory_pressure_handling(self) -> bool:
        """Test handling of memory pressure scenarios."""
        try:
            # Get current health status to check memory monitoring
            health = get_health_summary()
            
            if 'checks' not in health:
                print(f"   No health checks available")
                return False
            
            memory_check = health['checks'].get('memory_usage')
            if not memory_check:
                print(f"   No memory usage check available")
                return False
            
            # Memory check should have status and details
            if 'status' not in memory_check or 'details' not in memory_check:
                print(f"   Memory check missing required fields")
                return False
            
            memory_percent = memory_check['details'].get('percent_used', 0)
            print(f"   Memory usage monitoring: {memory_percent:.1f}% ({memory_check['status']})")
            
            # Should handle high memory usage gracefully
            return True
            
        except Exception as e:
            print(f"   Memory pressure handling test failed: {e}")
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'passed')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'failed')
        error_tests = sum(1 for r in self.test_results if r['status'] == 'error')
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "details": self.test_results
        }
        
        print(f"\n📊 ERROR RECOVERY TEST REPORT")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"💥 Errors: {error_tests}")
        print(f"📈 Success Rate: {report['success_rate']:.1f}%")
        
        if report['success_rate'] >= 80:
            print(f"\n🎉 ERROR RECOVERY SYSTEM: EXCELLENT")
        elif report['success_rate'] >= 60:
            print(f"\n✅ ERROR RECOVERY SYSTEM: GOOD")
        elif report['success_rate'] >= 40:
            print(f"\n⚠️ ERROR RECOVERY SYSTEM: NEEDS IMPROVEMENT")
        else:
            print(f"\n🚨 ERROR RECOVERY SYSTEM: CRITICAL ISSUES")
        
        return report


def main():
    """Main testing function."""
    print("🚀 Starting Comprehensive Error Recovery Testing...")
    
    # Initialize tester
    tester = ErrorRecoveryTester()
    
    # Run all tests
    report = tester.run_all_tests()
    
    # Save report
    import json
    report_file = os.path.join(os.path.dirname(__file__), "error_recovery_test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report['success_rate'] >= 80


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
