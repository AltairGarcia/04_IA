#!/usr/bin/env python3
"""
End-to-End Testing Script for LangGraph 101 Streamlit Application
Tests all integrated systems working together with comprehensive validation.
"""

import asyncio
import json
import logging
import requests
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('end_to_end_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EndToEndTester:
    """Comprehensive end-to-end testing for the LangGraph 101 application."""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test_result(self, test_name: str, passed: bool, details: str = "", duration: float = 0):
        """Log test result with structured data."""
        result = {
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()        }
        self.test_results.append(result)
        status = "PASSED" if passed else "FAILED"
        logger.info(f"[{status}]: {test_name} ({duration:.2f}s)")
        if details:
            logger.info(f"   Details: {details}")
    
    def test_application_availability(self) -> bool:
        """Test if the main application is accessible."""
        start_time = time.time()
        try:
            response = requests.get(self.base_url, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test_result(
                    "Application Availability",
                    True,
                    f"Status code: {response.status_code}, Response size: {len(response.content)} bytes",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "Application Availability",
                    False,
                    f"Unexpected status code: {response.status_code}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Application Availability",
                False,
                f"Connection error: {str(e)}",
                duration
            )
            return False
    
    def test_health_monitoring_system(self) -> bool:
        """Test the health monitoring system."""
        start_time = time.time()
        try:
            # Test health check endpoint
            response = requests.get("http://localhost:8502/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health check structure
                required_fields = ['status', 'checks', 'timestamp']
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if not missing_fields:
                    checks_count = len(health_data.get('checks', {}))
                    self.log_test_result(
                        "Health Monitoring System",
                        True,
                        f"Health endpoint responding with {checks_count} checks",
                        duration
                    )
                    return True
                else:
                    self.log_test_result(
                        "Health Monitoring System",
                        False,
                        f"Missing required fields: {missing_fields}",
                        duration
                    )
                    return False
            else:
                self.log_test_result(
                    "Health Monitoring System",
                    False,
                    f"Health endpoint returned status: {response.status_code}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(                "Health Monitoring System",
                False,
                f"Health check failed: {str(e)}",
                duration
            )
            return False
    
    def test_configuration_loading(self) -> bool:
        """Test configuration loading and validation."""
        start_time = time.time()
        try:
            # Import and test configuration
            sys.path.append(str(Path(__file__).parent))
            from config_robust import load_config_robust
            
            config = load_config_robust()
            
            # Validate configuration attributes
            required_attrs = ['development_mode', 'debug', 'log_level']
            missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
            
            duration = time.time() - start_time
            
            if not missing_attrs:
                env = getattr(config, 'environment', 'development')
                self.log_test_result(
                    "Configuration Loading",
                    True,
                    f"Config loaded successfully in {env} mode",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "Configuration Loading",
                    False,
                    f"Missing configuration attributes: {missing_attrs}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(                "Configuration Loading",
                False,
                f"Configuration loading failed: {str(e)}",
                duration
            )
            return False
    
    def test_error_recovery_system(self) -> bool:
        """Test error recovery mechanisms."""
        start_time = time.time()
        try:
            # Import error handling components
            from error_handling import ErrorHandler
            from app_health import robust_function
            
            # Test robust function decorator
            @robust_function(max_retries=2, delay=0.1)
            def test_function():
                if not hasattr(test_function, 'attempt_count'):
                    test_function.attempt_count = 0
                test_function.attempt_count += 1
                
                if test_function.attempt_count < 2:
                    raise ValueError("Test error for retry mechanism")
                return "Success after retry"
            
            result = test_function()
            
            # Test ErrorHandler with retry - using direct call instead of decorator
            error_handler = ErrorHandler()
            
            call_count = 0
            def failing_operation():
                nonlocal call_count
                call_count += 1
                
                if call_count < 2:
                    raise ConnectionError("Test connection error")
                return "Recovery successful"
            
            # Use the retry decorator as a function wrapper
            retry_decorator = ErrorHandler.with_retry(max_retries=2, delay=0.1)
            wrapped_function = retry_decorator(failing_operation)
            recovery_result = wrapped_function()
            
            duration = time.time() - start_time
            
            if result == "Success after retry" and recovery_result == "Recovery successful":
                self.log_test_result(
                    "Error Recovery System",
                    True,
                    "Both robust_function and ErrorHandler.with_retry working correctly",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "Error Recovery System",
                    False,
                    f"Unexpected results: {result}, {recovery_result}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Error Recovery System",
                False,
                f"Error recovery testing failed: {str(e)}",
                duration
            )
            return False
    
    def test_system_initialization(self) -> bool:
        """Test system initialization components."""
        start_time = time.time()
        try:
            # Import system initialization
            from system_initialization import SystemInitializer
            
            initializer = SystemInitializer()
            
            # Test initialization status
            init_status = initializer.get_initialization_status()
            
            duration = time.time() - start_time
            
            if init_status and init_status.get('status') == 'initialized':
                self.log_test_result(
                    "System Initialization",
                    True,
                    f"System initialized with {len(init_status.get('components', {}))} components",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "System Initialization",
                    False,
                    f"System not properly initialized: {init_status}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "System Initialization",
                False,
                f"System initialization test failed: {str(e)}",
                duration
            )
            return False
    
    def test_deployment_readiness(self) -> bool:
        """Test deployment readiness checks."""
        start_time = time.time()
        try:
            # Import deployment readiness
            from deployment_readiness import DeploymentChecker
            
            checker = DeploymentChecker()
            readiness_result = checker.check_deployment_readiness()
            
            duration = time.time() - start_time
            
            # Check if deployment readiness passes with warnings (acceptable for development)
            if readiness_result and readiness_result.get('overall_status') in ['READY', 'WARNING']:
                passed_checks = len([c for c in readiness_result.get('checks', []) if c.get('status') == 'PASS'])
                total_checks = len(readiness_result.get('checks', []))
                
                self.log_test_result(
                    "Deployment Readiness",
                    True,
                    f"Status: {readiness_result.get('overall_status')}, Passed: {passed_checks}/{total_checks}",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "Deployment Readiness",
                    False,
                    f"Deployment not ready: {readiness_result.get('overall_status', 'UNKNOWN')}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Deployment Readiness",
                False,
                f"Deployment readiness test failed: {str(e)}",
                duration
            )
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance monitoring and metrics collection."""
        start_time = time.time()
        try:
            # Test multiple requests to check response times
            response_times = []
            
            for i in range(5):
                req_start = time.time()
                response = requests.get(self.base_url, timeout=5)
                req_duration = time.time() - req_start
                response_times.append(req_duration)
                
                if response.status_code != 200:
                    raise Exception(f"Request {i+1} failed with status {response.status_code}")
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            duration = time.time() - start_time
            
            # Performance thresholds (reasonable for development)
            if avg_response_time < 5.0 and max_response_time < 10.0:
                self.log_test_result(
                    "Performance Metrics",
                    True,
                    f"Avg response: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "Performance Metrics",
                    False,
                    f"Performance degraded - Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Metrics",
                False,
                f"Performance testing failed: {str(e)}",
                duration
            )
            return False
    
    def test_streamlit_features(self) -> bool:
        """Test Streamlit-specific features and UI components."""
        start_time = time.time()
        try:
            # Test main page content
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Check for key Streamlit components
            required_elements = [
                'streamlit',  # Streamlit framework
                'LangGraph',  # Application title/content
                'health',     # Health dashboard
                'status'      # Status indicators
            ]
            
            missing_elements = []
            for element in required_elements:
                if element.lower() not in content.lower():
                    missing_elements.append(element)
            
            duration = time.time() - start_time
            
            if not missing_elements:
                self.log_test_result(
                    "Streamlit Features",
                    True,
                    f"All key elements found, Content size: {len(content)} chars",
                    duration
                )
                return True
            else:
                self.log_test_result(
                    "Streamlit Features",
                    False,
                    f"Missing elements: {missing_elements}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Streamlit Features",
                False,
                f"Streamlit features test failed: {str(e)}",
                duration
            )
            return False
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        passed_tests = [r for r in self.test_results if r['passed']]
        failed_tests = [r for r in self.test_results if not r['passed']]
        
        success_rate = len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0
        
        report = {
            "test_session": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": total_duration,
                "tests_run": len(self.test_results)
            },
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": success_rate,
                "overall_status": "PASSED" if success_rate >= 80 else "FAILED"
            },
            "test_results": self.test_results,
            "failed_tests": failed_tests,
            "recommendations": self.generate_recommendations(failed_tests)
        }
        
        return report
    
    def generate_recommendations(self, failed_tests: List[Dict]) -> List[str]:
        """Generate recommendations based on failed tests."""
        recommendations = []
        
        if not failed_tests:
            recommendations.append("All tests passed! System is functioning correctly.")
            recommendations.append("Consider running performance optimization for production deployment.")
            return recommendations
        
        for test in failed_tests:
            test_name = test['test_name']
            
            if "Application Availability" in test_name:
                recommendations.append("Check if Streamlit application is running on port 8501")
                recommendations.append("Verify network connectivity and firewall settings")
            
            elif "Health Monitoring" in test_name:
                recommendations.append("Review health monitoring system configuration")
                recommendations.append("Check app_health.py for proper endpoint setup")
            
            elif "Configuration" in test_name:
                recommendations.append("Verify config_robust.py and .env file setup")
                recommendations.append("Check environment variable configuration")
            
            elif "Error Recovery" in test_name:
                recommendations.append("Review error_handling.py implementation")
                recommendations.append("Test retry mechanisms and error categorization")
            elif "Performance" in test_name:
                recommendations.append("Optimize application performance and resource usage")
                recommendations.append("Consider increasing server resources or optimizing code")
        
        return recommendations
    
    async def run_all_tests(self) -> Dict:
        """Run all end-to-end tests."""
        logger.info("Starting End-to-End Testing for LangGraph 101 Application")
        logger.info("=" * 70)
        
        # Define test sequence
        tests = [
            ("Application Availability", self.test_application_availability),
            ("Health Monitoring System", self.test_health_monitoring_system),
            ("Configuration Loading", self.test_configuration_loading),
            ("Error Recovery System", self.test_error_recovery_system),
            ("System Initialization", self.test_system_initialization),
            ("Deployment Readiness", self.test_deployment_readiness),
            ("Performance Metrics", self.test_performance_metrics),
            ("Streamlit Features", self.test_streamlit_features)
        ]
        
        # Run tests sequentially
        for test_name, test_func in tests:
            logger.info(f"\nRunning: {test_name}")
            try:
                test_func()
            except Exception as e:
                self.log_test_result(test_name, False, f"Test execution error: {str(e)}")
            
            # Brief pause between tests
            await asyncio.sleep(0.5)
        
        # Generate final report
        report = self.generate_test_report()
        
        # Save report
        report_file = Path('end_to_end_test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("END-TO-END TESTING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Tests: {report['summary']['total_tests']}")
        logger.info(f"Passed: {report['summary']['passed']}")
        logger.info(f"Failed: {report['summary']['failed']}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        logger.info(f"Overall Status: {report['summary']['overall_status']}")
        logger.info(f"Duration: {report['test_session']['total_duration']:.2f} seconds")
        
        if report['recommendations']:
            logger.info("\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"{i}. {rec}")
        
        logger.info(f"\nDetailed report saved to: {report_file.absolute()}")
        
        return report


async def main():
    """Main function to run end-to-end testing."""
    tester = EndToEndTester()
    report = await tester.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if report['summary']['overall_status'] == 'PASSED' else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
