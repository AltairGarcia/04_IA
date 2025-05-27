#!/usr/bin/env python3
"""
Simplified Performance Impact Assessment

This module provides a simplified performance assessment for security enhancements
without complex dependencies that may have import issues.
"""

import os
import sys
import time
import json
import psutil
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import tracemalloc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    execution_time: float
    operations_per_second: float
    test_name: str

@dataclass
class PerformanceTestResult:
    """Performance test result."""
    test_name: str
    baseline_time: float
    security_enhanced_time: float
    overhead_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    status: str  # PASS/FAIL
    threshold_ms: float
    details: Dict[str, Any]

class PerformanceMonitor:
    """Simple performance monitoring."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                metric = PerformanceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_mb=memory.used / (1024 * 1024),
                    execution_time=0.0,
                    operations_per_second=0.0,
                    test_name="background_monitoring"
                )
                
                self.metrics.append(metric)
                time.sleep(0.5)  # Monitor every 500ms
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average performance metrics."""
        if not self.metrics:
            return {}
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in self.metrics) / len(self.metrics),
            "avg_memory_percent": sum(m.memory_percent for m in self.metrics) / len(self.metrics),
            "avg_memory_mb": sum(m.memory_mb for m in self.metrics) / len(self.metrics),
            "max_cpu_percent": max(m.cpu_percent for m in self.metrics),
            "max_memory_mb": max(m.memory_mb for m in self.metrics)
        }

class SimplifiedPerformanceAssessment:
    """Simplified performance assessment implementation."""
    
    def __init__(self):
        self.results: List[PerformanceTestResult] = []
        self.monitor = PerformanceMonitor()
        
        # Performance thresholds (in milliseconds)
        self.thresholds = {
            "authentication": 1000.0,      # 1 second
            "session_validation": 200.0,   # 200ms
            "rate_limiting": 50.0,          # 50ms
            "input_validation": 100.0,      # 100ms
            "encryption": 500.0,            # 500ms
            "security_headers": 10.0,       # 10ms
            "password_hashing": 2000.0      # 2 seconds
        }
    
    def measure_execution_time(self, func, *args, **kwargs) -> tuple:
        """Measure function execution time and resource usage."""
        # Start memory tracking
        tracemalloc.start()
        
        # Get initial memory and CPU
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        end_time = time.time()
        
        # Get final memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_used = final_memory - initial_memory
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        
        return {
            "result": result,
            "success": success,
            "error": error,
            "execution_time_ms": execution_time,
            "memory_used_mb": memory_used,
            "peak_memory_mb": peak / (1024 * 1024)
        }
    
    def test_authentication_performance(self) -> PerformanceTestResult:
        """Test authentication performance."""
        def baseline_auth():
            # Simulate basic authentication
            import hashlib
            username = "test_user"
            password = "test_password"
            return hashlib.sha256(f"{username}:{password}".encode()).hexdigest()
        
        def enhanced_auth():
            # Simulate enhanced authentication with more security
            import hashlib
            import secrets
            username = "test_user"
            password = "test_password"
            salt = secrets.token_hex(16)
            # Multiple rounds of hashing for security
            hashed = password
            for _ in range(1000):  # Simulate PBKDF2-like iterations
                hashed = hashlib.sha256(f"{hashed}:{salt}".encode()).hexdigest()
            return hashed
        
        # Measure baseline
        baseline_result = self.measure_execution_time(baseline_auth)
        
        # Measure enhanced
        enhanced_result = self.measure_execution_time(enhanced_auth)
        
        overhead_percent = ((enhanced_result["execution_time_ms"] - baseline_result["execution_time_ms"]) 
                           / baseline_result["execution_time_ms"]) * 100
        
        status = "PASS" if enhanced_result["execution_time_ms"] <= self.thresholds["authentication"] else "FAIL"
        
        return PerformanceTestResult(
            test_name="Authentication Performance",
            baseline_time=baseline_result["execution_time_ms"],
            security_enhanced_time=enhanced_result["execution_time_ms"],
            overhead_percent=overhead_percent,
            memory_usage_mb=enhanced_result["memory_used_mb"],
            cpu_usage_percent=0.0,  # Would need more complex measurement
            status=status,
            threshold_ms=self.thresholds["authentication"],
            details={
                "baseline_success": baseline_result["success"],
                "enhanced_success": enhanced_result["success"],
                "baseline_error": baseline_result.get("error"),
                "enhanced_error": enhanced_result.get("error")
            }
        )
    
    def test_session_performance(self) -> PerformanceTestResult:
        """Test session management performance."""
        def baseline_session():
            # Simple session validation
            import uuid
            session_id = str(uuid.uuid4())
            return len(session_id) == 36
        
        def enhanced_session():
            # Enhanced session with security checks
            import uuid
            import secrets
            import hashlib
            
            session_id = str(uuid.uuid4())
            # Add security token
            security_token = secrets.token_urlsafe(32)
            # Validate format
            session_valid = len(session_id) == 36
            token_valid = len(security_token) >= 32
            # Add hash verification
            verification_hash = hashlib.sha256(f"{session_id}:{security_token}".encode()).hexdigest()
            
            return session_valid and token_valid and len(verification_hash) == 64
        
        baseline_result = self.measure_execution_time(baseline_session)
        enhanced_result = self.measure_execution_time(enhanced_session)
        
        overhead_percent = ((enhanced_result["execution_time_ms"] - baseline_result["execution_time_ms"]) 
                           / baseline_result["execution_time_ms"]) * 100
        
        status = "PASS" if enhanced_result["execution_time_ms"] <= self.thresholds["session_validation"] else "FAIL"
        
        return PerformanceTestResult(
            test_name="Session Management Performance",
            baseline_time=baseline_result["execution_time_ms"],
            security_enhanced_time=enhanced_result["execution_time_ms"],
            overhead_percent=overhead_percent,
            memory_usage_mb=enhanced_result["memory_used_mb"],
            cpu_usage_percent=0.0,
            status=status,
            threshold_ms=self.thresholds["session_validation"],
            details={
                "baseline_success": baseline_result["success"],
                "enhanced_success": enhanced_result["success"]
            }
        )
    
    def test_encryption_performance(self) -> PerformanceTestResult:
        """Test encryption performance."""
        def baseline_encryption():
            # Simple encoding
            import base64
            data = "sensitive_data_test" * 100  # Make it substantial
            return base64.b64encode(data.encode()).decode()
        
        def enhanced_encryption():
            # Real encryption
            from cryptography.fernet import Fernet
            data = "sensitive_data_test" * 100
            key = Fernet.generate_key()
            cipher = Fernet(key)
            encrypted = cipher.encrypt(data.encode())
            decrypted = cipher.decrypt(encrypted)
            return decrypted.decode() == data
        
        baseline_result = self.measure_execution_time(baseline_encryption)
        enhanced_result = self.measure_execution_time(enhanced_encryption)
        
        overhead_percent = ((enhanced_result["execution_time_ms"] - baseline_result["execution_time_ms"]) 
                           / baseline_result["execution_time_ms"]) * 100
        
        status = "PASS" if enhanced_result["execution_time_ms"] <= self.thresholds["encryption"] else "FAIL"
        
        return PerformanceTestResult(
            test_name="Encryption Performance",
            baseline_time=baseline_result["execution_time_ms"],
            security_enhanced_time=enhanced_result["execution_time_ms"],
            overhead_percent=overhead_percent,
            memory_usage_mb=enhanced_result["memory_used_mb"],
            cpu_usage_percent=0.0,
            status=status,
            threshold_ms=self.thresholds["encryption"],
            details={
                "baseline_success": baseline_result["success"],
                "enhanced_success": enhanced_result["success"]
            }
        )
    
    def test_input_validation_performance(self) -> PerformanceTestResult:
        """Test input validation performance."""
        def baseline_validation():
            # Basic validation
            test_input = "user_input_test_data_123"
            return len(test_input) > 0 and isinstance(test_input, str)
        
        def enhanced_validation():
            # Comprehensive validation
            import re
            test_input = "user_input_test_data_123"
            
            # Multiple validation checks
            checks = [
                len(test_input) > 0,
                len(test_input) < 1000,
                isinstance(test_input, str),
                not re.search(r'<script', test_input, re.IGNORECASE),
                not re.search(r'javascript:', test_input, re.IGNORECASE),
                not re.search(r'drop\s+table', test_input, re.IGNORECASE),
                test_input.isprintable(),
                not any(char in test_input for char in ['<', '>', '"', "'"] * 5)  # Make it more work
            ]
            
            return all(checks)
        
        baseline_result = self.measure_execution_time(baseline_validation)
        enhanced_result = self.measure_execution_time(enhanced_validation)
        
        overhead_percent = ((enhanced_result["execution_time_ms"] - baseline_result["execution_time_ms"]) 
                           / baseline_result["execution_time_ms"]) * 100 if baseline_result["execution_time_ms"] > 0 else 0
        
        status = "PASS" if enhanced_result["execution_time_ms"] <= self.thresholds["input_validation"] else "FAIL"
        
        return PerformanceTestResult(
            test_name="Input Validation Performance",
            baseline_time=baseline_result["execution_time_ms"],
            security_enhanced_time=enhanced_result["execution_time_ms"],
            overhead_percent=overhead_percent,
            memory_usage_mb=enhanced_result["memory_used_mb"],
            cpu_usage_percent=0.0,
            status=status,
            threshold_ms=self.thresholds["input_validation"],
            details={
                "baseline_success": baseline_result["success"],
                "enhanced_success": enhanced_result["success"]
            }
        )
    
    def test_concurrent_performance(self, num_threads: int = 10) -> Dict[str, Any]:
        """Test performance under concurrent load."""
        def test_operation():
            # Simulate a typical security operation
            import hashlib
            import time
            data = f"test_data_{time.time()}"
            return hashlib.sha256(data.encode()).hexdigest()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(test_operation) for _ in range(num_threads * 5)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        total_time = end_time - start_time
        operations_per_second = len(results) / total_time
        
        metrics = self.monitor.get_average_metrics()
        
        return {
            "total_operations": len(results),
            "total_time_seconds": total_time,
            "operations_per_second": operations_per_second,
            "concurrent_threads": num_threads,
            "average_metrics": metrics,
            "status": "PASS" if operations_per_second > 50 else "FAIL"  # Expect at least 50 ops/sec
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        logger.info("Starting simplified performance assessment...")
        
        start_time = time.time()
        
        # Run individual performance tests
        tests = [
            self.test_authentication_performance,
            self.test_session_performance,
            self.test_encryption_performance,
            self.test_input_validation_performance
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.results.append(result)
                logger.info(f"Test {result.test_name}: {result.status} "
                           f"({result.security_enhanced_time:.2f}ms, {result.overhead_percent:.1f}% overhead)")
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed: {e}")
        
        # Run concurrent test
        concurrent_result = self.test_concurrent_performance()
        
        total_time = time.time() - start_time
        
        # Calculate summary
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        total_tests = len(self.results)
        
        avg_overhead = sum(r.overhead_percent for r in self.results) / len(self.results) if self.results else 0
        max_execution_time = max(r.security_enhanced_time for r in self.results) if self.results else 0
        
        # Performance score (0-100)
        performance_score = 100.0
        
        # Deduct points for failed tests
        performance_score -= (total_tests - passed_tests) * 15
        
        # Deduct points for high overhead
        if avg_overhead > 50:
            performance_score -= (avg_overhead - 50) * 0.5
        
        # Deduct points for slow operations
        if max_execution_time > 1000:  # > 1 second
            performance_score -= (max_execution_time - 1000) * 0.01
        
        performance_score = max(0.0, performance_score)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "performance_score": performance_score,
            "average_overhead_percent": avg_overhead,
            "max_execution_time_ms": max_execution_time,
            "concurrent_performance": concurrent_result,
            "test_results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check for failed tests
        failed_tests = [r for r in self.results if r.status == "FAIL"]
        if failed_tests:
            for test in failed_tests:
                recommendations.append(f"Optimize {test.test_name.lower()}: "
                                     f"current {test.security_enhanced_time:.1f}ms exceeds "
                                     f"threshold {test.threshold_ms:.1f}ms")
        
        # Check for high overhead
        high_overhead_tests = [r for r in self.results if r.overhead_percent > 100]
        if high_overhead_tests:
            recommendations.append("Consider optimizing security implementations with >100% overhead")
        
        # Check for memory usage
        high_memory_tests = [r for r in self.results if r.memory_usage_mb > 50]
        if high_memory_tests:
            recommendations.append("Monitor memory usage in security operations")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds")
        
        return recommendations

def main():
    """Main function to run performance assessment."""
    print("=" * 60)
    print("LANGGRAPH 101 - SIMPLIFIED PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    # Create assessment and run tests
    assessment = SimplifiedPerformanceAssessment()
    results = assessment.run_all_tests()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ASSESSMENT RESULTS")
    print("=" * 60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Performance Score: {results['performance_score']:.1f}%")
    print(f"Average Overhead: {results['average_overhead_percent']:.1f}%")
    print(f"Max Execution Time: {results['max_execution_time_ms']:.1f}ms")
    
    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)
    for result in results['test_results']:
        status = "✅ PASS" if result['status'] == 'PASS' else "❌ FAIL"
        print(f"{status} {result['test_name']}")
        print(f"    Enhanced: {result['security_enhanced_time']:.2f}ms "
              f"(+{result['overhead_percent']:.1f}% overhead)")
        print(f"    Memory: {result['memory_usage_mb']:.2f}MB")
        print(f"    Threshold: {result['threshold_ms']:.1f}ms")
    
    print("\n" + "-" * 60)
    print("CONCURRENT PERFORMANCE")
    print("-" * 60)
    concurrent = results['concurrent_performance']
    print(f"Operations: {concurrent['total_operations']}")
    print(f"Time: {concurrent['total_time_seconds']:.2f}s")
    print(f"Ops/sec: {concurrent['operations_per_second']:.1f}")
    print(f"Threads: {concurrent['concurrent_threads']}")
    print(f"Status: {'✅ PASS' if concurrent['status'] == 'PASS' else '❌ FAIL'}")
    
    if concurrent.get('average_metrics'):
        metrics = concurrent['average_metrics']
        print(f"Avg CPU: {metrics.get('avg_cpu_percent', 0):.1f}%")
        print(f"Avg Memory: {metrics.get('avg_memory_mb', 0):.1f}MB")
    
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Save report
    report_file = f"performance_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return exit code based on results
    return 0 if results['performance_score'] >= 70 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
