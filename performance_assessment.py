"""
Performance Impact Assessment for Security Enhancements

Evaluates the performance impact of security features on the LangGraph 101 platform
and provides optimization recommendations.

Features:
- Baseline performance measurement
- Security feature impact analysis
- Resource utilization monitoring
- Bottleneck identification
- Performance optimization recommendations
- Scalability assessment
"""

import os
import time
import psutil
import json
import threading
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import asyncio
import requests
import sqlite3
from contextlib import contextmanager
import tracemalloc

# Import security modules for testing
from security_management import SecurityManager
from advanced_auth import MFAManager
from session_manager import SessionManager
from ddos_protection import DDoSProtection
from security_headers import SecurityHeadersManager
from input_security import InputSecurityManager
from audit_system import AuditManager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    io_operations: int = 0
    network_operations: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResourceUtilization:
    """System resource utilization."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceTestResult:
    """Performance test result."""
    test_name: str
    baseline_metrics: PerformanceMetrics
    security_metrics: PerformanceMetrics
    impact_percentage: float
    resource_overhead: ResourceUtilization
    passed: bool
    threshold: float
    recommendations: List[str] = field(default_factory=list)

@dataclass
class PerformanceAssessmentResults:
    """Complete performance assessment results."""
    overall_impact: float
    resource_overhead: float
    scalability_score: float
    optimization_potential: float
    test_results: List[PerformanceTestResult] = field(default_factory=list)
    resource_usage: List[ResourceUtilization] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class PerformanceMonitor:
    """Real-time performance monitoring - lightweight version."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.process = psutil.Process()
        
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Monitor performance metrics."""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Process metrics
                process_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                
                utilization = ResourceUtilization(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_mb=process_memory,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_io_sent=network_io.bytes_sent if network_io else 0,
                    network_io_recv=network_io.bytes_recv if network_io else 0,
                    active_connections=len(psutil.net_connections())
                )
                
                self.metrics.append(utilization)
                
                # Keep only recent metrics (last 1000 points)
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                    
            except Exception as e:
                logger.warning(f"Performance monitoring error: {e}")
            
            time.sleep(interval)
    
    def get_current_metrics(self) -> Optional[ResourceUtilization]:
        """Get current performance metrics."""
        return self.metrics[-1] if self.metrics else None
    
    def get_average_metrics(self, duration_seconds: int = 60) -> Optional[ResourceUtilization]:
        """Get average metrics over specified duration."""
        if not self.metrics:
            return None
        
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        return ResourceUtilization(
            cpu_percent=statistics.mean(m.cpu_percent for m in recent_metrics),
            memory_percent=statistics.mean(m.memory_percent for m in recent_metrics),
            memory_mb=statistics.mean(m.memory_mb for m in recent_metrics),
            disk_io_read=max(m.disk_io_read for m in recent_metrics),
            disk_io_write=max(m.disk_io_write for m in recent_metrics),
            network_io_sent=max(m.network_io_sent for m in recent_metrics),
            network_io_recv=max(m.network_io_recv for m in recent_metrics),
            active_connections=max(m.active_connections for m in recent_metrics)
        )

class PerformanceImpactAssessment:
    """Comprehensive performance impact assessment."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results = PerformanceAssessmentResults()
        self.monitor = PerformanceMonitor()
        
        # Performance thresholds
        self.thresholds = {
            "authentication": 1.0,      # seconds
            "authorization": 0.5,       # seconds
            "session_validation": 0.2,  # seconds
            "input_validation": 0.1,    # seconds
            "rate_limiting": 0.05,      # seconds
            "audit_logging": 0.1,       # seconds
            "encryption": 0.3,          # seconds
            "overall_overhead": 10.0    # percentage
        }
        
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load assessment configuration."""
        default_config = {
            "test_iterations": 100,
            "concurrent_users": 50,
            "test_duration": 300,  # 5 minutes
            "memory_limit_mb": 512,
            "cpu_limit_percent": 80,
            "baseline_warmup": 10,
            "security_warmup": 10
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def run_assessment(self) -> PerformanceAssessmentResults:
        """Run complete performance impact assessment."""
        start_time = time.time()
        self.logger.info("Starting performance impact assessment")
        
        try:
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Run performance tests
            self._test_authentication_performance()
            self._test_authorization_performance()
            self._test_session_management_performance()
            self._test_input_validation_performance()
            self._test_rate_limiting_performance()
            self._test_audit_logging_performance()
            self._test_encryption_performance()
            
            # Run scalability tests
            self._test_scalability()
            
            # Run memory leak detection
            self._test_memory_leaks()
            
            # Analyze results
            self._analyze_performance_results()
            self._identify_bottlenecks()
            self._generate_recommendations()
            
        except Exception as e:
            self.logger.error(f"Performance assessment failed: {e}")
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
            self.results.resource_usage = self.monitor.metrics.copy()
        
        self.results.execution_time = time.time() - start_time
        self.logger.info(f"Performance assessment completed in {self.results.execution_time:.2f}s")
        
        return self.results
    
    def _test_authentication_performance(self):
        """Test authentication performance impact."""
        self.logger.info("Testing authentication performance")
        
        # Setup test environment
        test_db = "perf_test_auth.db"
        security_manager = SecurityManager(test_db)
        
        # Create test user
        from security_management import User, UserRole, SecurityLevel
        test_user = User(
            id="test_user_id",
            username="perf_test_user",
            email="test@example.com",
            password_hash=security_manager._hash_password("TestPass123!"),
            role=UserRole.USER,
            is_verified=True,
            security_level=SecurityLevel.STANDARD
        )
        security_manager._save_user(test_user)
        
        try:
            # Baseline test (minimal security)
            baseline_times = []
            baseline_memory = []
            
            for _ in range(self.config["test_iterations"]):
                start_memory = psutil.Process().memory_info().rss
                start_time = time.perf_counter()
                
                # Simple authentication without security features
                user = security_manager.authenticate_user("perf_test_user", "TestPass123!")
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss
                
                baseline_times.append(end_time - start_time)
                baseline_memory.append((end_memory - start_memory) / 1024 / 1024)  # MB
            
            baseline_metrics = PerformanceMetrics(
                operation="authentication_baseline",
                execution_time=statistics.mean(baseline_times),
                memory_usage=statistics.mean(baseline_memory),
                cpu_usage=psutil.cpu_percent()
            )
            
            # Security-enhanced test
            mfa_manager = MFAManager(test_db)
            
            # Setup MFA for user
            mfa_secret = mfa_manager.setup_totp(test_user.id)
            
            security_times = []
            security_memory = []
            
            for _ in range(self.config["test_iterations"]):
                start_memory = psutil.Process().memory_info().rss
                start_time = time.perf_counter()
                
                # Enhanced authentication with MFA
                user = security_manager.authenticate_user("perf_test_user", "TestPass123!")
                if user and mfa_secret:
                    import pyotp
                    totp = pyotp.TOTP(mfa_secret)
                    mfa_manager.verify_totp(user.id, totp.now())
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss
                
                security_times.append(end_time - start_time)
                security_memory.append((end_memory - start_memory) / 1024 / 1024)  # MB
            
            security_metrics = PerformanceMetrics(
                operation="authentication_security",
                execution_time=statistics.mean(security_times),
                memory_usage=statistics.mean(security_memory),
                cpu_usage=psutil.cpu_percent()
            )
            
            # Calculate impact
            impact = ((security_metrics.execution_time - baseline_metrics.execution_time) / 
                     baseline_metrics.execution_time * 100)
            
            result = PerformanceTestResult(
                test_name="Authentication Performance",
                baseline_metrics=baseline_metrics,
                security_metrics=security_metrics,
                impact_percentage=impact,
                resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
                passed=impact <= self.thresholds["authentication"] * 100,
                threshold=self.thresholds["authentication"] * 100
            )
            
            if impact > self.thresholds["authentication"] * 100:
                result.recommendations.extend([
                    "Consider caching authentication results",
                    "Optimize password hashing parameters",
                    "Implement connection pooling for database operations"
                ])
            
            self.results.test_results.append(result)
            
        finally:
            # Cleanup
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _test_authorization_performance(self):
        """Test authorization performance impact."""
        self.logger.info("Testing authorization performance")
        
        test_db = "perf_test_authz.db"
        security_manager = SecurityManager(test_db)
        
        # Create test users with different roles
        from security_management import User, UserRole, SecurityLevel
        admin_user = User(
            id="admin_user_id",
            username="admin_user",
            email="admin@example.com",
            password_hash=security_manager._hash_password("AdminPass123!"),
            role=UserRole.ADMIN,
            is_verified=True,
            security_level=SecurityLevel.RESTRICTED
        )
        security_manager._save_user(admin_user)
        
        try:
            # Test permission checking performance
            baseline_times = []
            security_times = []
            
            # Baseline: Simple role check
            for _ in range(self.config["test_iterations"]):
                start_time = time.perf_counter()
                has_permission = admin_user.role == UserRole.ADMIN
                end_time = time.perf_counter()
                baseline_times.append(end_time - start_time)
            
            # Security: Full permission system
            for _ in range(self.config["test_iterations"]):
                start_time = time.perf_counter()
                has_permission = security_manager.check_permission(admin_user, "admin_access")
                end_time = time.perf_counter()
                security_times.append(end_time - start_time)
            
            baseline_metrics = PerformanceMetrics(
                operation="authorization_baseline",
                execution_time=statistics.mean(baseline_times),
                memory_usage=0,
                cpu_usage=psutil.cpu_percent()
            )
            
            security_metrics = PerformanceMetrics(
                operation="authorization_security",
                execution_time=statistics.mean(security_times),
                memory_usage=0,
                cpu_usage=psutil.cpu_percent()
            )
            
            impact = ((security_metrics.execution_time - baseline_metrics.execution_time) / 
                     baseline_metrics.execution_time * 100)
            
            result = PerformanceTestResult(
                test_name="Authorization Performance",
                baseline_metrics=baseline_metrics,
                security_metrics=security_metrics,
                impact_percentage=impact,
                resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
                passed=impact <= self.thresholds["authorization"] * 100,
                threshold=self.thresholds["authorization"] * 100
            )
            
            self.results.test_results.append(result)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _test_session_management_performance(self):
        """Test session management performance impact."""
        self.logger.info("Testing session management performance")
        
        test_db = "perf_test_session.db"
        session_manager = SessionManager(test_db)
        
        try:
            user_id = "test_user_123"
            ip_address = "192.168.1.100"
            user_agent = "Performance Test Browser"
            
            # Test session creation performance
            creation_times = []
            for _ in range(self.config["test_iterations"]):
                start_time = time.perf_counter()
                session_id = session_manager.create_session(user_id, ip_address, user_agent)
                end_time = time.perf_counter()
                creation_times.append(end_time - start_time)
                
                # Clean up session
                if session_id:
                    session_manager.invalidate_session(session_id)
            
            # Test session validation performance
            session_id = session_manager.create_session(user_id, ip_address, user_agent)
            validation_times = []
            
            for _ in range(self.config["test_iterations"]):
                start_time = time.perf_counter()
                session = session_manager.validate_session(session_id, ip_address, user_agent)
                end_time = time.perf_counter()
                validation_times.append(end_time - start_time)
            
            # Calculate metrics
            avg_creation_time = statistics.mean(creation_times)
            avg_validation_time = statistics.mean(validation_times)
            
            metrics = PerformanceMetrics(
                operation="session_management",
                execution_time=avg_creation_time + avg_validation_time,
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage=psutil.cpu_percent()
            )
            
            baseline_metrics = PerformanceMetrics(
                operation="session_baseline",
                execution_time=0.001,  # Minimal baseline
                memory_usage=0,
                cpu_usage=0
            )
            
            impact = (metrics.execution_time / baseline_metrics.execution_time) * 100
            
            result = PerformanceTestResult(
                test_name="Session Management Performance",
                baseline_metrics=baseline_metrics,
                security_metrics=metrics,
                impact_percentage=impact,
                resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
                passed=metrics.execution_time <= self.thresholds["session_validation"],
                threshold=self.thresholds["session_validation"]
            )
            
            if metrics.execution_time > self.thresholds["session_validation"]:
                result.recommendations.extend([
                    "Implement session caching",
                    "Optimize database queries",
                    "Use Redis for session storage",
                    "Implement session cleanup background task"
                ])
            
            self.results.test_results.append(result)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _test_input_validation_performance(self):
        """Test input validation performance impact."""
        self.logger.info("Testing input validation performance")
        
        input_security = InputSecurityManager()
        
        # Test inputs of varying complexity
        test_inputs = [
            "simple text",
            "user@example.com",
            "<script>alert('xss')</script>complex payload with multiple patterns",
            "'; DROP TABLE users; -- very long SQL injection attempt with multiple statements",
            "A" * 1000,  # Large input
            "Mixed content with <tags> and 'quotes' and \"double quotes\" and numbers 123456"
        ]
        
        baseline_times = []
        security_times = []
        
        # Baseline: No validation
        for text in test_inputs:
            for _ in range(10):
                start_time = time.perf_counter()
                processed = text.strip()  # Minimal processing
                end_time = time.perf_counter()
                baseline_times.append(end_time - start_time)
        
        # Security: Full validation
        for text in test_inputs:
            for _ in range(10):
                start_time = time.perf_counter()
                is_valid, error, sanitized = input_security.validate_input(text, "text")
                end_time = time.perf_counter()
                security_times.append(end_time - start_time)
        
        baseline_metrics = PerformanceMetrics(
            operation="input_baseline",
            execution_time=statistics.mean(baseline_times),
            memory_usage=0,
            cpu_usage=psutil.cpu_percent()
        )
        
        security_metrics = PerformanceMetrics(
            operation="input_security",
            execution_time=statistics.mean(security_times),
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage=psutil.cpu_percent()
        )
        
        impact = ((security_metrics.execution_time - baseline_metrics.execution_time) / 
                 baseline_metrics.execution_time * 100)
        
        result = PerformanceTestResult(
            test_name="Input Validation Performance",
            baseline_metrics=baseline_metrics,
            security_metrics=security_metrics,
            impact_percentage=impact,
            resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
            passed=security_metrics.execution_time <= self.thresholds["input_validation"],
            threshold=self.thresholds["input_validation"]
        )
        
        if security_metrics.execution_time > self.thresholds["input_validation"]:
            result.recommendations.extend([
                "Optimize regex patterns",
                "Implement input caching for repeated validations",
                "Use compiled regex patterns",
                "Consider async validation for large inputs"
            ])
        
        self.results.test_results.append(result)
    
    def _test_rate_limiting_performance(self):
        """Test rate limiting performance impact."""
        self.logger.info("Testing rate limiting performance")
        
        ddos_protection = DDoSProtection()
        
        # Test rate limit checking performance
        test_ip = "192.168.1.100"
        rate_limit_times = []
        
        for _ in range(self.config["test_iterations"]):
            start_time = time.perf_counter()
            is_allowed = ddos_protection.check_rate_limit(test_ip, "api")
            end_time = time.perf_counter()
            rate_limit_times.append(end_time - start_time)
        
        baseline_metrics = PerformanceMetrics(
            operation="rate_limit_baseline",
            execution_time=0.0001,  # Minimal baseline
            memory_usage=0,
            cpu_usage=0
        )
        
        security_metrics = PerformanceMetrics(
            operation="rate_limit_security",
            execution_time=statistics.mean(rate_limit_times),
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage=psutil.cpu_percent()
        )
        
        impact = (security_metrics.execution_time / baseline_metrics.execution_time) * 100
        
        result = PerformanceTestResult(
            test_name="Rate Limiting Performance",
            baseline_metrics=baseline_metrics,
            security_metrics=security_metrics,
            impact_percentage=impact,
            resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
            passed=security_metrics.execution_time <= self.thresholds["rate_limiting"],
            threshold=self.thresholds["rate_limiting"]
        )
        
        if security_metrics.execution_time > self.thresholds["rate_limiting"]:
            result.recommendations.extend([
                "Use Redis for distributed rate limiting",
                "Implement token bucket algorithm",
                "Optimize sliding window calculations",
                "Cache rate limit decisions"
            ])
        
        self.results.test_results.append(result)
    
    def _test_audit_logging_performance(self):
        """Test audit logging performance impact."""
        self.logger.info("Testing audit logging performance")
        
        test_db = "perf_test_audit.db"
        audit_manager = AuditManager(test_db)
        
        try:
            audit_times = []
            
            for _ in range(self.config["test_iterations"]):
                start_time = time.perf_counter()
                audit_manager.log_authentication("test_user", True, "192.168.1.100")
                end_time = time.perf_counter()
                audit_times.append(end_time - start_time)
            
            baseline_metrics = PerformanceMetrics(
                operation="audit_baseline",
                execution_time=0.0001,  # Minimal baseline
                memory_usage=0,
                cpu_usage=0
            )
            
            security_metrics = PerformanceMetrics(
                operation="audit_security",
                execution_time=statistics.mean(audit_times),
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage=psutil.cpu_percent()
            )
            
            impact = (security_metrics.execution_time / baseline_metrics.execution_time) * 100
            
            result = PerformanceTestResult(
                test_name="Audit Logging Performance",
                baseline_metrics=baseline_metrics,
                security_metrics=security_metrics,
                impact_percentage=impact,
                resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
                passed=security_metrics.execution_time <= self.thresholds["audit_logging"],
                threshold=self.thresholds["audit_logging"]
            )
            
            if security_metrics.execution_time > self.thresholds["audit_logging"]:
                result.recommendations.extend([
                    "Implement asynchronous logging",
                    "Use batch inserts for audit events",
                    "Optimize database schema",
                    "Consider log aggregation services"
                ])
            
            self.results.test_results.append(result)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _test_encryption_performance(self):
        """Test encryption performance impact."""
        self.logger.info("Testing encryption performance")
        
        test_db = "perf_test_encrypt.db"
        security_manager = SecurityManager(test_db)
        
        try:
            # Test data encryption/decryption performance
            test_data = "This is sensitive test data that needs to be encrypted"
            
            encryption_times = []
            decryption_times = []
            
            for _ in range(self.config["test_iterations"]):
                # Encryption performance
                start_time = time.perf_counter()
                encrypted = security_manager.encrypt_data(test_data)
                end_time = time.perf_counter()
                encryption_times.append(end_time - start_time)
                
                # Decryption performance
                start_time = time.perf_counter()
                decrypted = security_manager.decrypt_data(encrypted)
                end_time = time.perf_counter()
                decryption_times.append(end_time - start_time)
            
            baseline_metrics = PerformanceMetrics(
                operation="encryption_baseline",
                execution_time=0.0001,  # Minimal baseline
                memory_usage=0,
                cpu_usage=0
            )
            
            total_crypto_time = statistics.mean(encryption_times) + statistics.mean(decryption_times)
            
            security_metrics = PerformanceMetrics(
                operation="encryption_security",
                execution_time=total_crypto_time,
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                cpu_usage=psutil.cpu_percent()
            )
            
            impact = (security_metrics.execution_time / baseline_metrics.execution_time) * 100
            
            result = PerformanceTestResult(
                test_name="Encryption Performance",
                baseline_metrics=baseline_metrics,
                security_metrics=security_metrics,
                impact_percentage=impact,
                resource_overhead=self.monitor.get_current_metrics() or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
                passed=security_metrics.execution_time <= self.thresholds["encryption"],
                threshold=self.thresholds["encryption"]
            )
            
            if security_metrics.execution_time > self.thresholds["encryption"]:
                result.recommendations.extend([
                    "Use hardware acceleration for encryption",
                    "Optimize key derivation parameters",
                    "Implement encryption caching",
                    "Consider AES-NI instructions"
                ])
            
            self.results.test_results.append(result)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _test_scalability(self):
        """Test system scalability under load."""
        self.logger.info("Testing scalability under concurrent load")
        
        concurrent_users = self.config["concurrent_users"]
        test_duration = 30  # seconds
        
        # Setup test environment
        test_db = "perf_test_scale.db"
        security_manager = SecurityManager(test_db)
        session_manager = SessionManager(test_db)
        
        # Create test user
        from security_management import User, UserRole, SecurityLevel
        test_user = User(
            id="scale_test_user",
            username="scale_user",
            email="scale@example.com",
            password_hash=security_manager._hash_password("ScaleTest123!"),
            role=UserRole.USER,
            is_verified=True,
            security_level=SecurityLevel.STANDARD
        )
        security_manager._save_user(test_user)
        
        try:
            # Concurrent operation metrics
            successful_operations = 0
            failed_operations = 0
            response_times = []
            error_details = []
            
            def simulate_user_operations():
                """Simulate typical user operations."""
                nonlocal successful_operations, failed_operations, response_times, error_details
                
                end_time = time.time() + test_duration
                user_response_times = []
                
                while time.time() < end_time:
                    try:
                        operation_start = time.perf_counter()
                        
                        # Authenticate user
                        user = security_manager.authenticate_user("scale_user", "ScaleTest123!")
                        if not user:
                            failed_operations += 1
                            continue
                        
                        # Create session
                        session_id = session_manager.create_session(
                            user.id, "192.168.1.100", "Scale Test Browser"
                        )
                        if not session_id:
                            failed_operations += 1
                            continue
                        
                        # Validate session
                        session = session_manager.validate_session(
                            session_id, "192.168.1.100", "Scale Test Browser"
                        )
                        if not session:
                            failed_operations += 1
                            continue
                        
                        # Clean up session
                        session_manager.invalidate_session(session_id)
                        
                        operation_time = time.perf_counter() - operation_start
                        user_response_times.append(operation_time)
                        successful_operations += 1
                        
                        time.sleep(0.1)  # Simulate user think time
                        
                    except Exception as e:
                        failed_operations += 1
                        error_details.append(str(e))
                
                response_times.extend(user_response_times)
            
            # Run concurrent users
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(simulate_user_operations) for _ in range(concurrent_users)]
                
                # Wait for completion
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.warning(f"Scalability test thread failed: {e}")
                        failed_operations += 1
            
            execution_time = time.time() - start_time
            
            # Calculate scalability metrics
            total_operations = successful_operations + failed_operations
            success_rate = successful_operations / total_operations if total_operations > 0 else 0
            avg_response_time = statistics.mean(response_times) if response_times else 0
            
            # Throughput (operations per second)
            throughput = successful_operations / execution_time if execution_time > 0 else 0
            
            # Scalability score (0-100)
            scalability_score = min(100, (success_rate * 100) * (1 / max(1, avg_response_time)))
            
            self.results.scalability_score = scalability_score
            
            # Performance degradation under load
            baseline_response_time = 0.1  # Expected single-user response time
            degradation = ((avg_response_time - baseline_response_time) / baseline_response_time * 100)
            
            scalability_passed = (success_rate >= 0.95 and 
                                 avg_response_time <= 2.0 and 
                                 degradation <= 200)  # Allow 2x degradation
            
            result = PerformanceTestResult(
                test_name="Scalability Test",
                baseline_metrics=PerformanceMetrics("scalability_baseline", baseline_response_time, 0, 0),
                security_metrics=PerformanceMetrics("scalability_load", avg_response_time, 0, psutil.cpu_percent()),
                impact_percentage=degradation,
                resource_overhead=self.monitor.get_average_metrics(30) or ResourceUtilization(0, 0, 0, 0, 0, 0, 0),
                passed=scalability_passed,
                threshold=200.0  # 200% degradation threshold
            )
            
            result.recommendations.extend([
                f"Scalability Score: {scalability_score:.1f}/100",
                f"Throughput: {throughput:.1f} ops/sec",
                f"Success Rate: {success_rate:.1%}",
                f"Average Response Time: {avg_response_time:.3f}s"
            ])
            
            if not scalability_passed:
                result.recommendations.extend([
                    "Implement connection pooling",
                    "Add database read replicas",
                    "Implement caching layers",
                    "Optimize database queries",
                    "Consider horizontal scaling"
                ])
            
            self.results.test_results.append(result)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _test_memory_leaks(self):
        """Test for memory leaks in security components."""
        self.logger.info("Testing for memory leaks")
        
        # Start memory tracing
        tracemalloc.start()
        
        # Initial memory snapshot
        initial_snapshot = tracemalloc.take_snapshot()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations that might leak memory
        test_db = "perf_test_memory.db"
        security_manager = SecurityManager(test_db)
        session_manager = SessionManager(test_db)
        
        try:
            # Create and destroy many objects
            for i in range(1000):
                # Create sessions and invalidate them
                session_id = session_manager.create_session(
                    f"user_{i}", "192.168.1.100", "Memory Test Browser"
                )
                if session_id:
                    session_manager.validate_session(session_id, "192.168.1.100", "Memory Test Browser")
                    session_manager.invalidate_session(session_id)
                
                # Encrypt/decrypt data
                test_data = f"Test data {i}"
                encrypted = security_manager.encrypt_data(test_data)
                decrypted = security_manager.decrypt_data(encrypted)
                
                # Force garbage collection periodically
                if i % 100 == 0:
                    import gc
                    gc.collect()
            
            # Final memory snapshot
            final_snapshot = tracemalloc.take_snapshot()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Compare snapshots
            top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
            memory_growth = final_memory - initial_memory
            
            # Check for significant memory growth (>50MB for 1000 operations)
            memory_leak_detected = memory_growth > 50
            
            result = PerformanceTestResult(
                test_name="Memory Leak Test",
                baseline_metrics=PerformanceMetrics("memory_baseline", 0, initial_memory, 0),
                security_metrics=PerformanceMetrics("memory_final", 0, final_memory, 0),
                impact_percentage=memory_growth,
                resource_overhead=ResourceUtilization(0, 0, memory_growth, 0, 0, 0, 0),
                passed=not memory_leak_detected,
                threshold=50.0  # 50MB threshold
            )
            
            if memory_leak_detected:
                result.recommendations.extend([
                    f"Memory growth detected: {memory_growth:.1f}MB",
                    "Review object lifecycle management",
                    "Implement proper cleanup in finally blocks",
                    "Check for circular references",
                    "Monitor database connection cleanup"
                ])
                
                # Add top memory consumers
                for stat in top_stats[:5]:
                    result.recommendations.append(f"Top memory: {stat}")
            
            self.results.test_results.append(result)
            
        finally:
            tracemalloc.stop()
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def _analyze_performance_results(self):
        """Analyze overall performance results."""
        if not self.results.test_results:
            return
        
        # Calculate overall impact
        impacts = [r.impact_percentage for r in self.results.test_results if r.impact_percentage > 0]
        self.results.overall_impact = statistics.mean(impacts) if impacts else 0
        
        # Calculate resource overhead
        if self.results.resource_usage:
            avg_cpu = statistics.mean(r.cpu_percent for r in self.results.resource_usage)
            avg_memory = statistics.mean(r.memory_percent for r in self.results.resource_usage)
            self.results.resource_overhead = (avg_cpu + avg_memory) / 2
        
        # Calculate optimization potential
        failed_tests = [r for r in self.results.test_results if not r.passed]
        self.results.optimization_potential = len(failed_tests) / len(self.results.test_results) * 100
    
    def _identify_bottlenecks(self):
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check execution time bottlenecks
        for result in self.results.test_results:
            if not result.passed:
                if result.security_metrics.execution_time > result.threshold:
                    bottlenecks.append(f"Slow {result.test_name}: {result.security_metrics.execution_time:.3f}s")
        
        # Check resource bottlenecks
        if self.results.resource_usage:
            max_cpu = max(r.cpu_percent for r in self.results.resource_usage)
            max_memory = max(r.memory_percent for r in self.results.resource_usage)
            
            if max_cpu > 80:
                bottlenecks.append(f"High CPU usage: {max_cpu:.1f}%")
            if max_memory > 80:
                bottlenecks.append(f"High memory usage: {max_memory:.1f}%")
        
        # Check scalability bottlenecks
        if self.results.scalability_score < 70:
            bottlenecks.append(f"Poor scalability: {self.results.scalability_score:.1f}/100")
        
        self.results.bottlenecks = bottlenecks
    
    def _generate_recommendations(self):
        """Generate optimization recommendations."""
        recommendations = []
        
        # Overall recommendations based on results
        if self.results.overall_impact > self.thresholds["overall_overhead"]:
            recommendations.extend([
                "Overall performance impact exceeds threshold",
                "Consider implementing async processing for security operations",
                "Review and optimize database queries",
                "Implement caching strategies for frequently accessed data"
            ])
        
        # Specific recommendations from failed tests
        for result in self.results.test_results:
            if not result.passed and result.recommendations:
                recommendations.extend(result.recommendations)
        
        # Resource-based recommendations
        if self.results.resource_overhead > 50:
            recommendations.extend([
                "High resource utilization detected",
                "Consider horizontal scaling",
                "Implement resource pooling",
                "Optimize memory usage patterns"
            ])
        
        # Remove duplicates and limit recommendations
        unique_recommendations = list(set(recommendations))
        self.results.recommendations = unique_recommendations[:20]  # Top 20 recommendations
    
    def generate_report(self, output_path: str = "performance_assessment_report.json") -> Dict[str, Any]:
        """Generate comprehensive performance assessment report."""
        report = {
            "assessment_summary": {
                "execution_time": self.results.execution_time,
                "overall_impact": self.results.overall_impact,
                "resource_overhead": self.results.resource_overhead,
                "scalability_score": self.results.scalability_score,
                "optimization_potential": self.results.optimization_potential,
                "tests_passed": sum(1 for r in self.results.test_results if r.passed),
                "tests_failed": sum(1 for r in self.results.test_results if not r.passed),
                "total_tests": len(self.results.test_results)
            },
            "performance_thresholds": self.thresholds,
            "bottlenecks": self.results.bottlenecks,
            "recommendations": self.results.recommendations,
            "detailed_results": [],
            "resource_utilization": []
        }
        
        # Add detailed test results
        for result in self.results.test_results:
            report["detailed_results"].append({
                "test_name": result.test_name,
                "passed": result.passed,
                "impact_percentage": result.impact_percentage,
                "threshold": result.threshold,
                "baseline_time": result.baseline_metrics.execution_time,
                "security_time": result.security_metrics.execution_time,
                "memory_usage": result.security_metrics.memory_usage,
                "cpu_usage": result.security_metrics.cpu_usage,
                "recommendations": result.recommendations
            })
        
        # Add resource utilization data (sample every 10th point to reduce size)
        for i, util in enumerate(self.results.resource_usage[::10]):
            report["resource_utilization"].append({
                "timestamp": util.timestamp.isoformat(),
                "cpu_percent": util.cpu_percent,
                "memory_percent": util.memory_percent,
                "memory_mb": util.memory_mb,
                "disk_io_read": util.disk_io_read,
                "disk_io_write": util.disk_io_write,
                "network_io_sent": util.network_io_sent,
                "network_io_recv": util.network_io_recv
            })
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance assessment report saved to {output_path}")
        return report

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run performance impact assessment
    assessment = PerformanceImpactAssessment()
    results = assessment.run_assessment()
    
    # Generate report
    report = assessment.generate_report()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE IMPACT ASSESSMENT RESULTS")
    print("="*60)
    print(f"Overall Impact: {results.overall_impact:.1f}%")
    print(f"Resource Overhead: {results.resource_overhead:.1f}%")
    print(f"Scalability Score: {results.scalability_score:.1f}/100")
    print(f"Optimization Potential: {results.optimization_potential:.1f}%")
    print(f"Tests Passed: {sum(1 for r in results.test_results if r.passed)}/{len(results.test_results)}")
    
    if results.bottlenecks:
        print(f"\nIdentified Bottlenecks:")
        for bottleneck in results.bottlenecks:
            print(f"  • {bottleneck}")
    
    if results.recommendations:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results.recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nDetailed report saved to performance_assessment_report.json")
