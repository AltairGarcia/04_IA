"""
Application Health Check and Robustness Module.

This module provides comprehensive health checks, error monitoring,
and robustness improvements for the LangGraph 101 application.
"""
import os
import sys
import logging
import traceback
import functools
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import threading
import time

# Configure logging
logger = logging.getLogger(__name__)


class HealthCheckResult:
    """Represents the result of a health check."""
    
    def __init__(self, name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status  # 'ok', 'warning', 'critical'
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class HealthChecker:
    """Comprehensive health checker for the application."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.checks = []
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.checks = [
            self._check_python_version,
            self._check_required_packages,
            self._check_configuration,
            self._check_file_system,
            self._check_memory_usage,
            self._check_disk_space,
            self._check_network_connectivity,
            self._check_langchain_compatibility
        ]
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for check in self.checks:
            try:
                result = check()
                results[result.name] = result
            except Exception as e:
                self.logger.error(f"Health check failed: {check.__name__}: {e}")
                results[check.__name__] = HealthCheckResult(
                    name=check.__name__,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e), "traceback": traceback.format_exc()}
                )
        
        return results
    
    def _check_python_version(self) -> HealthCheckResult:
        """Check Python version compatibility."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            return HealthCheckResult(
                name="python_version",
                status="ok",
                message=f"Python {current_version[0]}.{current_version[1]} is compatible",
                details={"version": f"{current_version[0]}.{current_version[1]}"}
            )
        else:
            return HealthCheckResult(
                name="python_version",
                status="critical",
                message=f"Python {current_version[0]}.{current_version[1]} is too old. Minimum required: {min_version[0]}.{min_version[1]}",
                details={"version": f"{current_version[0]}.{current_version[1]}", "min_required": f"{min_version[0]}.{min_version[1]}"}
            )
    
    def _check_required_packages(self) -> HealthCheckResult:
        """Check if required packages are installed."""
        required_packages = [
            "streamlit", "langchain", "langchain_google_genai", 
            "dotenv", "requests", "chardet"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            return HealthCheckResult(
                name="required_packages",
                status="ok",
                message="All required packages are installed",
                details={"checked_packages": required_packages}
            )
        else:
            return HealthCheckResult(
                name="required_packages",
                status="critical",
                message=f"Missing packages: {', '.join(missing_packages)}",
                details={"missing_packages": missing_packages, "required_packages": required_packages}
            )
    
    def _check_configuration(self) -> HealthCheckResult:
        """Check application configuration."""
        try:
            from config_robust import load_config_robust
            config = load_config_robust()
            
            return HealthCheckResult(
                name="configuration",
                status="ok",
                message="Configuration loaded successfully",
                details={"api_key_configured": bool(config.api_key), "tavily_configured": bool(config.tavily_api_key)}
            )
        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status="critical",
                message=f"Configuration loading failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_file_system(self) -> HealthCheckResult:
        """Check file system access and required directories."""
        base_dir = os.path.dirname(__file__)
        required_dirs = ["analytics_data", "error_logs", "content_output", "performance_cache"]
        
        issues = []
        for dir_name in required_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            try:
                os.makedirs(dir_path, exist_ok=True)
                # Test write access
                test_file = os.path.join(dir_path, "health_check_test.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                issues.append(f"{dir_name}: {str(e)}")
        
        if not issues:
            return HealthCheckResult(
                name="file_system",
                status="ok",
                message="File system access is working correctly",
                details={"checked_directories": required_dirs}
            )
        else:
            return HealthCheckResult(
                name="file_system",
                status="critical",
                message=f"File system issues: {'; '.join(issues)}",
                details={"issues": issues}
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent < 80:
                status = "ok"
            elif memory.percent < 90:
                status = "warning"
            else:
                status = "critical"
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=f"Memory usage: {memory.percent:.1f}%",
                details={
                    "percent_used": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2)
                }
            )
        except ImportError:
            return HealthCheckResult(
                name="memory_usage",
                status="warning",
                message="psutil not available for memory monitoring",
                details={"error": "psutil not installed"}
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status="warning",
                message=f"Memory check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            import psutil
            base_dir = os.path.dirname(__file__)
            disk = psutil.disk_usage(base_dir)
            
            free_percent = (disk.free / disk.total) * 100
            
            if free_percent > 20:
                status = "ok"
            elif free_percent > 10:
                status = "warning"
            else:
                status = "critical"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=f"Free disk space: {free_percent:.1f}%",
                details={
                    "free_percent": free_percent,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2)
                }
            )
        except ImportError:
            return HealthCheckResult(
                name="disk_space",
                status="warning",
                message="psutil not available for disk monitoring",
                details={"error": "psutil not installed"}
            )
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status="warning",
                message=f"Disk space check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            import requests
            
            # Test connectivity to Google (for Gemini API)
            response = requests.get("https://www.google.com", timeout=10)
            if response.status_code == 200:
                return HealthCheckResult(
                    name="network_connectivity",
                    status="ok",
                    message="Network connectivity is working",
                    details={"test_url": "https://www.google.com"}
                )
            else:
                return HealthCheckResult(
                    name="network_connectivity",
                    status="warning",
                    message=f"Network test returned status code: {response.status_code}",
                    details={"status_code": response.status_code}
                )
        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                status="critical",
                message=f"Network connectivity failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_langchain_compatibility(self) -> HealthCheckResult:
        """Check LangChain compatibility."""
        try:
            from langchain_robust import check_langchain_compatibility
            compat = check_langchain_compatibility()
            
            status_map = {
                "fully_compatible": "ok",
                "partially_compatible": "warning",
                "incompatible": "critical",
                "not_installed": "critical"
            }
            
            return HealthCheckResult(
                name="langchain_compatibility",
                status=status_map.get(compat["status"], "warning"),
                message=f"LangChain status: {compat['status']}",
                details=compat
            )
        except Exception as e:
            return HealthCheckResult(
                name="langchain_compatibility",
                status="warning",
                message=f"LangChain compatibility check failed: {str(e)}",
                details={"error": str(e)}
            )


def robust_function(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator to make functions more robust with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        
        return wrapper
    return decorator


class ApplicationMonitor:
    """Monitor application health and performance."""
    
    def __init__(self, check_interval: int = 300):  # 5 minutes default
        self.check_interval = check_interval
        self.health_checker = HealthChecker()
        self.last_check_results = {}
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring is already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Started health monitoring with {self.check_interval}s interval")
    
    def stop_monitoring_process(self):
        """Stop health monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        self.logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.check_interval):
            try:
                self.last_check_results = self.health_checker.run_all_checks()
                self._process_health_results()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _process_health_results(self):
        """Process health check results and take action if needed."""
        critical_issues = []
        warning_issues = []
        
        for name, result in self.last_check_results.items():
            if result.status == "critical":
                critical_issues.append(result)
            elif result.status == "warning":
                warning_issues.append(result)
        
        if critical_issues:
            self.logger.error(f"Critical health issues detected: {[r.name for r in critical_issues]}")
        
        if warning_issues:
            self.logger.warning(f"Health warnings detected: {[r.name for r in warning_issues]}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.last_check_results:
            self.last_check_results = self.health_checker.run_all_checks()
        
        total_checks = len(self.last_check_results)
        ok_count = sum(1 for r in self.last_check_results.values() if r.status == "ok")
        warning_count = sum(1 for r in self.last_check_results.values() if r.status == "warning")
        critical_count = sum(1 for r in self.last_check_results.values() if r.status == "critical")
        
        overall_status = "ok"
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "total_checks": total_checks,
            "ok_count": ok_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "last_check_time": max(r.timestamp for r in self.last_check_results.values()).isoformat() if self.last_check_results else None,
            "checks": {name: result.to_dict() for name, result in self.last_check_results.items()}
        }


# Global health checker and monitor instances
health_checker = HealthChecker()
app_monitor = ApplicationMonitor()


def run_health_check() -> Dict[str, Any]:
    """Run a complete health check and return results."""
    return health_checker.run_all_checks()


def get_health_summary() -> Dict[str, Any]:
    """Get application health summary."""
    return app_monitor.get_health_summary()


def start_health_monitoring():
    """Start continuous health monitoring."""
    app_monitor.start_monitoring()


def stop_health_monitoring():
    """Stop health monitoring."""
    app_monitor.stop_monitoring_process()


if __name__ == "__main__":
    # Run health check
    print("Running application health check...")
    results = run_health_check()
    
    for name, result in results.items():
        status_emoji = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}.get(result.status, "❓")
        print(f"{status_emoji} {result.name}: {result.message}")
    
    # Get summary
    summary = get_health_summary()
    print(f"\nOverall Status: {summary['overall_status']}")
    print(f"Checks: {summary['ok_count']} OK, {summary['warning_count']} Warning, {summary['critical_count']} Critical")
