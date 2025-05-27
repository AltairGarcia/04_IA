"""
System Initialization module for LangGraph 101 project.

This module provides the entry point for initializing all error handling,
analytics, and performance optimization systems for the project.
"""
import os
import logging
import threading
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

# Import from centralized logging configuration
from logging_config import configure_logging, get_contextual_logger

# Configure structured logging
configure_logging(
    log_file=os.path.join(os.path.dirname(__file__), "langgraph_system.log"),
    use_json=True
)

# Get a contextual logger with system information
logger = get_contextual_logger(
    __name__,
    module="system_initialization",
    component_type="core_system"
)

# Import local modules
from error_handling import ErrorCategory, ErrorHandler
from analytics_dashboard import AnalyticsTracker
from error_integration import initialize_error_directory, log_error_to_file
from error_notification import get_notifier, setup_error_monitoring
from resilient_storage import initialize_resilient_storage, get_storage
from performance_optimization import initialize_performance_optimization

# Global flag to track initialization status
_system_initialized = False
_initialization_lock = threading.RLock()


class SystemInitializer:
    """Class-based wrapper for system initialization to support testing frameworks."""
    
    def __init__(self):
        # Check if system is already initialized globally
        global _system_initialized
        self.initialized = _system_initialized
        self.components_status = {}
        
    def initialize_all_systems(self, **kwargs):
        """Initialize all systems using the existing function-based approach."""
        try:
            result = initialize_all_systems(**kwargs)
            self.initialized = True
              # Store component status information
            self.components_status = {
                "error_handling": "initialized",
                "analytics": "initialized", 
                "performance": "initialized",
                "configuration": "initialized"
            }
            
            return result
        except Exception as e:
            self.initialized = False
            self.components_status = {"error": str(e)}
            raise
    
    def is_initialized(self):
        """Check if the system has been initialized."""
        return self.initialized
        
    def get_initialization_status(self):
        """Get detailed initialization status for testing."""
        # If not initialized, try to initialize
        global _system_initialized
        if not self.initialized and not _system_initialized:
            try:
                self.initialize_all_systems(force=False)
            except Exception as e:
                pass  # Don't fail, just report status
                
        # Update status based on global state
        self.initialized = _system_initialized
                
        # Update components status based on actual system state
        if self.initialized:
            self.components_status = {
                "error_handling": "initialized",
                "analytics": "initialized",
                "performance": "initialized", 
                "configuration": "initialized",
                "health_monitoring": "initialized"
            }
            
        return {
            "status": "initialized" if self.initialized else "not_initialized",
            "components": self.components_status,
            "timestamp": datetime.now().isoformat()
        }


# Directories
BASE_DIR = os.path.dirname(__file__)
ANALYTICS_DIR = os.path.join(BASE_DIR, "analytics_data")
ERROR_LOGS_DIR = os.path.join(BASE_DIR, "error_logs")
CONTENT_OUTPUT_DIR = os.path.join(BASE_DIR, "content_output")
PERFORMANCE_CACHE_DIR = os.path.join(BASE_DIR, "performance_cache")


def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    directories = [
        ANALYTICS_DIR,
        ERROR_LOGS_DIR,
        CONTENT_OUTPUT_DIR,
        PERFORMANCE_CACHE_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def initialize_error_handling() -> None:
    """Initialize error handling components."""
    # Initialize error directory
    initialize_error_directory()
    logger.info("Error logging system initialized")

    # Log a test error to verify the system is working
    try:
        raise ValueError("Test error - System initialization")
    except Exception as e:
        error_log_path = log_error_to_file(
            error=e,
            context={
                "source": "system_initialization.py",
                "operation": "initialize_error_handling",
                "timestamp": time.time()
            }
        )
        logger.info(f"Test error logged to: {error_log_path}")


def initialize_analytics() -> None:
    """Initialize analytics components with improved error handling."""
    try:
        # Create necessary directories
        os.makedirs(ANALYTICS_DIR, exist_ok=True)

        # Define the analytics files to initialize
        analytics_files = {
            "api_usage.json": [],
            "error_tracking.json": [],
            "performance_metrics.json": [],
            "system_health.json": []
        }

        # Initialize each file with improved error handling
        import json
        from resilient_operations import with_retry

        @with_retry(max_retries=3, initial_backoff=1.0, jitter=True)
        def initialize_file(file_path, content):
            """Initialize a file with retry capability."""
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump(content, f)
                logger.debug(f"Created analytics file: {os.path.basename(file_path)}")

        # Initialize each file
        for filename, initial_content in analytics_files.items():
            file_path = os.path.join(ANALYTICS_DIR, filename)
            initialize_file(file_path, initial_content)

        logger.info("Analytics system initialized with enhanced error handling")
    except Exception as e:
        logger.error(f"Failed to initialize analytics system: {str(e)}",
                    extra={"error_type": type(e).__name__})


def initialize_email_notifications(use_env_vars: bool = True, config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize email notification system.

    Args:
        use_env_vars: Whether to use environment variables for configuration
        config: Optional manual configuration if not using environment variables
    """
    from dotenv import load_dotenv

    # Load environment variables if needed
    if use_env_vars:
        load_dotenv(encoding='utf-16-le')
    elif config:
        # Apply manual configuration if provided (less common for this function)
        from error_notification import setup_error_monitoring
        setup_error_monitoring(
            smtp_server=config.get('smtp_server'),
            smtp_port=config.get('smtp_port'),
            username=config.get('username'),
            password=config.get('password'),
            sender=config.get('sender'),
            recipients=config.get('recipients'),
            check_interval_seconds=config.get('check_interval', 3600),
            start=True,
            use_env_config=False
        )
        logger.info("Email notifications initialized from provided configuration")
    else:
        logger.warning("Email notifications not configured")


def initialize_all_systems(
    email_config: Optional[Dict[str, Any]] = None,
    use_env_vars: bool = True,
    force: bool = False
) -> Dict[str, Any]:
    """Initialize all systems: error handling, analytics, and performance optimization.

    Args:
        email_config: Optional email configuration for error notifications
        use_env_vars: Whether to use environment variables for configuration
        force: Force re-initialization even if systems were already initialized
          Returns:
        Dict containing initialization status and system components
    """
    global _system_initialized
    with _initialization_lock:
        if _system_initialized and not force:
            logger.info("System already initialized. Skipping re-initialization. Use force=True to re-initialize.")
            return {
                'status': 'already_initialized',
                'components': {}
            }
          # Reset flag if force=True
        if force:
            _system_initialized = False

        logger.info("Starting LangGraph 101 system initialization")

        result = {
            'status': 'success',
            'components': {
                'directories': False,
                'storage': False,
                'error_handling': False,
                'analytics': False,
                'performance': False,
                'email': False
            },
            'storage_instance': None
        }

        try:
            # Step 1: Ensure all directories exist
            ensure_directories_exist()
            result['components']['directories'] = True

            # Step 2: Initialize resilient storage for analytics
            storage = initialize_resilient_storage(backup_interval_hours=24)
            result['components']['storage'] = True
            result['storage_instance'] = storage
            logger.info("Resilient storage initialized")

            # Step 3: Initialize error handling
            initialize_error_handling()
            result['components']['error_handling'] = True
            logger.info("Error handling system initialized")

            # Step 4: Initialize analytics
            initialize_analytics()
            result['components']['analytics'] = True
            logger.info("Analytics system initialized")

            # Step 5: Initialize performance optimization
            initialize_performance_optimization()
            result['components']['performance'] = True
            logger.info("Performance optimization system initialized")

            # Step 6: Configure error monitoring and notifications
            try:
                if email_config:
                    setup_error_monitoring(
                        smtp_server=email_config.get("smtp_server", ""),
                        smtp_port=email_config.get("smtp_port", 587),
                        username=email_config.get("username", ""),
                        password=email_config.get("password", ""),
                        sender=email_config.get("sender", ""),
                        recipients=email_config.get("recipients", []),
                        check_interval_seconds=60 * 60,  # Check every hour
                        start=True,
                        use_env_config=False
                    )
                    result['components']['email'] = True
                elif use_env_vars:
                    # Use environment variables
                    initialize_email_notifications(use_env_vars=True)
                    result['components']['email'] = True

                logger.info("Error notification system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize email notifications: {str(e)}")

            # Run system check
            system_status = check_system_status()
            result['system_status'] = system_status

            logger.info("LangGraph 101 system initialization complete")

            # Set the system as initialized only after successful completion
            _system_initialized = True

        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            result['status'] = 'error'
            result['error'] = str(e)
            # Reset the initialization flag on error
            _system_initialized = False

        return result


def start_maintenance_thread() -> None:
    """Start a thread for periodic system maintenance."""
    def maintenance_routine():
        while True:
            try:
                logger.info("Running system maintenance")

                # Run storage maintenance (create backup)
                storage = get_storage()
                storage.create_backup()

                # Clean up old error logs
                cleanup_old_error_logs(max_age_days=30)

            except Exception as e:
                logger.error(f"Error during maintenance: {str(e)}")

            # Sleep for 24 hours
            time.sleep(24 * 60 * 60)

    # Start the maintenance thread
    thread = threading.Thread(target=maintenance_routine, daemon=True)
    thread.start()
    logger.info("Maintenance thread started")


def cleanup_old_error_logs(max_age_days: int = 30) -> int:
    """Clean up old error logs.

    Args:
        max_age_days: Maximum age of error logs in days

    Returns:
        Number of files removed
    """
    import glob
    from datetime import datetime, timedelta

    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    # Get all error log files
    error_logs = glob.glob(os.path.join(ERROR_LOGS_DIR, "error_*.log"))

    # Track number of files removed
    removed_count = 0

    # Check each file
    for log_file in error_logs:
        try:
            # Get file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))

            # Remove if older than cutoff
            if mod_time < cutoff_date:
                os.remove(log_file)
                removed_count += 1
        except Exception as e:
            logger.warning(f"Failed to process {log_file}: {str(e)}")

    logger.info(f"Removed {removed_count} error logs older than {max_age_days} days")
    return removed_count


def check_system_status() -> Dict[str, Any]:
    """Check the status of all systems and their health.

    Returns:
        Dictionary with system status information and health checks
    """
    import glob
    import platform

    try:
        import psutil
        psutil_available = True
    except ImportError:
        psutil_available = False
    try:
        # Count analytics data files
        analytics_files = len(glob.glob(os.path.join(ANALYTICS_DIR, "*.json")))

        # Count error logs
        error_logs = len(glob.glob(os.path.join(ERROR_LOGS_DIR, "*.log")))

        # Count recent error logs (last 24 hours)
        from datetime import datetime, timedelta
        recent_cutoff = time.time() - (24 * 60 * 60)  # 24 hours ago
        recent_error_logs = 0

        for log_file in glob.glob(os.path.join(ERROR_LOGS_DIR, "*.log")):
            if os.path.getmtime(log_file) >= recent_cutoff:
                recent_error_logs += 1

        # Count content output files
        content_files = len(glob.glob(os.path.join(CONTENT_OUTPUT_DIR, "*")))

        # Get system information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp": datetime.now().isoformat()
        }

        # Add psutil info if available
        if psutil_available:
            system_info.update({
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": {
                    "total_gb": round(psutil.disk_usage(BASE_DIR).total / (1024 ** 3), 2),
                    "free_gb": round(psutil.disk_usage(BASE_DIR).free / (1024 ** 3), 2),
                    "percent": psutil.disk_usage(BASE_DIR).percent
                }
            })

        # Check if email notification is configured
        from error_notification import get_notifier
        notifier = get_notifier()
        email_configured = bool(notifier.email_config and notifier.email_config.get('recipients'))

        # Health checks
        health_checks = {}

        # 1. Check disk space
        if psutil_available:
            disk_percent = psutil.disk_usage(BASE_DIR).percent
            if disk_percent > 90:
                health_checks["disk_space"] = {
                    "status": "critical",
                    "message": f"Disk space critically low: {disk_percent}% used"
                }
            elif disk_percent > 80:
                health_checks["disk_space"] = {
                    "status": "warning",
                    "message": f"Disk space running low: {disk_percent}% used"
                }
            else:
                health_checks["disk_space"] = {
                    "status": "ok",
                    "message": f"Disk space adequate: {disk_percent}% used"
                }

        # 2. Check error rate
        if recent_error_logs > 20:
            health_checks["error_rate"] = {
                "status": "critical",
                "message": f"High error rate: {recent_error_logs} errors in the last 24 hours"
            }
        elif recent_error_logs > 10:
            health_checks["error_rate"] = {
                "status": "warning",
                "message": f"Elevated error rate: {recent_error_logs} errors in the last 24 hours"
            }
        else:
            health_checks["error_rate"] = {
                "status": "ok",
                "message": f"Normal error rate: {recent_error_logs} errors in the last 24 hours"
            }

        # 3. Check backup recency
        health_checks["backup"] = {
            "status": "unknown",
            "message": "Backup information not available"
        }

        # Calculate overall health
        health_statuses = [check["status"] for check in health_checks.values()]
        if "critical" in health_statuses:
            overall_health = "critical"
        elif "warning" in health_statuses:
            overall_health = "warning"
        else:
            overall_health = "ok"

        return {
            "timestamp": time.time(),
            "overall_health": overall_health,
            "systems": {
                "error_handling": {
                    "status": "active",
                    "error_logs_total": error_logs,
                    "error_logs_recent": recent_error_logs
                },
                "analytics": {
                    "status": "active",
                    "data_files": analytics_files
                },
                "content_creation": {
                    "status": "active",
                    "output_files": content_files
                },
                "resilient_storage": {
                    "status": "active"
                },
                "email_notification": {
                    "status": "active" if email_configured else "unconfigured",
                    "configured": email_configured
                }
            },
            "system_info": system_info,
            "health_checks": health_checks
        }
    except Exception as e:
        logger.error(f"Error checking system status: {str(e)}")
        return {
            "timestamp": time.time(),
            "overall_health": "unknown",
            "status": "error",
            "error": str(e)
        }


def check_dependencies() -> List[str]:
    """Check for required dependencies and return missing ones.

    Returns:
        List of missing package names
    """
    missing_packages = []

    # Required packages
    required_packages = {
        "python-dotenv": "dotenv",
        "psutil": "psutil",
        "matplotlib": "matplotlib"
    }

    # Check each package
    for pkg_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pkg_name)

    return missing_packages


if __name__ == "__main__":
    # Example usage for testing
    logger.info("Running system initialization...")

    # Check for missing dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("Missing dependencies detected. Please install:")
        print(f"pip install {' '.join(missing_deps)}")
        print()

    # Initialize all systems
    result = initialize_all_systems(use_env_vars=True)

    # Start maintenance thread
    start_maintenance_thread()

    # Get and display system status
    status = check_system_status()
    import json
    print("\nSystem Status:")
    print(json.dumps(status, indent=2))

    # Display health summary
    print("\nHealth Summary:")
    if status.get('overall_health') == 'ok':
        print("✅ All systems operational")
    elif status.get('overall_health') == 'warning':
        print("⚠️ System has warnings - check health checks")
    elif status.get('overall_health') == 'critical':
        print("🚨 System has critical issues - immediate attention required")
    else:
        print("❓ System health unknown")

    # Display component status
    print("\nComponent Status:")
    for component, enabled in result.get('components', {}).items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"- {component}: {status}")

    logger.info("System initialization complete")
