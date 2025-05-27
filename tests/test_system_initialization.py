"""
Tests for system_initialization.py
"""
import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the error_notification module
sys.modules['error_notification'] = MagicMock()
sys.modules['error_notification'].get_notifier = MagicMock()
sys.modules['error_notification'].setup_error_monitoring = MagicMock()
sys.modules['error_notification'].setup_error_monitoring_from_env = MagicMock()

import logging
from system_initialization import (
    ensure_directories_exist,
    initialize_analytics,
    initialize_email_notifications,
    initialize_all_systems,
    check_dependencies,
    cleanup_old_error_logs,
    check_system_status
)

class TestSystemInitialization(unittest.TestCase):
    """Test cases for system initialization module."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Mock the BASE_DIR
        self.base_dir_patcher = patch('system_initialization.BASE_DIR', self.test_dir)
        self.mock_base_dir = self.base_dir_patcher.start()

        # Create mock directories
        self.analytics_dir = os.path.join(self.test_dir, "analytics_data")
        self.error_logs_dir = os.path.join(self.test_dir, "error_logs")
        self.content_output_dir = os.path.join(self.test_dir, "content_output")
        self.performance_cache_dir = os.path.join(self.test_dir, "performance_cache")

        # Patch directory paths
        self.dir_patchers = [
            patch('system_initialization.ANALYTICS_DIR', self.analytics_dir),
            patch('system_initialization.ERROR_LOGS_DIR', self.error_logs_dir),
            patch('system_initialization.CONTENT_OUTPUT_DIR', self.content_output_dir),
            patch('system_initialization.PERFORMANCE_CACHE_DIR', self.performance_cache_dir)
        ]
        for patcher in self.dir_patchers:
            patcher.start()

        # Disable logging for tests
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """Clean up test environment."""
        # Stop all patches
        self.base_dir_patcher.stop()
        for patcher in self.dir_patchers:
            patcher.stop()

        # Remove temporary directory
        shutil.rmtree(self.test_dir)

        # Re-enable logging
        logging.disable(logging.NOTSET)

    def test_ensure_directories_exist(self):
        """Test ensure_directories_exist creates all required directories."""
        # Call the function
        ensure_directories_exist()
        # Check that directories were created
        self.assertTrue(os.path.exists(self.analytics_dir))
        self.assertTrue(os.path.exists(self.error_logs_dir))
        self.assertTrue(os.path.exists(self.content_output_dir))
        self.assertTrue(os.path.exists(self.performance_cache_dir))

    def test_initialize_analytics(self):
        """Test initialize_analytics creates required JSON files."""
        # Call the function
        initialize_analytics()

        # Check that analytics directory was created
        self.assertTrue(os.path.exists(self.analytics_dir))

        # Check that JSON files were created
        self.assertTrue(os.path.exists(os.path.join(self.analytics_dir, "api_usage.json")))
        self.assertTrue(os.path.exists(os.path.join(self.analytics_dir, "error_tracking.json")))
        self.assertTrue(os.path.exists(os.path.join(self.analytics_dir, "performance_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.analytics_dir, "system_health.json")))

        # Check file contents
        with open(os.path.join(self.analytics_dir, "api_usage.json"), 'r') as f:
            self.assertEqual(json.load(f), [])

    @patch('resilient_operations.with_retry')
    def test_initialize_analytics_with_retry(self, mock_with_retry):
        """Test initialize_analytics uses retry mechanism for file creation."""
        # Setup the mock to ensure it's called
        mock_decorator = MagicMock()
        mock_with_retry.return_value = mock_decorator

        # Call the function
        initialize_analytics()

        # Verify retry decorator was used
        mock_with_retry.assert_called_once()
        self.assertEqual(mock_with_retry.call_args[1]['max_retries'], 3)
        self.assertEqual(mock_with_retry.call_args[1]['initial_backoff'], 1.0)
        self.assertTrue(mock_with_retry.call_args[1]['jitter'])

    @patch('json.dump')
    def test_initialize_analytics_error_handling(self, mock_dump):
        """Test initialize_analytics handles errors properly."""
        # Setup the mock to raise an exception
        mock_dump.side_effect = IOError("Test file error")

        # Call the function without raising exception outside
        initialize_analytics()

        # The function should complete without raising an exception
        # We can't easily verify the error logging without complex mocking

    @patch('error_notification.setup_error_monitoring_from_env')
    @patch('error_notification.setup_error_monitoring')
    def test_initialize_email_notifications(self, mock_setup, mock_setup_from_env):
        """Test initialize_email_notifications with different configurations."""
        # Test with environment variables
        initialize_email_notifications(use_env_vars=True)
        mock_setup_from_env.assert_called_once_with(start=True)
        mock_setup.assert_not_called()

        # Reset mocks
        mock_setup.reset_mock()
        mock_setup_from_env.reset_mock()

        # Test with config
        test_config = {
            'smtp_server': 'test-server',
            'smtp_port': 587,
            'username': 'test-user',
            'password': 'test-pass',
            'sender': 'test@example.com',
            'recipients': ['recipient@example.com'],
            'check_interval': 7200
        }
        initialize_email_notifications(use_env_vars=False, config=test_config)
        mock_setup.assert_called_once()
        self.assertEqual(mock_setup.call_args[1]['smtp_server'], 'test-server')
        self.assertEqual(mock_setup.call_args[1]['smtp_port'], 587)
        mock_setup_from_env.assert_not_called()

        # Reset mocks
        mock_setup.reset_mock()
        mock_setup_from_env.reset_mock()

        # Test with neither env vars nor config
        initialize_email_notifications(use_env_vars=False, config=None)
        mock_setup.assert_not_called()
        mock_setup_from_env.assert_not_called()

    @patch('system_initialization.initialize_resilient_storage')
    @patch('system_initialization.initialize_error_handling')
    @patch('system_initialization.initialize_analytics')
    @patch('system_initialization.initialize_performance_optimization')
    @patch('system_initialization.setup_error_monitoring')
    @patch('system_initialization.initialize_email_notifications')
    @patch('system_initialization.check_system_status')
    def test_initialize_all_systems(
        self, mock_check_status, mock_init_email, mock_setup,
        mock_init_perf, mock_init_analytics, mock_init_error, mock_init_storage
    ):
        """Test initialize_all_systems calls all initialization functions."""        # Setup mocks
        mock_storage = MagicMock()
        mock_init_storage.return_value = mock_storage
        mock_check_status.return_value = {"overall_health": "ok"}

        # Call the function with force=True to ensure re-initialization
        result = initialize_all_systems(force=True)

        # Check that all initialization functions were called
        mock_init_storage.assert_called_once()
        mock_init_error.assert_called_once()
        mock_init_analytics.assert_called_once()
        mock_init_perf.assert_called_once()
        mock_init_email.assert_called_once()
        mock_check_status.assert_called_once()
          # Check the result
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['components']['directories'], True)
        self.assertEqual(result['components']['storage'], True)
        self.assertEqual(result['components']['error_handling'], True)
        self.assertEqual(result['components']['analytics'], True)
        self.assertEqual(result['components']['performance'], True)
        self.assertEqual(result['storage_instance'], mock_storage)
        self.assertEqual(result['system_status']['overall_health'], 'ok')

    @patch('system_initialization.initialize_resilient_storage')
    def test_initialize_all_systems_with_error(self, mock_init_storage):
        """Test initialize_all_systems handles errors."""
        # Setup mock to raise an exception
        mock_init_storage.side_effect = Exception("Test error")

        # Call the function with force=True to ensure re-initialization
        result = initialize_all_systems(force=True)

        # Check the result
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], 'Test error')

    def test_cleanup_old_error_logs(self):
        """Test cleanup_old_error_logs removes old log files."""
        import time
        from datetime import datetime, timedelta

        # Create test log files
        os.makedirs(self.error_logs_dir, exist_ok=True)

        # Create an old log file (40 days old)
        old_log = os.path.join(self.error_logs_dir, "error_old.log")
        with open(old_log, 'w') as f:
            f.write("Old error log")

        # Set modification time to 40 days ago
        old_time = time.time() - (40 * 24 * 60 * 60)
        os.utime(old_log, (old_time, old_time))

        # Create a recent log file
        recent_log = os.path.join(self.error_logs_dir, "error_recent.log")
        with open(recent_log, 'w') as f:
            f.write("Recent error log")

        # Call the function (30 days max age)
        removed = cleanup_old_error_logs(max_age_days=30)

        # Check results
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(old_log))
        self.assertTrue(os.path.exists(recent_log))

    def test_check_dependencies(self):
        """Test check_dependencies identifies missing packages."""
        def mock_import(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError(f"No module named '{name}'")
            return None

        with patch('builtins.__import__', side_effect=mock_import):
            missing = check_dependencies()
            self.assertIn('psutil', missing)


if __name__ == '__main__':
    unittest.main()
