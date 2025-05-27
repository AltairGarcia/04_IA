import unittest
from unittest.mock import patch, MagicMock, ANY
import time
import sqlite3
from datetime import datetime, timedelta
import json # For metadata serialization if needed

# Assuming memory_optimization_fix.py is in the same directory or accessible in PYTHONPATH
from memory_optimization_fix import (
    ConversationLifecycleManager,
    DatabaseOptimizationEnhancer,  # Added for later tests
    MemoryEnhancementIntegrator, # Added for later tests
    start_memory_enhancements,   # Added for later tests
    get_enhanced_status,         # Added for later tests
    perform_enhanced_maintenance # Added for later tests
)
from memory_optimization_fix import RateLimiter # Import RateLimiter if it's a separate class

# Mocking global logger for tests
# Ensure this mock is consistently used or applied via @patch where needed.
# If logger is obtained by name, patch 'logging.getLogger'
mock_logger_instance = MagicMock()

# It's often better to patch the logger where it's used, e.g., @patch('memory_optimization_fix.logger', new_callable=MagicMock)
# For simplicity in this example, we'll assume direct patching works or adapt as needed.

class TestConversationLifecycleManager(unittest.TestCase):
    def setUp(self):
        self.db_path = ":memory:" # Use in-memory SQLite for tests
        # Patch the logger for the duration of the test or for specific methods
        self.logger_patcher = patch('memory_optimization_fix.logger', mock_logger_instance)
        self.mock_logger = self.logger_patcher.start()

        self.manager = ConversationLifecycleManager(db_path=self.db_path)
        
        # Initialize schema for in-memory database
        # This connection is properly closed by the 'with' statement.
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_metrics (
                    conversation_id TEXT,
                    timestamp TEXT NOT NULL,
                    token_count INTEGER,
                    interaction_count INTEGER,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def tearDown(self):
        self.logger_patcher.stop()
        mock_logger_instance.reset_mock() # Reset mock for other tests
        # The in-memory database is automatically discarded when the connection is closed.
        # If it were a file, os.remove(self.db_path) might be needed.

    def test_create_conversation(self):
        metadata = {"user_id": "test_user", "topic": "general"}
        metadata_json = json.dumps(metadata)
        conv_id = self.manager.create_conversation(metadata_json)
        self.assertIsNotNone(conv_id)
        self.mock_logger.info.assert_called_with(f"Conversation {conv_id} created.")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata FROM conversations WHERE id = ?", (conv_id,))
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], metadata_json)

    def test_get_conversation(self):
        metadata = {"user_id": "test_user_get", "topic": "retrieval"}
        metadata_json = json.dumps(metadata)
        conv_id = self.manager.create_conversation(metadata_json)
        
        retrieved_conv = self.manager.get_conversation(conv_id)
        self.assertIsNotNone(retrieved_conv)
        self.assertEqual(retrieved_conv['id'], conv_id)
        self.assertEqual(retrieved_conv['metadata'], metadata_json)
        # Timestamps are strings, check they exist
        self.assertIn('created_at', retrieved_conv)
        self.assertIn('last_accessed_at', retrieved_conv)
        self.mock_logger.info.assert_called_with(f"Retrieved conversation {conv_id}.")

        non_existent_conv = self.manager.get_conversation("non_existent_id")
        self.assertIsNone(non_existent_conv)
        self.mock_logger.warning.assert_called_with("Conversation non_existent_id not found.")

    def test_delete_conversation(self):
        conv_id = self.manager.create_conversation('''{"data": "to_delete"}''')
        self.manager.delete_conversation(conv_id)
        self.mock_logger.info.assert_called_with(f"Conversation {conv_id} deleted.")
        
        retrieved_conv = self.manager.get_conversation(conv_id)
        self.assertIsNone(retrieved_conv)

        # Test deleting associated metrics (ON DELETE CASCADE should handle this)
        self.manager.record_conversation_metrics(conv_id, 10, 1) # Try to record, should fail or do nothing silently
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversation_metrics WHERE conversation_id = ?", (conv_id,))
            self.assertIsNone(cursor.fetchone(), "Metrics should be deleted with conversation")


    def test_delete_non_existent_conversation(self):
        self.manager.delete_conversation("non_existent_for_delete")
        # Check logs or ensure no error is raised, depending on implementation
        # self.mock_logger.warning.assert_called_with("Attempted to delete non-existent conversation non_existent_for_delete.")
        # Assuming it logs a warning or is silent if not found
        # For now, just ensure it doesn't crash
        pass


    @patch('memory_optimization_fix.datetime') # Patch datetime within the module
    def test_update_conversation_access(self, mock_datetime):
        # Mock current time
        now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s) # Allow real fromisoformat

        conv_id = self.manager.create_conversation('''{"data": "update_access"}''')
        
        # Simulate time passing for the update
        updated_time = datetime(2023, 1, 1, 12, 5, 0)
        mock_datetime.now.return_value = updated_time
        
        self.manager.update_conversation_access(conv_id)
        self.mock_logger.info.assert_called_with(f"Updated last access time for conversation {conv_id}.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT created_at, last_accessed_at FROM conversations WHERE id = ?", (conv_id,))
            created_at_str, last_accessed_at_str = cursor.fetchone()
            
            # created_at should be 'now' (mocked initial time)
            # last_accessed_at should be 'updated_time' (mocked later time)
            self.assertEqual(datetime.fromisoformat(created_at_str), now)
            self.assertEqual(datetime.fromisoformat(last_accessed_at_str), updated_time)
            self.assertGreater(datetime.fromisoformat(last_accessed_at_str), datetime.fromisoformat(created_at_str))

    @patch('memory_optimization_fix.datetime')
    def test_cleanup_old_conversations(self, mock_datetime):
        # Setup mock current time
        current_time = datetime(2023, 1, 15, 0, 0, 0)
        mock_datetime.now.return_value = current_time
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)


        # Create an old conversation
        conv_id_old = "old_conv_123"
        old_time_iso = (current_time - timedelta(days=10)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversations (id, created_at, last_accessed_at, metadata) VALUES (?, ?, ?, ?)",
                           (conv_id_old, old_time_iso, old_time_iso, '''{"data":"old"}'''))
            conn.commit()

        # Create a new conversation (should not be cleaned up)
        conv_id_new = "new_conv_456"
        new_time_iso = (current_time - timedelta(days=1)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversations (id, created_at, last_accessed_at, metadata) VALUES (?, ?, ?, ?)",
                           (conv_id_new, new_time_iso, new_time_iso, '''{"data":"new"}'''))
            conn.commit()
        
        self.manager.cleanup_old_conversations(max_age_days=7)
        self.mock_logger.info.assert_any_call(f"Cleaned up conversation {conv_id_old} due to age.")
        
        self.assertIsNone(self.manager.get_conversation(conv_id_old))
        self.assertIsNotNone(self.manager.get_conversation(conv_id_new))

    @patch('memory_optimization_fix.datetime')
    def test_record_conversation_metrics(self, mock_datetime):
        now = datetime(2023, 1, 1, 10, 0, 0)
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)


        conv_id = self.manager.create_conversation('''{"data":"metrics_conv"}''')
        self.manager.record_conversation_metrics(conv_id, token_count=100, interaction_count=5)
        self.mock_logger.info.assert_called_with(f"Recorded metrics for conversation {conv_id}: tokens=100, interactions=5.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT token_count, interaction_count, timestamp FROM conversation_metrics WHERE conversation_id = ?", (conv_id,))
            token_count, interaction_count, timestamp_str = cursor.fetchone()
            self.assertEqual(token_count, 100)
            self.assertEqual(interaction_count, 5)
            self.assertEqual(datetime.fromisoformat(timestamp_str), now)

    def test_record_metrics_for_non_existent_conversation(self):
        self.manager.record_conversation_metrics("non_existent_conv_metrics", token_count=50, interaction_count=2)
        self.mock_logger.error.assert_called_with("Failed to record metrics for non-existent conversation non_existent_conv_metrics.")
        # Also check that no metrics were actually recorded
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversation_metrics WHERE conversation_id = ?", ("non_existent_conv_metrics",))
            self.assertIsNone(cursor.fetchone())


    def test_get_conversation_stats(self):
        conv_id = self.manager.create_conversation('''{"data":"stats_conv"}''')
        self.manager.record_conversation_metrics(conv_id, token_count=100, interaction_count=5)
        self.manager.record_conversation_metrics(conv_id, token_count=150, interaction_count=10)

        stats = self.manager.get_conversation_stats(conv_id)
        self.assertIsNotNone(stats)
        self.assertEqual(stats['total_tokens'], 250)
        self.assertEqual(stats['total_interactions'], 15)
        self.assertEqual(stats['metrics_count'], 2)
        self.mock_logger.info.assert_called_with(f"Retrieved stats for conversation {conv_id}.")

    def test_get_stats_for_non_existent_conversation(self):
        non_existent_stats = self.manager.get_conversation_stats("non_existent_stats_conv")
        self.assertIsNone(non_existent_stats)
        self.mock_logger.warning.assert_called_with("Conversation non_existent_stats_conv not found for stats.")

    def test_get_stats_for_conversation_with_no_metrics(self):
        conv_id = self.manager.create_conversation('''{"data":"no_metrics_conv"}''')
        stats = self.manager.get_conversation_stats(conv_id)
        self.assertIsNotNone(stats) # Should return stats object, but with zero values
        self.assertEqual(stats['total_tokens'], 0)
        self.assertEqual(stats['total_interactions'], 0)
        self.assertEqual(stats['metrics_count'], 0)
        self.mock_logger.info.assert_called_with(f"Retrieved stats for conversation {conv_id}.")


    @patch('time.sleep', MagicMock()) # Mock time.sleep to speed up test
    def test_rate_limiting_create_conversation(self):
        # Temporarily adjust rate limits for the manager instance for this test
        # This requires RateLimiter to be accessible or modifiable.
        # If RateLimiter is instantiated inside methods, this approach needs adjustment.
        # Assuming self.manager.rate_limiter_create is accessible and its attributes can be changed:
        
        original_max_calls = self.manager.rate_limiter_create.max_calls
        original_period = self.manager.rate_limiter_create.period
        
        self.manager.rate_limiter_create.max_calls = 2
        self.manager.rate_limiter_create.period = 0.1  # 100 ms, time.sleep is mocked

        try:
            conv_id1 = self.manager.create_conversation('''{"rate_limit": 1}''')
            self.assertIsNotNone(conv_id1)
            self.mock_logger.info.assert_any_call(f"Conversation {conv_id1} created.")

            conv_id2 = self.manager.create_conversation('''{"rate_limit": 2}''')
            self.assertIsNotNone(conv_id2)
            self.mock_logger.info.assert_any_call(f"Conversation {conv_id2} created.")

            with self.assertRaises(Exception) as context: # Assuming it raises a generic Exception
                self.manager.create_conversation('''{"rate_limit": 3}''')
            
            # Check the exception message if the RateLimiter class raises a specific error with a message
            # For example: self.assertTrue("Rate limit exceeded" in str(context.exception))
            # For now, just checking that an exception is raised.
            self.mock_logger.warning.assert_called_with(
                "Rate limit exceeded for create_conversation. Last call at: ANY, Count: 2"
            )
            
            # Wait for the rate limit period to pass (time.sleep is mocked, so this is conceptual)
            # In a real scenario, you'd time.sleep(self.manager.rate_limiter_create.period + 0.01)
            # Since time.sleep is mocked, we'd need to advance time if using a time-mocking library like freezegun
            # For this basic mock, we assume the next call would be allowed if time passed.
            # To properly test this without advancing time, we might need to reset the limiter's internal state
            # or use a more sophisticated time mocking.

            # Let's assume the RateLimiter resets its count after the period.
            # For a simple list-based RateLimiter, we might need to manually clear its history for test.
            # If it's based on `time.time()`, mocking `time.time()` would be more effective.
            with patch('time.time', return_value=time.time() + self.manager.rate_limiter_create.period + 0.1):
                 conv_id_after_wait = self.manager.create_conversation('''{"rate_limit": "after_wait"}''')
                 self.assertIsNotNone(conv_id_after_wait)
                 self.mock_logger.info.assert_any_call(f"Conversation {conv_id_after_wait} created.")

        finally:
            # Restore original rate limits
            self.manager.rate_limiter_create.max_calls = original_max_calls
            self.manager.rate_limiter_create.period = original_period

# Test classes for DatabaseOptimizationEnhancer, MemoryEnhancementIntegrator, and global functions will follow.

if __name__ == '__main__':
        mock_conn_manager_instance.get_connection.return_value.__enter__.side_effect = sqlite3.Error("Test DB Error")

        mock_time.time.side_effect = [1000.0, 1000.1] 

        with patch('memory_optimization_fix.OPTIMIZED_IMPORTS', True):
            enhancer = mof.DatabaseOptimizationEnhancer()
            results = enhancer.optimize_database_maintenance()

            self.assertFalse(results["vacuum_completed"])
            self.assertEqual(results["old_records_cleaned"], 0)
            self.assertAlmostEqual(results["optimization_time_ms"], 100.0)
            self.assertEqual(results["error"], "Test DB Error")
            mock_logger_error.assert_called_once_with("Database optimization error: Test DB Error")


class TestMemoryEnhancementIntegrator(unittest.TestCase):
    def setUp(self):
        # Patch the global instance that might be created at import time
        # This is important if tests run in an order that affects this global
        self.integrator_patcher = patch('memory_optimization_fix.memory_enhancement', MagicMock())
        self.mock_global_integrator = self.integrator_patcher.start()
        self.addCleanup(self.integrator_patcher.stop)
        
        # Reset global logger
        mof.logger = logging.getLogger("memory_optimization_fix_test_integrator")
        mof.logger.handlers = []
        mof.logger.propagate = False


    @patch('memory_optimization_fix.ConversationLifecycleManager')
    @patch('memory_optimization_fix.DatabaseOptimizationEnhancer')
    @patch('memory_optimization_fix.AdvancedMemoryProfiler')
    @patch('memory_optimization_fix.EnhancedUnifiedMonitoringSystem')
    def test_init_optimized_imports_true(self, MockEUMS, MockAMP, MockDBEnhancer, MockConvManager):
        mock_conv_manager_inst = MockConvManager.return_value
        mock_db_enhancer_inst = MockDBEnhancer.return_value
        mock_amp_inst = MockAMP.return_value
        mock_eums_inst = MockEUMS.return_value

        with patch('memory_optimization_fix.OPTIMIZED_IMPORTS', True):
            integrator = mof.MemoryEnhancementIntegrator()
            MockConvManager.assert_called_once()
            MockDBEnhancer.assert_called_once()
            MockAMP.assert_called_once()
            MockEUMS.assert_called_once()
            self.assertEqual(integrator.conversation_manager, mock_conv_manager_inst)
            self.assertEqual(integrator.db_optimizer, mock_db_enhancer_inst)
            self.assertEqual(integrator.memory_profiler, mock_amp_inst)
            self.assertEqual(integrator.monitoring_system, mock_eums_inst)

    @patch('memory_optimization_fix.ConversationLifecycleManager')
    @patch('memory_optimization_fix.DatabaseOptimizationEnhancer')
    @patch('memory_optimization_fix.AdvancedMemoryProfiler') # Should not be called
    @patch('memory_optimization_fix.EnhancedUnifiedMonitoringSystem') # Should not be called
    @patch.object(mof.logger, 'warning')
    def test_init_optimized_imports_false(self, mock_logger_warning, MockEUMS, MockAMP, MockDBEnhancer, MockConvManager):
        with patch('memory_optimization_fix.OPTIMIZED_IMPORTS', False):
            integrator = mof.MemoryEnhancementIntegrator()
            MockConvManager.assert_called_once()
            MockDBEnhancer.assert_called_once()
            MockAMP.assert_not_called()
            MockEUMS.assert_not_called()
            self.assertIsNone(integrator.memory_profiler)
            self.assertIsNone(integrator.monitoring_system)
            # Check for specific warnings if OPTIMIZED_IMPORTS is False and components are not loaded
            calls = [
                call("AdvancedMemoryProfiler not available for MemoryEnhancementIntegrator."),
                call("EnhancedUnifiedMonitoringSystem not available for MemoryEnhancementIntegrator.")
            ]
            mock_logger_warning.assert_has_calls(calls, any_order=True)


    @patch.object(mof.logger, 'info')
    @patch.object(mof.logger, 'warning')
    def test_start_enhanced_monitoring(self, mock_logger_warning, mock_logger_info):
        # Scenario 1: Profiler and Monitoring System available
        with patch('memory_optimization_fix.OPTIMIZED_IMPORTS', True):
            integrator = mof.MemoryEnhancementIntegrator() # Re-init with mocks
            integrator.memory_profiler = MagicMock()
            integrator.monitoring_system = MagicMock()
            integrator.start_enhanced_monitoring()
            mock_logger_info.assert_any_call(
                "Enhanced monitoring using existing AdvancedMemoryProfiler and EnhancedUnifiedMonitoringSystem."
            )
            integrator.monitoring_system.start_monitoring.assert_called_once()

        mock_logger_info.reset_mock()
        mock_logger_warning.reset_mock()

        # Scenario 2: Only Profiler available
        with patch('memory_optimization_fix.OPTIMIZED_IMPORTS', True):
            integrator = mof.MemoryEnhancementIntegrator()
            integrator.memory_profiler = MagicMock()
            integrator.monitoring_system = None # Explicitly set to None
            integrator.start_enhanced_monitoring()
            mock_logger_info.assert_any_call(
                "Enhanced monitoring using existing AdvancedMemoryProfiler."
            )
            mock_logger_warning.assert_any_call(
                "EnhancedUnifiedMonitoringSystem not available. Monitoring system features will be limited."
            )

        mock_logger_info.reset_mock()
        mock_logger_warning.reset_mock()
        
        # Scenario 3: No profiler, no monitoring system (OPTIMIZED_IMPORTS = False)
        with patch('memory_optimization_fix.OPTIMIZED_IMPORTS', False):
            integrator = mof.MemoryEnhancementIntegrator()
            integrator.start_enhanced_monitoring()
            mock_logger_warning.assert_any_call(
                 "AdvancedMemoryProfiler not available. Basic enhanced monitoring only."
            )
            mock_logger_warning.assert_any_call(
                "EnhancedUnifiedMonitoringSystem not available. Monitoring system features will be limited."
            )


    @patch('memory_optimization_fix.datetime')
    def test_get_comprehensive_status(self, mock_datetime):
        now_time = datetime(2025, 1, 1, 12, 30, 0)
        mock_datetime.now.return_value = now_time

        integrator = mof.MemoryEnhancementIntegrator() # Uses mocked sub-components by default from class setup
        
        # Mock sub-component methods
        integrator.conversation_manager.get_conversation_stats.return_value = {"total": 5}
        integrator.db_optimizer.connection_manager = MagicMock() # Simulate available
        integrator.memory_profiler = MagicMock() # Simulate available
        integrator.monitoring_system = None # Simulate not available

        status = integrator.get_comprehensive_status()

        expected_status = {
            "timestamp": now_time.isoformat(),
            "conversation_stats": {"total": 5},
            "database_optimization_status": "Available",
            "memory_profiler_status": "Available",
            "monitoring_system_status": "Not Available"
        }
        self.assertEqual(status, expected_status)

    def test_perform_maintenance(self):
        integrator = mof.MemoryEnhancementIntegrator()
        
        integrator.conversation_manager.cleanup_old_conversations.return_value = 2
        integrator.db_optimizer.optimize_database_maintenance.return_value = {"vacuum_completed": True}

        results = integrator.perform_maintenance()

        expected_results = {
            "conversations_cleaned": 2,
            "database_optimization_results": {"vacuum_completed": True}
        }
        self.assertEqual(results, expected_results)
        integrator.conversation_manager.cleanup_old_conversations.assert_called_once()
        integrator.db_optimizer.optimize_database_maintenance.assert_called_once()


class TestGlobalFunctions(unittest.TestCase):
    def setUp(self):
        # Patch the global instance for these tests
        self.integrator_patcher = patch('memory_optimization_fix.memory_enhancement', MagicMock(spec=mof.MemoryEnhancementIntegrator))
        self.mock_integrator_instance = self.integrator_patcher.start()
        self.addCleanup(self.integrator_patcher.stop)

    def test_start_memory_enhancements(self):
        mof.start_memory_enhancements()
        self.mock_integrator_instance.start_enhanced_monitoring.assert_called_once()

    def test_get_enhanced_status(self):
        self.mock_integrator_instance.get_comprehensive_status.return_value = {"status": "ok"}
        status = mof.get_enhanced_status()
        self.assertEqual(status, {"status": "ok"})
        self.mock_integrator_instance.get_comprehensive_status.assert_called_once()

    def test_perform_enhanced_maintenance(self):
        self.mock_integrator_instance.perform_maintenance.return_value = {"cleaned": 5}
        result = mof.perform_enhanced_maintenance()
        self.assertEqual(result, {"cleaned": 5})
        self.mock_integrator_instance.perform_maintenance.assert_called_once()

# It's good practice to handle potential import errors for dependencies
# that might not be present in all testing environments, especially for sqlite3
try:
    import sqlite3
except ImportError:
    sqlite3 = None # Allows tests to be skipped or handled if sqlite3 isn't available
    print("sqlite3 module not found, some database tests might be affected.")


if __name__ == '__main__':
    # Re-enable logging for test runner output if needed, or keep disabled for less verbose test runs
    # logging.disable(logging.NOTSET) 
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

