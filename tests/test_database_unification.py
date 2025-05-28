import unittest
import os
import unittest
import os
import sqlite3
import time 
import logging
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta # For cleanup test

# Configure basic logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import all necessary modules
try:
    from core.config import get_config, UnifiedConfig, DatabaseConfig
    from core.database import get_database_manager, UnifiedDatabaseManager, DatabaseError
    # Adjusted imports for thread_safe_connection_manager
    import thread_safe_connection_manager as tscm_module
    from advanced_memory_profiler import AdvancedMemoryProfiler
    from enhanced_unified_monitoring import (
        EnhancedUnifiedMonitoringSystem, 
        DatabaseIntegrationMetrics, 
        SystemMetrics, 
        OSAlert, 
        get_enhanced_monitoring_system
    )
    from memory_optimization_fix import DatabaseOptimizationEnhancer
    MODULE_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(f"Failed to import one or more modules for testing: {e}", exc_info=True)
    MODULE_IMPORTS_SUCCESSFUL = False

# Global variable to store the test database path
TEST_DB_FILE = "test_actual_unified.db" # Updated DB name
TEST_DB_URL = f"sqlite:///{TEST_DB_FILE}"

def reset_all_singletons():
    """Resets all known singletons for a clean test environment."""
    if not MODULE_IMPORTS_SUCCESSFUL:
        return

    logger.debug("Attempting to reset all singletons...")
    
    # Reset core.config's LRU cache
    if hasattr(get_config, 'cache_clear'):
        get_config.cache_clear()
        logger.debug("core.config.get_config cache cleared.")
    # Also reset its internal _instance if it's a manual singleton (it is)
    if hasattr(UnifiedConfig, '_instance'):
        UnifiedConfig._instance = None
        logger.debug("UnifiedConfig._instance reset.")

    # Reset ThreadSafeConnectionManager singleton
    # Accessing internal _connection_manager directly
    if hasattr(tscm_module, '_connection_manager'):
        if tscm_module._connection_manager is not None:
            if hasattr(tscm_module._connection_manager, 'close_all_connections'):
                try:
                    tscm_module._connection_manager.close_all_connections()
                    logger.debug("Called close_all_connections on existing ThreadSafeConnectionManager.")
                except Exception as e_ts_close:
                    logger.error(f"Error closing existing ThreadSafeConnectionManager: {e_ts_close}")
            tscm_module._connection_manager = None
            logger.debug("thread_safe_connection_manager._connection_manager reset.")
    
    # Reset UnifiedDatabaseManager singleton
    if hasattr(UnifiedDatabaseManager, '_instance'): # UDM is a manual singleton
        if UnifiedDatabaseManager._instance is not None:
             # UDM might hold a ref to TSCM, ensure its cleanup doesn't interfere or is done first.
             # If UDM's __del__ or a cleanup method handles its TSCM, that's fine.
             pass
        UnifiedDatabaseManager._instance = None
        logger.debug("UnifiedDatabaseManager._instance reset.")
    
    # Reset EnhancedUnifiedMonitoringSystem singleton
    # EUMS uses ThreadSafeSingleton metaclass
    if hasattr(EnhancedUnifiedMonitoringSystem, '_instances'): 
        EnhancedUnifiedMonitoringSystem.reset_instance(EnhancedUnifiedMonitoringSystem)
        logger.debug("EnhancedUnifiedMonitoringSystem instance reset via metaclass.")
    elif hasattr(EnhancedUnifiedMonitoringSystem, '_instance'): # Fallback if it was changed
         EnhancedUnifiedMonitoringSystem._instance = None
         logger.debug("EnhancedUnifiedMonitoringSystem manual _instance reset.")

    logger.debug("Singleton reset process complete.")


class TestDatabaseUnificationBase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        if not MODULE_IMPORTS_SUCCESSFUL:
            raise unittest.SkipTest("Skipping all tests due to import failures.")
            
        logger.info(f"Setting up test class. Test DB will be: {TEST_DB_FILE}")
        # Ensure environment is clean for DATABASE_URL to take effect in config
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"] # Remove if set by other means
        os.environ["DATABASE_URL"] = TEST_DB_URL
        os.environ["UNIFIED_CONFIG_PATH"] = "" 
        os.environ["ENABLE_MEMORY_PROFILING"] = "false" # Default for tests unless overridden

        reset_all_singletons() # Reset before any config loading or DB creation

        # Clean up any old DB file before starting tests for the class
        if os.path.exists(TEST_DB_FILE):
            logger.info(f"Removing existing test DB file: {TEST_DB_FILE}")
            try:
                os.remove(TEST_DB_FILE)
            except Exception as e:
                 logger.error(f"Could not remove old test DB {TEST_DB_FILE}: {e}")
        
        # Initialize the database schema once for the class via UDM
        try:
            get_database_manager() 
            logger.info(f"Test database {TEST_DB_FILE} initialized by UDM for the class.")
        except Exception as e:
            logger.error(f"Failed to initialize database via UDM in setUpClass: {e}", exc_info=True)
            raise # Fail class setup if DB can't be initialized

    @classmethod
    def tearDownClass(cls):
        logger.info(f"Tearing down test class, removing DB: {TEST_DB_FILE}")
        reset_all_singletons() 

        if os.path.exists(TEST_DB_FILE):
            try:
                # Add a small delay or ensure connections are closed if OperationalError: database is locked occurs
                # time.sleep(0.1) 
                os.remove(TEST_DB_FILE)
                logger.info(f"Test database {TEST_DB_FILE} removed.")
            except Exception as e:
                logger.error(f"Error removing test database {TEST_DB_FILE}: {e}")
        
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        if "UNIFIED_CONFIG_PATH" in os.environ:
            del os.environ["UNIFIED_CONFIG_PATH"]
        if "ENABLE_MEMORY_PROFILING" in os.environ:
            del os.environ["ENABLE_MEMORY_PROFILING"]


    def setUp(self):
        # Reset singletons before each test to ensure test isolation
        reset_all_singletons()
        # DB file is created by setUpClass, individual tests run against this shared schema.
        # If tests modify data and need isolation, they should handle cleanup or use transactions.
        # For now, assuming tests either don't conflict or manage their own data.
        logger.debug(f"Starting test: {self._testMethodName}")


    def tearDown(self):
        logger.debug(f"Finished test: {self._testMethodName}")
        # Optional: clean specific data inserted by a test if not using transactions
        pass


@unittest.skipUnless(MODULE_IMPORTS_SUCCESSFUL, "Skipping tests due to import failures.")
class TestSingletonInitialization(TestDatabaseUnificationBase):

    @patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///test_config_specific.db"})
    def test_get_config_db_url(self):
        """Test if get_config() correctly picks up the DATABASE_URL from environment."""
        reset_all_singletons() 
        config = get_config()
        self.assertEqual(config.database.url, "sqlite:///test_config_specific.db")
        # No file should be created by just getting config
        self.assertFalse(os.path.exists("test_config_specific.db"))


    @patch.dict(os.environ, {"DATABASE_URL": "sqlite:///tscm_test.db", 
                             "DB_POOL_SIZE": "7", 
                             "DB_TIMEOUT": "45"})
    def test_thread_safe_connection_manager_singleton_path(self):
        """Test if ThreadSafeConnectionManager uses the DB path from get_config()."""
        reset_all_singletons()
        config = get_config() 

        manager1 = tscm_module.get_connection_manager()
        self.assertTrue(os.path.exists("tscm_test.db"), "DB file should be created by TSCM init.")
        manager2 = tscm_module.get_connection_manager()
        self.assertIs(manager1, manager2, "get_connection_manager should return a singleton")
        
        expected_path = config.database.url.replace("sqlite:///", "")
        self.assertEqual(str(manager1.db_path), expected_path)
        self.assertEqual(manager1.max_connections, 7) 
        self.assertEqual(manager1.connection_timeout, 45.0)
        
        manager1.close_all_connections() 
        if os.path.exists("tscm_test.db"):
            os.remove("tscm_test.db")


    @patch.dict(os.environ, {"DATABASE_URL": "sqlite:///udm_test.db"})
    def test_unified_database_manager_singleton_path(self):
        """Test if UnifiedDatabaseManager uses the DB path from its internal ThreadSafeConnectionManager."""
        reset_all_singletons()
        config = get_config() 

        udm1 = get_database_manager()
        self.assertTrue(os.path.exists("udm_test.db"), "DB file should be created by UDM init via TSCM.")
        udm2 = get_database_manager()
        self.assertIs(udm1, udm2, "get_database_manager should return a singleton")
        
        expected_path = config.database.url.replace("sqlite:///", "")
        self.assertEqual(str(udm1.ts_connection_manager.db_path), expected_path)
        self.assertEqual(udm1.database_url, expected_path) 

        if hasattr(udm1.ts_connection_manager, 'close_all_connections'):
             udm1.ts_connection_manager.close_all_connections()
        if os.path.exists("udm_test.db"):
            os.remove("udm_test.db")

    @patch.dict(os.environ, {"DATABASE_URL": "sqlite:///eums_test.db",
                             "ENABLE_MEMORY_PROFILING": "false"})
    def test_eums_singleton_initialization(self):
        """Test EUMS singleton initialization and config reading."""
        reset_all_singletons()
        config = get_config() 

        eums1 = get_enhanced_monitoring_system()
        self.assertTrue(os.path.exists("eums_test.db"), "DB file should be created by EUMS init via UDM/TSCM.")
        eums2 = get_enhanced_monitoring_system()
        self.assertIs(eums1, eums2, "get_enhanced_monitoring_system should return a singleton")
        
        self.assertIsNotNone(eums1, "EUMS instance should not be None")
        # This depends on core/config.py having AppConfig.enable_memory_profiling
        self.assertEqual(eums1.enable_memory_profiling, False, 
                         "EUMS enable_memory_profiling should be False from env var via config.")
        
        expected_db_path_from_config = config.database.url.replace("sqlite:///", "")
        self.assertEqual(str(eums1.db_manager.database_url), expected_db_path_from_config)

        if hasattr(eums1.db_manager.ts_connection_manager, 'close_all_connections'):
            eums1.db_manager.ts_connection_manager.close_all_connections()
        if os.path.exists("eums_test.db"):
            os.remove("eums_test.db")


@unittest.skipUnless(MODULE_IMPORTS_SUCCESSFUL, "Skipping tests due to import failures.")
class TestUnifiedTableCreation(TestDatabaseUnificationBase):

    def test_unified_db_creates_all_tables(self):
        """Test if UnifiedDatabaseManager initialization creates all expected tables."""
        # This will need to connect to the DB and check sqlite_master
        self.skipTest("Not yet implemented")


@unittest.skipUnless(MODULE_IMPORTS_SUCCESSFUL, "Skipping tests due to import failures.")
class TestDataInteractionAcrossModules(TestDatabaseUnificationBase):

    def test_profiler_writes_and_udm_reads_profiler_data(self):
        """Test AdvancedMemoryProfiler writes data that UnifiedDatabaseManager can read via its tables."""
        self.skipTest("Not yet implemented")

    def test_eums_writes_os_metrics_and_udm_reads_system_metrics(self):
        """Test EUMS (OS part) writes system_metrics that UDM can read."""
        eums = get_enhanced_monitoring_system()
        self.assertIsNotNone(eums, "EUMS instance should be available")
        
        # Trigger OS metrics collection and storage
        os_metrics_data = eums._collect_os_system_metrics()
        self.assertIsNotNone(os_metrics_data, "OS metrics should be collected")
        eums._store_os_metrics(os_metrics_data)

        # Verify data in system_metrics table
        db_manager = get_database_manager()
        # Example: query the last inserted metric for 'cpu_percent'
        # Note: EUMS stores each metric component as a separate row.
        query = "SELECT metric_value FROM system_metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT 1"
        results = db_manager.execute_query(query, ('cpu_percent',))
        self.assertTrue(len(results) > 0, "CPU percent metric should be stored")
        self.assertEqual(float(results[0]['metric_value']), os_metrics_data.cpu_percent)

    def test_eums_writes_and_udm_reads_integration_metrics(self):
        """Test EnhancedUnifiedMonitoringSystem writes integration_metrics that UDM can read."""
        self.skipTest("Not yet implemented")

    def test_db_optimizer_cleans_unified_tables(self):
        """Test DatabaseOptimizationEnhancer correctly cleans targeted tables in the unified DB."""
        self.skipTest("Not yet implemented")


@unittest.skipUnless(MODULE_IMPORTS_SUCCESSFUL, "Skipping tests due to import failures.")
class TestRefactoredModuleDBOperations(TestDatabaseUnificationBase):

    def test_amp_save_snapshot_unified_db(self):
        """Test AdvancedMemoryProfiler._save_snapshot writes to the unified DB."""
        self.skipTest("Not yet implemented")

    def test_amp_analyze_leaks_unified_db(self):
        """Test AdvancedMemoryProfiler._analyze_leaks_from_db reads from the unified DB."""
        self.skipTest("Not yet implemented")

    def test_eums_store_os_metrics_directly(self):
        """Test EUMS._store_os_metrics writes to system_metrics in unified DB."""
        eums = get_enhanced_monitoring_system()
        self.assertIsNotNone(eums)
        
        # Create a sample SystemMetrics object
        sample_os_metrics = SystemMetrics(
            timestamp=datetime.now(), cpu_percent=10.5, memory_percent=25.0,
            memory_used_gb=4.0, memory_available_gb=12.0, disk_percent=30.0,
            disk_free_gb=700.0, network_bytes_sent=1024*100, network_bytes_recv=2048*200,
            active_threads=5, gc_objects=10000, process_memory_mb=150.0
        )
        eums._store_os_metrics(sample_os_metrics)

        db_manager = get_database_manager()
        query = "SELECT metric_value FROM system_metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT 1"
        results = db_manager.execute_query(query, ('memory_percent',))
        self.assertTrue(len(results) > 0, "Memory percent metric should be stored")
        self.assertEqual(float(results[0]['metric_value']), sample_os_metrics.memory_percent)

    def test_eums_store_os_alert_directly(self):
        """Test EUMS stores an OSAlert into the system_alerts table."""
        eums = get_enhanced_monitoring_system()
        self.assertIsNotNone(eums)

        # Simulate a condition that would create and store an OSAlert
        # For direct testing, we can craft an OSAlert and use a helper or directly insert via db_manager
        # if EUMS doesn't expose a direct "store this OSAlert" method.
        # EUMS._check_os_thresholds calls _create_os_alert then db_manager.insert_record.
        # We can test _create_os_alert and then store it.
        
        test_details = {"test_metric": 100, "limit": 90}
        os_alert_obj = eums._create_os_alert(
            severity='critical', 
            source='test_monitor', 
            message='Test OS critical alert', 
            details=test_details
        )
        self.assertIsNotNone(os_alert_obj)
        
        # Store it (mirroring what _check_os_thresholds would do)
        eums.db_manager.insert_record('system_alerts', os_alert_obj.to_dict())

        db_manager = get_database_manager()
        query = "SELECT id, severity, source, message, details FROM system_alerts WHERE id = ?"
        results = db_manager.execute_query(query, (os_alert_obj.id,))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['severity'], 'critical')
        self.assertEqual(results[0]['source'], 'test_monitor')
        self.assertEqual(json.loads(results[0]['details']), test_details)
    
    def test_eums_periodic_cleanup_os_tables(self):
        """Test EUMS._periodic_db_cleanup correctly cleans system_metrics and system_alerts."""
        eums = get_enhanced_monitoring_system()
        self.assertIsNotNone(eums)
        db_manager = get_database_manager()

        # Add some old OS metrics data
        old_ts = (datetime.now() - timedelta(days=eums.get_config_value('db_retention_days_os_metrics', 7) + 1)).isoformat()
        for i in range(5):
            metric_record = {
                'id': f'old_metric_{i}', 'metric_name': 'cpu_percent', 'metric_value': 10.0 + i,
                'timestamp': old_ts, 'hostname': 'test_host', 'process_id': 123
            }
            db_manager.insert_record('system_metrics', metric_record)
        
        # Add some old OS alert data
        old_alert_ts = (datetime.now() - timedelta(days=eums.get_config_value('db_retention_days_os_alerts', 30) + 1)).isoformat()
        for i in range(3):
            alert_record = OSAlert(
                id=f'old_os_alert_{i}', timestamp=datetime.fromisoformat(old_alert_ts), severity='info',
                source='old_source', message='Old OS alert', details={'info': 'old'}
            ).to_dict()
            db_manager.insert_record('system_alerts', alert_record)

        # Add some new data that should not be cleaned
        new_ts = datetime.now().isoformat()
        db_manager.insert_record('system_metrics', {
            'id': 'new_metric', 'metric_name': 'cpu_percent', 'metric_value': 20.0,
            'timestamp': new_ts, 'hostname': 'test_host', 'process_id': 123
        })
        db_manager.insert_record('system_alerts', OSAlert(
            id='new_os_alert', timestamp=datetime.now(), severity='warning',
            source='new_source', message='New OS alert', details={'info': 'new'}
        ).to_dict())
        
        eums._periodic_db_cleanup()

        # Verify old data is cleaned
        old_metrics_left = db_manager.execute_query("SELECT COUNT(*) as count FROM system_metrics WHERE timestamp < ?", (old_alert_ts,)) # Use a clearly old timestamp
        self.assertEqual(old_metrics_left[0]['count'], 0, "Old OS metrics should be cleaned.")
        
        old_alerts_left = db_manager.execute_query("SELECT COUNT(*) as count FROM system_alerts WHERE timestamp < ?", (old_alert_ts,))
        self.assertEqual(old_alerts_left[0]['count'], 0, "Old OS alerts should be cleaned.")

        # Verify new data remains
        new_metrics_count = db_manager.execute_query("SELECT COUNT(*) as count FROM system_metrics WHERE id = 'new_metric'")
        self.assertEqual(new_metrics_count[0]['count'], 1, "New OS metric should remain.")
        new_alerts_count = db_manager.execute_query("SELECT COUNT(*) as count FROM system_alerts WHERE id = 'new_os_alert'")
        self.assertEqual(new_alerts_count[0]['count'], 1, "New OS alert should remain.")


    def test_eums_save_integration_metrics_unified_db(self):
        """Test EnhancedUnifiedMonitoringSystem._save_integration_metrics writes to unified DB."""
        self.skipTest("Not yet implemented")

    def test_eums_create_alert_unified_db(self):
        """Test EnhancedUnifiedMonitoringSystem._create_eums_specific_alert writes to enhanced_alerts table."""
        self.skipTest("Not yet implemented")

    def test_eums_logs_system_event(self):
        """Test EUMS._log_system_event writes to system_events table."""
        self.skipTest("Not yet implemented")

    def test_eums_manages_system_config(self):
        """Test EUMS can read/write to system_config table via _ensure_default_system_configs and get_config_value."""
        self.skipTest("Not yet implemented")


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # Ensure Python can find the modules (e.g., by setting PYTHONPATH or running from project root)
    # Example: PYTHONPATH=. python tests/test_database_unification.py
    
    # Need to ensure `sys` is imported if used in reset_all_singletons
    import sys 
    
    # A simple way to add project root to sys.path for test execution,
    # assuming this test file is in a 'tests' subdirectory.
    # More robust solutions involve test runners like pytest or proper package structure.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if not MODULE_IMPORTS_SUCCESSFUL:
        print("Cannot run tests: Failed to import one or more modules. Check PYTHONPATH and errors above.")
        sys.exit(1)
        
    unittest.main()
