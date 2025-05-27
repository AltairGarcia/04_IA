import unittest
import os
import sqlite3
import json
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path to allow importing analytics_logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Temporarily modify DATABASE_NAME for testing to avoid using the production DB
TEST_DB_DIR = "test_analytics_data"
TEST_DB_NAME = os.path.join(TEST_DB_DIR, "test_analytics.db")

# Must import after setting up test DB path, or mock it before import
# For simplicity here, we'll ensure analytics_logger uses our test path
# by patching its DATABASE_NAME and DATABASE_DIR constants.

# It's cleaner to patch the constants *within* the module being tested.
# So, we'll patch 'analytics_logger.DATABASE_NAME' and 'analytics_logger.DATABASE_DIR'

class TestAnalyticsLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure the test database directory exists."""
        if not os.path.exists(TEST_DB_DIR):
            os.makedirs(TEST_DB_DIR)

    def setUp(self):
        """Ensure a clean database for each test."""
        if os.path.exists(TEST_DB_NAME):
            os.remove(TEST_DB_NAME)
        # Patch the global constants in analytics_logger for the duration of the test
        self.patch_db_name = patch('analytics_logger.DATABASE_NAME', TEST_DB_NAME)
        self.patch_db_dir = patch('analytics_logger.DATABASE_DIR', TEST_DB_DIR)
        self.mock_db_name = self.patch_db_name.start()
        self.mock_db_dir = self.patch_db_dir.start()
        
        # Now import analytics_logger or its functions
        global analytics_logger
        import analytics_logger
        analytics_logger.initialize_database() # Initialize with patched name

    def tearDown(self):
        """Clean up the database file after each test."""
        if os.path.exists(TEST_DB_NAME):
            os.remove(TEST_DB_NAME)
        self.patch_db_name.stop()
        self.patch_db_dir.stop()

    @classmethod
    def tearDownClass(cls):
        """Remove the test database directory after all tests."""
        if os.path.exists(TEST_DB_DIR):
            # Make sure no db file is left before removing dir
            if os.path.exists(TEST_DB_NAME):
                os.remove(TEST_DB_NAME)
            os.rmdir(TEST_DB_DIR)

    def test_database_file_creation(self):
        """Test that the database file is created in the correct directory."""
        self.assertTrue(os.path.exists(TEST_DB_DIR))
        # initialize_database is called in setUp
        self.assertTrue(os.path.exists(TEST_DB_NAME))

    def test_initialize_database(self):
        """Test that the 'events' table is created."""
        # analytics_logger.initialize_database() is called in setUp
        conn = sqlite3.connect(TEST_DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
        table_exists = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(table_exists)

    def test_log_event_minimal_fields(self):
        """Test logging an event with only the required event_type."""
        analytics_logger.log_event(event_type='test_event_minimal')
        conn = sqlite3.connect(TEST_DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE event_type='test_event_minimal'")
        event = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(event)
        self.assertEqual(event['event_type'], 'test_event_minimal')
        self.assertIsNotNone(event['timestamp'])
        self.assertIsNone(event['user_id'])
        self.assertIsNone(event['session_id'])
        self.assertIsNone(event['model_id'])
        self.assertIsNone(event['prompt_template_id'])
        self.assertIsNone(event['details_json'])

    def test_log_event_all_fields(self):
        """Test logging an event with all fields populated."""
        details_data = {'key1': 'value1', 'key2': 100}
        analytics_logger.log_event(
            event_type='test_event_full',
            user_id='user001',
            session_id='sessionXYZ',
            model_id='model-alpha',
            prompt_template_id='template-beta',
            details=details_data
        )

        conn = sqlite3.connect(TEST_DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE event_type='test_event_full'")
        event = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(event)
        self.assertEqual(event['event_type'], 'test_event_full')
        self.assertEqual(event['user_id'], 'user001')
        self.assertEqual(event['session_id'], 'sessionXYZ')
        self.assertEqual(event['model_id'], 'model-alpha')
        self.assertEqual(event['prompt_template_id'], 'template-beta')
        self.assertIsNotNone(event['timestamp'])
        self.assertEqual(json.loads(event['details_json']), details_data)

    def test_log_multiple_events(self):
        """Test logging multiple events and retrieving them."""
        analytics_logger.log_event(event_type='event1', user_id='userA')
        analytics_logger.log_event(event_type='event2', user_id='userB', details={'source': 'test'})

        conn = sqlite3.connect(TEST_DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 2)

if __name__ == '__main__':
    unittest.main()