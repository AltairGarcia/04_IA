"""
Resilient Storage module for LangGraph 101 project.

This module provides resilient storage capabilities for analytics data,
ensuring data integrity and backup functionality.
"""
import os
import json
import logging
import shutil
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, Any, List, Optional, Union, Callable
from cryptography.fernet import Fernet

# Configure logging
logger = logging.getLogger(__name__)

class ResilientStorage:
    """Resilient storage system with backup and recovery features."""

    def __init__(self, base_path: str, backup_interval_hours: int = 24):
        """Initialize the resilient storage system.

        Args:
            base_path: Base directory for data storage
            backup_interval_hours: Interval between automated backups (in hours)
        """
        self.base_path = base_path
        self.backup_path = os.path.join(base_path, "backups")
        self.temp_path = os.path.join(base_path, "temp")
        self.backup_interval = timedelta(hours=backup_interval_hours)
        self.last_backup_file = os.path.join(self.backup_path, "last_backup.txt")
        self.lock = threading.RLock()
        self._backup_thread = None
        self._stop_event = threading.Event()

        # Create necessary directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)

        logger.info(f"Initialized resilient storage at {self.base_path}")

    def _get_file_path(self, filename: str) -> str:
        """Get the full path for a file.

        Args:
            filename: Name of the file

        Returns:
            Full path to the file
        """
        # Ensure the filename doesn't try to navigate outside the base path
        if ".." in filename:
            raise ValueError("Invalid filename - cannot contain '..'")

        return os.path.join(self.base_path, filename)

    def save_data(self, filename: str, data: Any) -> bool:
        """Save data to a file with atomic write.

        Args:
            filename: Name of the file
            data: Data to save (must be JSON-serializable)

        Returns:
            True if successful, False otherwise
        """
        file_path = self._get_file_path(filename)
        temp_file = os.path.join(self.temp_path, f"{filename}.{int(time.time())}.tmp")

        with self.lock:
            try:
                # First write to a temporary file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                # Then atomically replace the target file
                shutil.move(temp_file, file_path)

                logger.debug(f"Successfully saved data to {filename}")
                return True

            except Exception as e:
                logger.error(f"Failed to save data to {filename}: {str(e)}")

                # Clean up the temp file if it exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                return False

    def load_data(self, filename: str, default: Any = None) -> Any:
        """Load data from a file with fallback to backup.

        Args:
            filename: Name of the file
            default: Default value if file doesn't exist or can't be loaded

        Returns:
            Loaded data or default value
        """
        file_path = self._get_file_path(filename)

        with self.lock:
            # Try to load from the main file
            data = self._load_file(file_path)

            # If main file failed, try the latest backup
            if data is None:
                logger.warning(f"Failed to load {filename}, attempting to restore from backup")
                backup_file = self._find_latest_backup(filename)

                if backup_file:
                    data = self._load_file(backup_file)

                    # If backup loaded successfully, restore it as the main file
                    if data is not None:
                        logger.info(f"Restoring {filename} from backup")
                        try:
                            # Ensure directory exists before copying
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            shutil.copy2(backup_file, file_path)
                        except Exception as e:
                            logger.error(f"Failed to restore backup: {str(e)}")

            # Return data or default
            return data if data is not None else default

    def _load_file(self, filepath: str) -> Optional[Any]:
        """Load data from a file.

        Args:
            filepath: Path to the file

        Returns:
            Loaded data or None if failed
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    return json.load(f)
                else:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None

    def create_backup(self) -> bool:
        """Create a backup of all data files.

        Returns:
            True if successful, False otherwise
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.backup_path, timestamp)

        with self.lock:
            try:
                # Create backup directory
                os.makedirs(backup_dir, exist_ok=True)

                # Copy all files (excluding backups and temp directories)
                files_backed_up = 0
                for item in os.listdir(self.base_path):
                    src_path = os.path.join(self.base_path, item)

                    # Skip directories and non-data files
                    if item in ["backups", "temp"] or os.path.isdir(src_path):
                        continue

                    # Copy the file
                    dst_path = os.path.join(backup_dir, item)
                    shutil.copy2(src_path, dst_path)
                    files_backed_up += 1

                # Record the backup time
                try:
                    # Ensure directory exists before writing
                    os.makedirs(os.path.dirname(self.last_backup_file), exist_ok=True)

                    # Write to a temporary file first
                    temp_file = f"{self.last_backup_file}.tmp"
                    with open(temp_file, 'w') as f:
                        f.write(f"{timestamp}\n{datetime.now().isoformat()}")
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk

                    # Rename for atomic write
                    if os.path.exists(self.last_backup_file):
                        os.replace(temp_file, self.last_backup_file)
                    else:
                        shutil.move(temp_file, self.last_backup_file)
                except Exception as e:
                    logger.error(f"Failed to record backup time: {str(e)}")

                logger.info(f"Created backup at {backup_dir} ({files_backed_up} files)")

                # Clean up old backups
                self._cleanup_old_backups()

                return True

            except Exception as e:
                logger.error(f"Backup failed: {str(e)}")
                return False

    def _find_latest_backup(self, filename: str) -> Optional[str]:
        """Find the latest backup of a specific file.

        Args:
            filename: Name of the file

        Returns:
            Path to the latest backup or None if not found
        """
        try:
            if not os.path.exists(self.backup_path):
                logger.warning(f"Backup path does not exist: {self.backup_path}")
                return None

            # Get all backup directories sorted by name (which includes timestamp)
            backup_dirs = sorted([
                d for d in os.listdir(self.backup_path)
                if os.path.isdir(os.path.join(self.backup_path, d))
            ], reverse=True)

            if not backup_dirs:
                logger.warning(f"No backup directories found in {self.backup_path}")
                return None

            # Find the latest backup containing the file
            for backup_dir in backup_dirs:
                backup_file = os.path.join(self.backup_path, backup_dir, filename)
                if os.path.exists(backup_file):
                    logger.debug(f"Found backup for {filename} in {backup_dir}")
                    return backup_file

            logger.warning(f"No backup found for {filename} in any backup directory")
            return None
        except Exception as e:
            logger.error(f"Error finding latest backup for {filename}: {str(e)}")
            return None

    def _cleanup_old_backups(self, max_backups: int = 10) -> None:
        """Clean up old backups, keeping only the most recent ones.

        Args:
            max_backups: Maximum number of backups to keep
        """
        if not os.path.exists(self.backup_path):
            return

        # Get all backup directories sorted by name (which includes timestamp)
        backup_dirs = sorted([
            d for d in os.listdir(self.backup_path)
            if os.path.isdir(os.path.join(self.backup_path, d))
        ])

        # Remove excess backups
        if len(backup_dirs) > max_backups:
            for old_dir in backup_dirs[:-max_backups]:
                try:
                    shutil.rmtree(os.path.join(self.backup_path, old_dir))
                    logger.info(f"Removed old backup: {old_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove old backup {old_dir}: {str(e)}")

    def start_auto_backup(self) -> None:
        """Start the automatic backup thread."""
        if self._backup_thread and self._backup_thread.is_alive():
            logger.warning("Automatic backup is already running")
            return

        self._stop_event.clear()
        self._backup_thread = threading.Thread(target=self._auto_backup_loop, daemon=True)
        self._backup_thread.start()

        logger.info("Started automatic backup thread")

    def stop_auto_backup(self) -> None:
        """Stop the automatic backup thread."""
        if not self._backup_thread or not self._backup_thread.is_alive():
            logger.warning("Automatic backup is not running")
            return

        self._stop_event.set()
        self._backup_thread.join(timeout=10.0)

        logger.info("Stopped automatic backup thread")

    def _auto_backup_loop(self) -> None:
        """Background thread for automatic backups."""
        logger.info(f"Auto-backup thread started with {self.backup_interval.total_seconds()/3600:.1f}h interval")

        while not self._stop_event.is_set():
            try:
                # Check if backup is due
                should_backup = False

                if os.path.exists(self.last_backup_file):
                    try:
                        with open(self.last_backup_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                timestamp = lines[0].strip()
                                backup_time = datetime.fromisoformat(lines[1].strip())

                                # Check if the interval has passed
                                if datetime.now() - backup_time >= self.backup_interval:
                                    should_backup = True
                            else:
                                logger.warning(f"Invalid format in {self.last_backup_file} - insufficient lines")
                                should_backup = True
                    except (ValueError, IOError, Exception) as e:
                        # If there's any issue reading the file, log it and do a backup
                        logger.warning(f"Error reading {self.last_backup_file}: {str(e)}")
                        should_backup = True
                else:
                    # No record of previous backup
                    logger.info(f"No backup record found at {self.last_backup_file}")
                    should_backup = True

                # Create backup if needed
                if should_backup:
                    self.create_backup()

                # Wait for the next check
                check_interval = 60 * 30  # Check every 30 minutes
                self._stop_event.wait(timeout=check_interval)

            except Exception as e:
                logger.error(f"Error in auto-backup thread: {str(e)}")
                # Wait a bit before retrying to avoid tight loops on persistent errors
                time.sleep(60)

    def recover_file(self, filename: str) -> bool:
        """Attempt to recover a corrupted file from backup.

        Args:
            filename: Name of the file to recover

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Find the latest backup
            backup_file = self._find_latest_backup(filename)

            if not backup_file:
                logger.error(f"No backup found for {filename}")
                return False

            # Restore the file
            try:
                file_path = self._get_file_path(filename)
                shutil.copy2(backup_file, file_path)
                logger.info(f"Recovered {filename} from backup")
                return True
            except Exception as e:
                logger.error(f"Failed to recover {filename}: {str(e)}")
                return False


def generate_encryption_key() -> bytes:
    """Generate a new Fernet encryption key."""
    return Fernet.generate_key()

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data using Fernet symmetric encryption."""
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_data(token: bytes, key: bytes) -> bytes:
    """Decrypt data using Fernet symmetric encryption."""
    f = Fernet(key)
    return f.decrypt(token)


# Singleton instance
_storage_instance = None

def get_storage(base_path: Optional[str] = None) -> ResilientStorage:
    """Get the singleton ResilientStorage instance.

    Args:
        base_path: Base directory for storage (only used on first call)

    Returns:
        ResilientStorage instance
    """
    global _storage_instance

    if _storage_instance is None:
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "analytics_data")

        _storage_instance = ResilientStorage(base_path)

    return _storage_instance


# Functions for saving and loading analytics data
def save_analytics_data(filename: str, data: List[Dict[str, Any]]) -> bool:
    """Save analytics data with resilient storage.

    Args:
        filename: Name of the file (without path)
        data: Analytics data to save

    Returns:
        True if successful, False otherwise
    """
    storage = get_storage()
    return storage.save_data(filename, data)


def load_analytics_data(filename: str) -> List[Dict[str, Any]]:
    """Load analytics data with resilient storage.

    Args:
        filename: Name of the file (without path)

    Returns:
        Analytics data or empty list if file doesn't exist
    """
    storage = get_storage()
    return storage.load_data(filename, default=[])


# Initialize resilient storage and start automatic backups
def initialize_resilient_storage(backup_interval_hours: int = 24) -> ResilientStorage:
    """Initialize resilient storage and start automatic backups.

    Args:
        backup_interval_hours: Interval between backups (in hours)

    Returns:
        ResilientStorage instance
    """
    storage = get_storage()

    # Set backup interval
    storage.backup_interval = timedelta(hours=backup_interval_hours)

    # Create initial backup
    storage.create_backup()

    # Start automatic backups
    storage.start_auto_backup()

    return storage


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)

    # Initialize storage with 1-hour backup interval for testing
    storage = initialize_resilient_storage(backup_interval_hours=1)

    # Example data
    test_data = [
        {"timestamp": datetime.now().isoformat(), "value": "test1"},
        {"timestamp": datetime.now().isoformat(), "value": "test2"}
    ]

    # Save and load test
    save_analytics_data("test_data.json", test_data)
    loaded_data = load_analytics_data("test_data.json")

    print(f"Saved data: {test_data}")
    print(f"Loaded data: {loaded_data}")
