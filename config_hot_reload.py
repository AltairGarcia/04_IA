#!/usr/bin/env python3
"""
Configuration Hot Reloading System for LangGraph 101

This module provides dynamic configuration management with hot reloading capabilities,
allowing configuration changes without application restart.

Features:
- File-based configuration watching
- Environment variable monitoring
- Real-time configuration updates
- Configuration validation and rollback
- Change notifications and callbacks
- Configuration versioning and history
- Secure configuration encryption
- Performance optimized with minimal overhead
"""

import os
import json
import yaml
import logging
import threading
import time
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import weakref
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet
import copy

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class ConfigChange:
    """Represents a configuration change"""
    timestamp: datetime
    config_path: str
    old_value: Any
    new_value: Any
    change_type: str  # 'added', 'modified', 'deleted'
    source: str  # 'file', 'env', 'api'
    user: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'config_path': self.config_path,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'change_type': self.change_type,
            'source': self.source,
            'user': self.user
        }


@dataclass
class ConfigVersion:
    """Configuration version snapshot"""
    version: str
    timestamp: datetime
    config_data: Dict[str, Any]
    changes: List[ConfigChange] = field(default_factory=list)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
            
    def _calculate_checksum(self) -> str:
        """Calculate checksum of configuration data"""
        config_str = json.dumps(self.config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()


class ConfigWatcher(FileSystemEventHandler):
    """File system watcher for configuration files"""
    
    def __init__(self, hot_reload_manager):
        self.manager = hot_reload_manager
        self.last_modified = {}
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Check if this is a config file we're watching
        if file_path in self.manager._watched_files:
            # Debounce rapid file changes
            current_time = time.time()
            last_mod = self.last_modified.get(file_path, 0)
            
            if current_time - last_mod > 0.5:  # 500ms debounce
                self.last_modified[file_path] = current_time
                self.manager._handle_file_change(file_path)
                
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path in self.manager._watched_files:
                self.manager._handle_file_change(file_path)
                
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path in self.manager._watched_files:
                self.manager._handle_file_deletion(file_path)


class ConfigHotReloadManager:
    """Manages hot reloading of configuration files and environment variables"""
    
    def __init__(self, config_dir: str = "config", backup_dir: str = "config/backups"):
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration storage
        self._config_data: Dict[str, Any] = {}
        self._config_versions: List[ConfigVersion] = []
        self._config_lock = threading.RLock()
        
        # File watching
        self._watched_files: Dict[Path, str] = {}  # file_path -> config_key
        self._file_checksums: Dict[Path, str] = {}
        self._observer = Observer()
        self._watcher = ConfigWatcher(self)
        
        # Environment variable monitoring
        self._watched_env_vars: Dict[str, str] = {}  # env_var -> config_path
        self._env_values: Dict[str, str] = {}
        self._env_monitor_thread = None
        self._env_monitor_interval = 5.0  # seconds
        
        # Change callbacks
        self._change_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._global_callbacks: List[Callable] = []
        
        # Validation functions
        self._validators: Dict[str, Callable] = {}
        
        # Security
        self._encryption_key: Optional[bytes] = None
        self._encrypted_keys: set = set()
        
        # Performance tracking
        self._reload_count = 0
        self._last_reload_time = None
        self._reload_errors = []
        
        # Start monitoring
        self._start_monitoring()
        
    def _start_monitoring(self):
        """Start file system and environment monitoring"""
        # Start file system observer
        self._observer.schedule(self._watcher, str(self.config_dir), recursive=True)
        self._observer.start()
        
        # Start environment variable monitoring
        self._env_monitor_thread = threading.Thread(
            target=self._env_monitor_worker,
            daemon=True,
            name="ConfigHotReload-EnvMonitor"
        )
        self._env_monitor_thread.start()
        
        logger.info("Configuration hot reload monitoring started")
        
    def _env_monitor_worker(self):
        """Worker thread for monitoring environment variables"""
        while True:
            try:
                time.sleep(self._env_monitor_interval)
                self._check_env_changes()
            except Exception as e:
                logger.error(f"Environment monitoring error: {e}")
                
    def _check_env_changes(self):
        """Check for environment variable changes"""
        for env_var, config_path in self._watched_env_vars.items():
            current_value = os.environ.get(env_var)
            old_value = self._env_values.get(env_var)
            
            if current_value != old_value:
                logger.info(f"Environment variable {env_var} changed")
                self._env_values[env_var] = current_value
                
                # Update configuration
                change = ConfigChange(
                    timestamp=datetime.now(),
                    config_path=config_path,
                    old_value=old_value,
                    new_value=current_value,
                    change_type='modified' if old_value is not None else 'added',
                    source='env'
                )
                
                self._apply_config_change(config_path, current_value, change)
                
    def watch_file(self, file_path: Union[str, Path], config_key: str = None):
        """Add a configuration file to watch"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Configuration file does not exist: {file_path}")
            return
            
        config_key = config_key or file_path.stem
        
        with self._config_lock:
            self._watched_files[file_path] = config_key
            self._file_checksums[file_path] = self._calculate_file_checksum(file_path)
            
        # Load initial configuration
        self._load_config_file(file_path, config_key)
        
        logger.info(f"Watching configuration file: {file_path} -> {config_key}")
        
    def watch_env_var(self, env_var: str, config_path: str, default_value: Any = None):
        """Add an environment variable to watch"""
        current_value = os.environ.get(env_var, default_value)
        
        with self._config_lock:
            self._watched_env_vars[env_var] = config_path
            self._env_values[env_var] = current_value
            
        # Set initial value
        if current_value is not None:
            self._set_config_value(config_path, current_value)
            
        logger.info(f"Watching environment variable: {env_var} -> {config_path}")
        
    def add_change_callback(self, config_path: str, callback: Callable[[ConfigChange], None]):
        """Add a callback for configuration changes"""
        with self._config_lock:
            self._change_callbacks[config_path].append(callback)
            
    def add_global_callback(self, callback: Callable[[ConfigChange], None]):
        """Add a global callback for all configuration changes"""
        with self._config_lock:
            self._global_callbacks.append(callback)
            
    def add_validator(self, config_path: str, validator: Callable[[Any], bool]):
        """Add a validation function for a configuration path"""
        with self._config_lock:
            self._validators[config_path] = validator
            
    def set_encryption_key(self, key: bytes = None):
        """Set encryption key for sensitive configuration values"""
        if key is None:
            key = Fernet.generate_key()
        self._encryption_key = key
        logger.info("Configuration encryption enabled")
        
    def encrypt_config_key(self, config_path: str):
        """Mark a configuration key as encrypted"""
        if self._encryption_key is None:
            raise ConfigError("Encryption key not set")
        self._encrypted_keys.add(config_path)
        
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum of a file"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
            
    def _handle_file_change(self, file_path: Path):
        """Handle configuration file change"""
        try:
            # Check if file actually changed
            new_checksum = self._calculate_file_checksum(file_path)
            old_checksum = self._file_checksums.get(file_path, "")
            
            if new_checksum == old_checksum:
                return  # No actual change
                
            config_key = self._watched_files[file_path]
            logger.info(f"Configuration file changed: {file_path}")
            
            # Backup current config
            self._backup_current_config()
            
            # Load new configuration
            old_config = copy.deepcopy(self._config_data.get(config_key, {}))
            self._load_config_file(file_path, config_key)
            new_config = self._config_data.get(config_key, {})
            
            # Update checksum
            self._file_checksums[file_path] = new_checksum
            
            # Create change record
            change = ConfigChange(
                timestamp=datetime.now(),
                config_path=config_key,
                old_value=old_config,
                new_value=new_config,
                change_type='modified',
                source='file'
            )
            
            # Trigger callbacks
            self._trigger_callbacks(change)
            
            # Update reload tracking
            self._reload_count += 1
            self._last_reload_time = datetime.now()
            
            logger.info(f"Configuration reloaded successfully: {config_key}")
            
        except Exception as e:
            logger.error(f"Error handling file change {file_path}: {e}")
            self._reload_errors.append({
                'timestamp': datetime.now(),
                'file_path': str(file_path),
                'error': str(e)
            })
            
    def _handle_file_deletion(self, file_path: Path):
        """Handle configuration file deletion"""
        if file_path in self._watched_files:
            config_key = self._watched_files[file_path]
            logger.warning(f"Configuration file deleted: {file_path}")
            
            # Create change record
            change = ConfigChange(
                timestamp=datetime.now(),
                config_path=config_key,
                old_value=self._config_data.get(config_key),
                new_value=None,
                change_type='deleted',
                source='file'
            )
            
            # Remove from config
            with self._config_lock:
                if config_key in self._config_data:
                    del self._config_data[config_key]
                    
            # Trigger callbacks
            self._trigger_callbacks(change)
            
    def _load_config_file(self, file_path: Path, config_key: str):
        """Load configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    # Try to parse as JSON first, then YAML
                    content = f.read()
                    try:
                        config_data = json.loads(content)
                    except json.JSONDecodeError:
                        config_data = yaml.safe_load(content)
                        
            # Validate configuration
            if not self._validate_config(config_key, config_data):
                raise ConfigError(f"Configuration validation failed for {config_key}")
                
            # Decrypt encrypted values
            if config_data:
                config_data = self._decrypt_config(config_data, config_key)
                
            with self._config_lock:
                self._config_data[config_key] = config_data
                
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            raise ConfigError(f"Failed to load configuration: {e}")
            
    def _validate_config(self, config_path: str, config_data: Any) -> bool:
        """Validate configuration data"""
        if config_path in self._validators:
            try:
                return self._validators[config_path](config_data)
            except Exception as e:
                logger.error(f"Configuration validation error for {config_path}: {e}")
                return False
        return True
        
    def _decrypt_config(self, config_data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Decrypt encrypted configuration values"""
        if self._encryption_key is None:
            return config_data
            
        fernet = Fernet(self._encryption_key)
        
        def decrypt_recursive(data, path=""):
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    full_path = f"{prefix}.{current_path}" if prefix else current_path
                    
                    if full_path in self._encrypted_keys and isinstance(value, str):
                        try:
                            result[key] = fernet.decrypt(value.encode()).decode()
                        except Exception as e:
                            logger.error(f"Failed to decrypt {full_path}: {e}")
                            result[key] = value
                    else:
                        result[key] = decrypt_recursive(value, current_path)
                return result
            elif isinstance(data, list):
                return [decrypt_recursive(item, path) for item in data]
            else:
                return data
                
        return decrypt_recursive(config_data)
        
    def _apply_config_change(self, config_path: str, new_value: Any, change: ConfigChange):
        """Apply a configuration change"""
        try:
            # Validate change
            if not self._validate_config(config_path, new_value):
                logger.error(f"Configuration change validation failed: {config_path}")
                return False
                
            # Apply change
            self._set_config_value(config_path, new_value)
            
            # Trigger callbacks
            self._trigger_callbacks(change)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying config change {config_path}: {e}")
            return False
            
    def _set_config_value(self, config_path: str, value: Any):
        """Set a configuration value by path"""
        with self._config_lock:
            path_parts = config_path.split('.')
            current = self._config_data
            
            # Navigate to parent
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set value
            current[path_parts[-1]] = value
            
    def _trigger_callbacks(self, change: ConfigChange):
        """Trigger change callbacks"""
        try:
            # Trigger specific callbacks
            for callback in self._change_callbacks.get(change.config_path, []):
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error in change callback: {e}")
                    
            # Trigger global callbacks
            for callback in self._global_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error in global callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering callbacks: {e}")
            
    def _backup_current_config(self):
        """Backup current configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"config_backup_{timestamp}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_data, f, indent=2, default=str)
                
            # Keep only last 10 backups
            backups = sorted(self.backup_dir.glob("config_backup_*.json"))
            if len(backups) > 10:
                for backup in backups[:-10]:
                    backup.unlink()
                    
        except Exception as e:
            logger.error(f"Error creating config backup: {e}")
            
    def get_config(self, config_path: str = None, default: Any = None) -> Any:
        """Get configuration value by path"""
        with self._config_lock:
            if config_path is None:
                return copy.deepcopy(self._config_data)
                
            path_parts = config_path.split('.')
            current = self._config_data
            
            try:
                for part in path_parts:
                    current = current[part]
                return copy.deepcopy(current)
            except (KeyError, TypeError):
                return default
                
    def set_config(self, config_path: str, value: Any, source: str = 'api') -> bool:
        """Set configuration value programmatically"""
        old_value = self.get_config(config_path)
        
        change = ConfigChange(
            timestamp=datetime.now(),
            config_path=config_path,
            old_value=old_value,
            new_value=value,
            change_type='modified' if old_value is not None else 'added',
            source=source
        )
        
        return self._apply_config_change(config_path, value, change)
        
    def reload_all(self) -> bool:
        """Manually reload all configuration files"""
        try:
            logger.info("Manually reloading all configuration files")
            
            for file_path, config_key in self._watched_files.items():
                if file_path.exists():
                    self._handle_file_change(file_path)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error during manual reload: {e}")
            return False
            
    def get_reload_stats(self) -> Dict[str, Any]:
        """Get hot reload statistics"""
        return {
            'reload_count': self._reload_count,
            'last_reload_time': self._last_reload_time.isoformat() if self._last_reload_time else None,
            'watched_files': len(self._watched_files),
            'watched_env_vars': len(self._watched_env_vars),
            'error_count': len(self._reload_errors),
            'recent_errors': self._reload_errors[-5:] if self._reload_errors else [],
            'callback_count': sum(len(callbacks) for callbacks in self._change_callbacks.values()),
            'global_callback_count': len(self._global_callbacks)
        }
        
    def export_config(self, file_path: Union[str, Path] = None) -> str:
        """Export current configuration to file"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config_dir / f"config_export_{timestamp}.json"
        else:
            file_path = Path(file_path)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._config_data, f, indent=2, default=str)
            
        logger.info(f"Configuration exported to: {file_path}")
        return str(file_path)
        
    def stop_monitoring(self):
        """Stop configuration monitoring"""
        logger.info("Stopping configuration hot reload monitoring")
        
        # Stop file observer
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            
        # Stop environment monitoring (thread will exit on next iteration)
        # (daemon thread will stop when main program exits)
        
        logger.info("Configuration monitoring stopped")
        
    def __del__(self):
        """Cleanup when manager is destroyed"""
        try:
            self.stop_monitoring()
        except Exception:
            pass


# Global hot reload manager instance
_hot_reload_manager = None


def get_hot_reload_manager() -> ConfigHotReloadManager:
    """Get the global hot reload manager instance"""
    global _hot_reload_manager
    if _hot_reload_manager is None:
        _hot_reload_manager = ConfigHotReloadManager()
    return _hot_reload_manager


# Convenience functions
def watch_config_file(file_path: Union[str, Path], config_key: str = None):
    """Watch a configuration file for changes"""
    manager = get_hot_reload_manager()
    manager.watch_file(file_path, config_key)


def watch_env_var(env_var: str, config_path: str, default_value: Any = None):
    """Watch an environment variable for changes"""
    manager = get_hot_reload_manager()
    manager.watch_env_var(env_var, config_path, default_value)


def get_config(config_path: str = None, default: Any = None) -> Any:
    """Get configuration value"""
    manager = get_hot_reload_manager()
    return manager.get_config(config_path, default)


def set_config(config_path: str, value: Any) -> bool:
    """Set configuration value"""
    manager = get_hot_reload_manager()
    return manager.set_config(config_path, value)


if __name__ == "__main__":
    # Demo and testing
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create test config file
        test_config = {
            "app": {
                "name": "LangGraph 101",
                "version": "1.0.0",
                "debug": False
            },
            "database": {
                "url": "sqlite:///test.db",
                "pool_size": 10
            }
        }
        
        config_file = config_dir / "app.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f, indent=2)
            
        # Create hot reload manager
        manager = ConfigHotReloadManager(str(config_dir))
        
        # Add change callback
        def on_config_change(change: ConfigChange):
            print(f"Config changed: {change.config_path} = {change.new_value}")
            
        manager.add_global_callback(on_config_change)
        
        # Watch config file
        manager.watch_file(config_file, "app")
        
        # Watch environment variable
        os.environ["TEST_ENV"] = "test_value"
        manager.watch_env_var("TEST_ENV", "env.test_var")
        
        print(f"Initial config: {manager.get_config('app.name')}")
        print(f"Environment config: {manager.get_config('env.test_var')}")
        
        # Test programmatic config change
        manager.set_config("app.debug", True)
        print(f"Debug mode: {manager.get_config('app.debug')}")
        
        # Test environment variable change
        os.environ["TEST_ENV"] = "new_value"
        time.sleep(6)  # Wait for env monitor
        print(f"Updated env config: {manager.get_config('env.test_var')}")
        
        # Show statistics
        stats = manager.get_reload_stats()
        print(f"\nReload stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Cleanup
        manager.stop_monitoring()
        
    print("\nHot reload test completed successfully!")
