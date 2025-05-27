"""
Legacy configuration management module for LangGraph 101.
This module has been updated to use the unified configuration system.
"""
import os
import logging
from typing import Dict, Any, Optional
from core.config import UnifiedConfig, get_config, ConfigError

logger = logging.getLogger(__name__)

"""
Legacy configuration management module for LangGraph 101.
This module has been updated to use the unified configuration system.
Provides backward compatibility with the legacy Config class interface.
"""
import os
import logging
from typing import Dict, Any, Optional
from core.config import UnifiedConfig, get_config, ConfigError

logger = logging.getLogger(__name__)

# Configure logger to display INFO level logs
logging.basicConfig(level=logging.INFO)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """Legacy configuration manager class - now uses unified configuration system."""

    _instance = None
    _unified_config: Optional[UnifiedConfig] = None

    def __new__(cls):
        """Singleton implementation."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._unified_config = get_config()
        return cls._instance

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using a dot-separated key path."""
        try:            # Map common legacy paths to unified config
            if key_path.startswith('api_keys.'):
                api_key = key_path.split('.', 1)[1]
                return getattr(self._unified_config.api, api_key, default)
            elif key_path.startswith('application.'):
                attr = key_path.split('.', 1)[1]
                return getattr(self._unified_config.app, attr, default)
            elif key_path.startswith('logging.'):
                attr = key_path.split('.', 1)[1]
                return getattr(self._unified_config.logging, attr, default)
            elif key_path.startswith('database.'):
                attr = key_path.split('.', 1)[1]
                return getattr(self._unified_config.database, attr, default)
            elif key_path.startswith('security.'):
                attr = key_path.split('.', 1)[1]
                return getattr(self._unified_config.security, attr, default)
            elif key_path == 'directories.analytics':
                return os.path.join(BASE_DIR, "analytics_data")
            elif key_path == 'directories.error_logs':
                return os.path.join(BASE_DIR, "error_logs")
            elif key_path == 'directories.content_output':
                return os.path.join(BASE_DIR, "content_output")
            elif key_path == 'directories.performance_cache':
                return os.path.join(BASE_DIR, "performance_cache")
            else:
                # Fallback to default
                return default
        except Exception as e:
            logger.warning(f"Error getting config value for {key_path}: {e}")
            return default

    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value using a dot-separated key path."""
        logger.warning(f"Config.set() is deprecated. Configuration changes should be made through environment variables or .env file.")
        logger.info(f"Attempted to set {key_path} = {value}")

    def save(self) -> None:
        """Save configuration - deprecated in unified system."""
        logger.warning("Config.save() is deprecated. Configuration is managed through environment variables.")

    @property
    def all_configs(self) -> Dict[str, Any]:
        """Return a dictionary representation of the configuration for backward compatibility."""
        try:
            return {
                "application": {
                    "name": self._unified_config.app.name,
                    "version": self._unified_config.app.version,
                    "environment": self._unified_config.app.environment,
                    "base_dir": BASE_DIR,
                },
                "logging": {
                    "level": self._unified_config.logging.level,
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "use_json": False,
                    "log_file": os.path.join(BASE_DIR, "langgraph_system.log"),
                },
                "directories": {
                    "analytics": os.path.join(BASE_DIR, "analytics_data"),
                    "error_logs": os.path.join(BASE_DIR, "error_logs"),
                    "content_output": os.path.join(BASE_DIR, "content_output"),
                    "performance_cache": os.path.join(BASE_DIR, "performance_cache"),
                },                "api_keys": {
                    "gemini": self._unified_config.api.gemini_api_key,
                    "tavily": self._unified_config.api.tavily_api_key,
                    "elevenlabs": self._unified_config.api.elevenlabs_api_key,
                    "stability_ai": self._unified_config.api.stability_api_key,
                    "assemblyai": self._unified_config.api.assemblyai_api_key,
                },
                "email_notification": {
                    "smtp_server": getattr(self._unified_config.email, 'smtp_server', ''),
                    "smtp_port": getattr(self._unified_config.email, 'smtp_port', 587),
                    "username": getattr(self._unified_config.email, 'username', ''),
                    "password": getattr(self._unified_config.email, 'password', ''),
                    "sender": getattr(self._unified_config.email, 'sender', ''),
                    "recipients": [],
                    "check_interval_seconds": 3600,
                },
                "resilient_storage": {
                    "backup_interval_hours": 24,
                    "max_backups": 5,
                },
                "analytics": {
                    "enabled": True,
                    "db_path": os.path.join(BASE_DIR, "analytics_data", "analytics.db"),
                },
                "performance": {
                    "cache_enabled": True,
                    "cache_expiration_seconds": 86400,
                }
            }
        except Exception as e:
            logger.error(f"Error creating legacy config dictionary: {e}")
            return {}


# Create a global instance
config = Config()

# Export the instance's methods as module-level functions
get = config.get
set = config.set
save = config.save
all_configs = config.all_configs

if __name__ == "__main__":
    logger.info("Config Manager using unified configuration system")
    unified_config = get_config()
    logger.info(f"API Keys configured: {bool(unified_config.api_keys.gemini_api_key)}")
    logger.info(f"Database URL: {unified_config.database.url}")
