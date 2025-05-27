"""
Legacy Robust Configuration Management for LangGraph 101 Application.
This module has been updated to use the unified configuration system.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from core.config import UnifiedConfig, get_config, ConfigError as UnifiedConfigError

# Configure logging
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class AppConfig:
    """Legacy configuration dataclass - now uses unified configuration system."""
    # Required API Keys
    api_key: str
    tavily_api_key: str
    
    # Optional API Keys
    elevenlabs_api_key: Optional[str] = None
    dalle_api_key: Optional[str] = None
    stabilityai_api_key: Optional[str] = None
    pixabay_api_key: Optional[str] = None
    pexels_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    assemblyai_api_key: Optional[str] = None
    youtube_data_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None
    
    # Email Configuration
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_from: Optional[str] = None
    alert_email_to: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Error Monitoring
    error_threshold: int = 10
    error_window_hours: int = 24
    
    # Application Settings
    debug: bool = False
    log_level: str = "INFO"
      # Development mode flag for flexible validation
    development_mode: bool = False
    
    @classmethod
    def from_unified_config(cls, unified_config: UnifiedConfig) -> 'AppConfig':
        """Create AppConfig from unified configuration."""
        return cls(
            api_key=unified_config.api.gemini_api_key or "",
            tavily_api_key=unified_config.api.tavily_api_key or "",
            elevenlabs_api_key=unified_config.api.elevenlabs_api_key,
            dalle_api_key=unified_config.api.stability_api_key,  # Using stability for DALL-E compatibility
            stabilityai_api_key=unified_config.api.stability_api_key,
            pixabay_api_key=unified_config.api.pixabay_api_key,
            pexels_api_key=unified_config.api.pexels_api_key,
            deepgram_api_key=unified_config.api.deepgram_api_key,
            assemblyai_api_key=unified_config.api.assemblyai_api_key,
            youtube_data_api_key=unified_config.api.youtube_data_api_key,
            news_api_key=unified_config.api.news_api_key,
            openweather_api_key=unified_config.api.openweather_api_key,
            smtp_server=getattr(unified_config, 'email', {}).get('smtp_server', None) if hasattr(unified_config, 'email') else None,
            smtp_port=getattr(unified_config, 'email', {}).get('smtp_port', 587) if hasattr(unified_config, 'email') else 587,
            smtp_username=getattr(unified_config, 'email', {}).get('username', None) if hasattr(unified_config, 'email') else None,
            smtp_password=getattr(unified_config, 'email', {}).get('password', None) if hasattr(unified_config, 'email') else None,
            alert_email_from=getattr(unified_config, 'email', {}).get('sender', None) if hasattr(unified_config, 'email') else None,
            alert_email_to=getattr(unified_config, 'email', {}).get('sender', None) if hasattr(unified_config, 'email') else None,  # Using sender as fallback
            smtp_use_tls=getattr(unified_config, 'email', {}).get('use_tls', True) if hasattr(unified_config, 'email') else True,
            debug=unified_config.app.debug,
            log_level=unified_config.logging.level,
            development_mode=unified_config.app.environment == "development"
        )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # In development mode, allow placeholder API keys with warnings
        if self.development_mode:
            if not self.api_key or self.api_key == "your_gemini_api_key_here":
                logger.warning("API_KEY (Google Gemini) is using placeholder value - this is OK for development but not for production")
            
            if not self.tavily_api_key or self.tavily_api_key == "your_tavily_api_key_here":
                logger.warning("TAVILY_API_KEY is using placeholder value - this is OK for development but not for production")
        else:
            # Production mode - strict validation
            if not self.api_key or self.api_key == "your_gemini_api_key_here":
                raise ConfigError("API_KEY (Google Gemini) is required. Please set it in your .env file.")
            
            if not self.tavily_api_key or self.tavily_api_key == "your_tavily_api_key_here":
                raise ConfigError("TAVILY_API_KEY is required. Please set it in your .env file.")


def load_config_robust() -> AppConfig:
    """
    Load configuration using the unified configuration system.
    
    Returns:
        AppConfig object with all configuration values
        
    Raises:
        ConfigError: If required configuration is missing or invalid
    """
    try:
        # Get unified configuration
        unified_config = get_config()
        
        # Create legacy AppConfig from unified config
        app_config = AppConfig.from_unified_config(unified_config)
        
        logger.info("Configuration loaded successfully using unified system")
        return app_config
        
    except UnifiedConfigError as e:
        logger.error(f"Unified configuration error: {e}")
        raise ConfigError(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise ConfigError(f"Failed to load configuration: {e}")


def get_config_dict() -> Dict[str, Any]:
    """
    Get configuration as a dictionary for backward compatibility.
    
    Returns:
        Dictionary representation of configuration
    """
    try:
        app_config = load_config_robust()
        
        return {
            'api_key': app_config.api_key,
            'tavily_api_key': app_config.tavily_api_key,
            'elevenlabs_api_key': app_config.elevenlabs_api_key,
            'dalle_api_key': app_config.dalle_api_key,
            'stabilityai_api_key': app_config.stabilityai_api_key,
            'pixabay_api_key': app_config.pixabay_api_key,
            'pexels_api_key': app_config.pexels_api_key,
            'deepgram_api_key': app_config.deepgram_api_key,
            'assemblyai_api_key': app_config.assemblyai_api_key,
            'youtube_data_api_key': app_config.youtube_data_api_key,
            'news_api_key': app_config.news_api_key,
            'openweather_api_key': app_config.openweather_api_key,
            'smtp_server': app_config.smtp_server,
            'smtp_port': app_config.smtp_port,
            'smtp_username': app_config.smtp_username,
            'smtp_password': app_config.smtp_password,
            'alert_email_from': app_config.alert_email_from,
            'alert_email_to': app_config.alert_email_to,
            'smtp_use_tls': app_config.smtp_use_tls,
            'error_threshold': app_config.error_threshold,
            'error_window_hours': app_config.error_window_hours,
            'debug': app_config.debug,
            'log_level': app_config.log_level,
            'development_mode': app_config.development_mode,
        }
    except ConfigError:
        raise
    except Exception as e:
        logger.error(f"Error creating config dictionary: {e}")
        raise ConfigError(f"Failed to create config dictionary: {e}")
        raise ConfigError(f"Failed to create config dictionary: {e}")


# Backward compatibility functions
def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required keys
        required_keys = ['api_key', 'tavily_api_key']
        for key in required_keys:
            if not config.get(key):
                logger.error(f"Required configuration key missing: {key}")
                return False
        
        logger.info("Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return False


if __name__ == "__main__":
    try:
        config = load_config_robust()
        logger.info("Configuration loaded successfully")
        logger.info(f"Development mode: {config.development_mode}")
        logger.info(f"Log level: {config.log_level}")
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


# Legacy function for backward compatibility
def load_config() -> Dict[str, Any]:
    """
    Legacy configuration loading function for backward compatibility.
    
    Returns:
        Dictionary containing configuration values
    """
    try:
        config = load_config_robust()
        
        # Convert to dictionary for legacy compatibility
        return {
            "api_key": config.api_key,
            "tavily_api_key": config.tavily_api_key,
            "elevenlabs_api_key": config.elevenlabs_api_key,
            "dalle_api_key": config.dalle_api_key,
            "stabilityai_api_key": config.stabilityai_api_key,
            "pixabay_api_key": config.pixabay_api_key,
            "pexels_api_key": config.pexels_api_key,
            "deepgram_api_key": config.deepgram_api_key,
            "assemblyai_api_key": config.assemblyai_api_key,
            "youtube_data_api_key": config.youtube_data_api_key,
            "news_api_key": config.news_api_key,
            "openweather_api_key": config.openweather_api_key,
            "smtp_server": config.smtp_server,
            "smtp_port": config.smtp_port,
            "smtp_username": config.smtp_username,
            "smtp_password": config.smtp_password,
            "alert_email_from": config.alert_email_from,
            "alert_email_to": config.alert_email_to,
            "smtp_use_tls": config.smtp_use_tls,
            "error_threshold": config.error_threshold,
            "error_window_hours": config.error_window_hours,
            "debug": config.debug,
            "log_level": config.log_level
        }
    
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Configuration loading failed: {e}")


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config_robust()
        print("✅ Configuration loaded successfully")
        print(f"Debug mode: {config.debug}")
        print(f"Log level: {config.log_level}")
        print(f"API Key configured: {'Yes' if config.api_key else 'No'}")
        print(f"Tavily API Key configured: {'Yes' if config.tavily_api_key else 'No'}")
    except ConfigError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
