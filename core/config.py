"""
Unified Configuration Management System for LangGraph 101

This module provides a centralized, secure, and type-safe configuration system
that consolidates all previous configuration approaches into a single source of truth.
"""

import os
import logging
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from functools import lru_cache
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///data/langgraph_101.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    timeout: int = 30
    backup_enabled: bool = True
    backup_path: str = "data/backups"
    migration_enabled: bool = True
    
    def __post_init__(self):
        # Ensure database directory exists
        db_path = Path(self.url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        Path(self.backup_path).mkdir(parents=True, exist_ok=True)


@dataclass 
class APIConfig:
    """API configuration settings"""
    # Core AI APIs
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Specialized APIs
    elevenlabs_api_key: Optional[str] = None
    stability_api_key: Optional[str] = None
    pexels_api_key: Optional[str] = None
    pixabay_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    assemblyai_api_key: Optional[str] = None
    youtube_data_api_key: Optional[str] = None
    news_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None
    
    # API endpoints
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    
    # Timeouts and retries
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Validate that required API keys are present"""
        if not any([self.openai_api_key, self.gemini_api_key, self.google_api_key]):
            logger.warning("No primary AI API key configured. Some features may not work.")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        return getattr(self, f"{provider}_api_key", None)


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_expiration: int = 3600  # 1 hour
    
    # Authentication
    auth_enabled: bool = True
    session_timeout: int = 1800  # 30 minutes
    
    # Rate limiting
    rate_limit_enabled: bool = True
    max_requests_per_hour: int = 1000
    
    # Input validation
    max_input_length: int = 10000
    sanitize_inputs: bool = True
    
    # Encryption
    encryption_enabled: bool = True
    _cipher_suite: Optional[Fernet] = field(default=None, init=False)
    
    def __post_init__(self):
        if not self.secret_key:
            self.secret_key = secrets.token_hex(32)
            logger.warning("Generated new secret key. Set SECRET_KEY environment variable for production.")
        
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_hex(32)
            logger.warning("Generated new JWT secret. Set JWT_SECRET environment variable for production.")
        
        if self.encryption_enabled:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption cipher"""
        try:
            # Derive key from secret
            password = self.secret_key.encode()
            salt = b'langgraph_101_salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self._cipher_suite = Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.encryption_enabled = False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self._cipher_suite:
            return data
        try:
            return self._cipher_suite.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self._cipher_suite:
            return encrypted_data
        try:
            return self._cipher_suite.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/langgraph_101.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True
    
    def __post_init__(self):
        # Ensure log directory exists
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str = "LangGraph 101"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Server settings
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    
    # Data directories
    data_dir: str = "data"
    temp_dir: str = "temp"
    uploads_dir: str = "uploads"
    
    def __post_init__(self):
        # Create necessary directories
        for directory in [self.data_dir, self.temp_dir, self.uploads_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


@dataclass
class LangGraphConfig:
    """LangGraph-specific configuration"""
    max_iterations: int = 10
    default_temperature: float = 0.7
    default_model_id: Optional[str] = None  # Will be set from ModelsConfig
    enable_memory: bool = True
    memory_type: str = "conversation"
    max_memory_size: int = 1000
    
    # Personas and system prompts
    default_persona: str = "assistant"
    personas_dir: str = "config/personas"
    
    def __post_init__(self):
        Path(self.personas_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelDetail:
    """Configuration for a single AI model."""
    model_id: str  # Unique identifier, e.g., "gemini-1.5-pro", "gpt-4o"
    provider: str  # e.g., "google", "openai", "anthropic", "local"
    api_key_env_var: Optional[str] = None # Environment variable name for the API key
    base_url: Optional[str] = None
    # Additional model-specific parameters (e.g., for local models: path, quantization)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelsConfig:
    """Manages configurations for multiple AI models."""
    available_models: List[ModelDetail] = field(default_factory=list)
    default_model_id: Optional[str] = None

    def __post_init__(self):
        # Attempt to load model configurations from environment variable
        models_json_env = os.getenv("MODELS_CONFIG_JSON")
        if models_json_env:
            try:
                models_data = json.loads(models_json_env)
                if "available_models" in models_data and isinstance(models_data["available_models"], list):
                    self.available_models = [ModelDetail(**m) for m in models_data["available_models"]]
                self.default_model_id = models_data.get("default_model_id", self.default_model_id)
                logger.info(f"Loaded {len(self.available_models)} model(s) from MODELS_CONFIG_JSON.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding MODELS_CONFIG_JSON: {e}. Using default models.")
            except TypeError as e:
                logger.error(f"Error parsing model data from MODELS_CONFIG_JSON: {e}. Using default models.")


        if not self.available_models:
            # Provide a default list if no configuration is found
            self.available_models = [
                ModelDetail(model_id="gemini-1.5-pro", provider="google", api_key_env_var="GEMINI_API_KEY"),
                ModelDetail(model_id="gpt-4o", provider="openai", api_key_env_var="OPENAI_API_KEY"),
                # Example for a local model (actual implementation would require more setup)
                # ModelDetail(model_id="llama3-8b-local", provider="local", parameters={"path": "/models/llama3-8b"})
            ]
            logger.info("MODELS_CONFIG_JSON not found or empty. Loaded a default list of models.")

        # Validate and set default_model_id
        if self.default_model_id:
            if not any(m.model_id == self.default_model_id for m in self.available_models):
                logger.warning(
                    f"Configured default_model_id '{self.default_model_id}' not found in available_models. "
                    f"Available: {[m.model_id for m in self.available_models]}"
                )
                self.default_model_id = None # Invalidate if not found

        if not self.default_model_id and self.available_models:
            self.default_model_id = self.available_models[0].model_id
            logger.info(f"Default model ID not set or invalid, using first available: {self.default_model_id}")
        elif not self.available_models:
            logger.warning("No AI models configured and no defaults available.")
            self.default_model_id = None


class UnifiedConfig:
    """
    Unified configuration management system that consolidates all configuration sources
    and provides a single interface for accessing application settings.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._config_cache = {}
        
        # Load environment variables
        self._load_environment()
        
        # Initialize configuration components
        self.app = AppConfig(
            app_name=os.getenv("APP_NAME", "LangGraph 101"),
            version=os.getenv("APP_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"),
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "1"))
        )
        
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///data/langgraph_101.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            timeout=int(os.getenv("DB_TIMEOUT", "30")),
            backup_enabled=os.getenv("DB_BACKUP_ENABLED", "true").lower() in ("true", "1", "yes"),
            backup_path=os.getenv("DB_BACKUP_PATH", "data/backups")
        )
        
        self.api = APIConfig(
            # Core AI APIs
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY"),
            
            # Specialized APIs
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            stability_api_key=os.getenv("STABILITY_API_KEY") or os.getenv("STABILITYAI_API_KEY"),
            pexels_api_key=os.getenv("PEXELS_API_KEY"),
            pixabay_api_key=os.getenv("PIXABAY_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            youtube_data_api_key=os.getenv("YOUTUBE_DATA_API_KEY") or os.getenv("YOUTUBE_API_KEY"),
            news_api_key=os.getenv("NEWS_API_KEY"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
            
            # Settings
            max_requests_per_minute=int(os.getenv("API_MAX_REQUESTS_PER_MINUTE", "60")),
            request_timeout=int(os.getenv("API_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS", "3"))
        )
        
        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY"),
            jwt_secret=os.getenv("JWT_SECRET"),
            jwt_expiration=int(os.getenv("JWT_EXPIRATION", "3600")),
            auth_enabled=os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes"),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "1800")),
            max_requests_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
            max_input_length=int(os.getenv("MAX_INPUT_LENGTH", "10000"))
        )
        
        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_path=os.getenv("LOG_FILE", "logs/langgraph_101.log"),
            console_output=os.getenv("LOG_CONSOLE", "true").lower() in ("true", "1", "yes")
        )

        self.models = ModelsConfig()
        
        self.langgraph = LangGraphConfig(
            max_iterations=int(os.getenv("LANGGRAPH_MAX_ITERATIONS", "10")),
            default_temperature=float(os.getenv("LANGGRAPH_TEMPERATURE", "0.7")),
            default_model_id=self.models.default_model_id, # Set from ModelsConfig
            enable_memory=os.getenv("LANGGRAPH_MEMORY", "true").lower() in ("true", "1", "yes"),
            memory_type=os.getenv("LANGGRAPH_MEMORY_TYPE", "conversation"),
            max_memory_size=int(os.getenv("LANGGRAPH_MEMORY_SIZE", "1000"))
        )
          # Validate configuration
        self._validate_configuration()
        
        # Add backward compatibility property
        self.api_keys = self.api
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv
            env_files = [".env", ".env.local", ".env.production"]
            
            for env_file in env_files:
                if os.path.exists(env_file):
                    load_dotenv(env_file, override=False)
                    logger.info(f"Loaded environment from {env_file}")
                    break
            else:
                logger.warning("No .env file found. Using environment variables only.")
        except ImportError:
            logger.warning("python-dotenv not installed. Using environment variables only.")
    
    def _validate_configuration(self):
        """Validate configuration and log warnings/errors"""
        errors = []
        warnings = []
        
        # Validate API keys
        if not any([
            self.api.openai_api_key, 
            self.api.gemini_api_key, 
            self.api.google_api_key
        ]):
            warnings.append("No primary AI API key configured")
        
        # Validate security for production
        if self.app.is_production():
            if not self.security.secret_key or len(self.security.secret_key) < 32:
                errors.append("Strong secret key required for production")
            
            if not self.security.jwt_secret or len(self.security.jwt_secret) < 32:
                errors.append("Strong JWT secret required for production")
        
        # Validate database configuration
        if not self.database.url:
            errors.append("Database URL is required")
        
        # Log validation results
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        for error in errors:
            logger.error(f"Configuration error: {error}")
        
        if errors and self.app.is_production():
            raise ConfigError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        return self.api.get_api_key(provider)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate required API keys
        if not any([
            self.api.openai_api_key,
            self.api.gemini_api_key,
            self.api.google_api_key
        ]):
            errors.append("At least one primary AI API key is required")
        
        # Validate database path
        if not self.database.url:
            errors.append("Database URL is required")
        
        # Validate security settings for production
        if self.app.is_production():
            if not self.security.secret_key or len(self.security.secret_key) < 32:
                errors.append("Strong secret key is required for production")
            
            if not self.security.jwt_secret:
                errors.append("JWT secret is required for production")
        
        return errors
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {
            "app": {
                "name": self.app.app_name,
                "version": self.app.version,
                "environment": self.app.environment,
                "debug": self.app.debug
            },
            "database": {
                "url": self.database.url if include_sensitive else "***",
                "pool_size": self.database.pool_size,
                "timeout": self.database.timeout
            },
            "api": {
                "max_requests_per_minute": self.api.max_requests_per_minute,
                "request_timeout": self.api.request_timeout,
                "retry_attempts": self.api.retry_attempts
            },
            "security": {
                "auth_enabled": self.security.auth_enabled,
                "session_timeout": self.security.session_timeout,
                "rate_limit_enabled": self.security.rate_limit_enabled
            },
            "logging": {
                "level": self.logging.level,
                "console_output": self.logging.console_output
            }
        }
        
        if include_sensitive:
            # Add API keys (masked)
            api_keys = {}
            for attr in dir(self.api):
                if attr.endswith('_api_key'):
                    key = getattr(self.api, attr)
                    if key:
                        api_keys[attr] = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            config_dict["api"]["keys"] = api_keys
        
        return config_dict


# Global configuration instance
@lru_cache(maxsize=1)
def get_config() -> UnifiedConfig:
    """Get the global configuration instance"""
    return UnifiedConfig()


# Convenience functions for backward compatibility
def load_config() -> Dict[str, Any]:
    """Load configuration as dictionary for backward compatibility"""
    config = get_config()
    return {
        # Core API keys
        "api_key": config.api.gemini_api_key or config.api.google_api_key,
        "openai_api_key": config.api.openai_api_key,
        "tavily_api_key": config.api.tavily_api_key,
        "elevenlabs_api_key": config.api.elevenlabs_api_key,
        "stability_api_key": config.api.stability_api_key,
        "pexels_api_key": config.api.pexels_api_key,
        "pixabay_api_key": config.api.pixabay_api_key,
        "deepgram_api_key": config.api.deepgram_api_key,
        "assemblyai_api_key": config.api.assemblyai_api_key,
        "youtube_data_api_key": config.api.youtube_data_api_key,
        "news_api_key": config.api.news_api_key,
        "openweather_api_key": config.api.openweather_api_key,
        
        # App settings
        "debug": config.app.debug,
        "environment": config.app.environment,
        "database_url": config.database.url,
        
        # LangGraph settings
        "model_name": config.langgraph.default_model_id,
        "temperature": config.langgraph.default_temperature,
        "max_iterations": config.langgraph.max_iterations,
        
        # Security settings
        "secret_key": config.security.secret_key,
        "jwt_secret": config.security.jwt_secret
    }


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider"""
    config = get_config()
    return config.get_api_key(provider)


def is_production() -> bool:
    """Check if running in production environment"""
    config = get_config()
    return config.app.is_production()


def is_development() -> bool:
    """Check if running in development environment"""
    config = get_config()
    return config.app.is_development()


# Validation function
def validate_config() -> List[str]:
    """Validate current configuration"""
    config = get_config()
    return config.validate()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"App: {config.app.app_name} v{config.app.version}")
        print(f"Environment: {config.app.environment}")
        print(f"Database: {config.database.url}")
        
        # Validate configuration
        errors = validate_config()
        if errors:
            print(f"❌ Configuration errors: {errors}")
        else:
            print("✅ Configuration validation passed")
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
