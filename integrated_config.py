#!/usr/bin/env python3
"""
LangGraph 101 - Integrated System Configuration
==============================================

Comprehensive configuration file for the LangGraph 101 integrated system.
This provides centralized configuration for all components including:

- Infrastructure components (API Gateway, Message Queue, Database Pool, etc.)
- Application services (Streamlit, CLI, etc.)
- Security settings (Authentication, Rate Limiting, etc.)
- Performance optimization settings
- Monitoring and health check configuration

Author: GitHub Copilot
Date: 2024
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    
    # Primary database settings
    primary_db_type: str = "sqlite"
    primary_db_path: str = "langgraph.db"
    primary_db_url: str = ""
    
    # Connection pool settings
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    idle_timeout: int = 300
    
    # Performance settings
    enable_query_logging: bool = False
    enable_connection_pooling: bool = True
    pool_pre_ping: bool = True
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    backup_retention: int = 7    # days

@dataclass
class RedisConfig:
    """Redis configuration settings"""
    
    host: str = "localhost"
    port: int = 6380  # Updated to use our Docker Redis instance
    db: int = 0
    password: str = ""
    
    # Connection settings
    connection_pool_size: int = 20
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    
    # Performance settings
    max_memory: str = "256mb"
    maxmemory_policy: str = "allkeys-lru"
    
    # Persistence settings
    save_enabled: bool = True
    save_interval: int = 900  # seconds

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    
    # Redis cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    
    # Cache behavior
    default_ttl: int = 3600      # seconds
    max_cache_size: int = 1000   # number of items
    enable_compression: bool = True
    
    # Cache strategies
    cache_strategy: str = "lru"  # lru, lfu, random
    
    # Performance settings
    cache_stats_enabled: bool = True
    async_cache_updates: bool = True

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    
    # Authentication
    jwt_secret: str = "langgraph-101-jwt-secret-change-in-production"
    jwt_expiry: int = 3600  # seconds
    api_key: str = "langgraph-101-api-key"
    
    # Rate limiting
    enable_rate_limiting: bool = True
    default_rate_limit: str = "100/minute"
    burst_rate_limit: str = "10/second"
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Security headers
    enable_security_headers: bool = True
    enable_csrf_protection: bool = True
    
    # Input validation
    max_request_size: int = 10485760  # 10MB
    enable_input_sanitization: bool = True

@dataclass
class ServiceConfig:
    """Service configuration settings"""
    
    # Service ports
    adapter_port: int = 9000
    streamlit_port: int = 8501
    cli_service_port: int = 8002
    gateway_port: int = 8000
    
    # Service hosts
    adapter_host: str = "0.0.0.0"
    streamlit_host: str = "0.0.0.0"
    gateway_host: str = "0.0.0.0"
    
    # Service management
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: int = 5  # seconds
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5    # seconds

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "langgraph.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_interval: int = 15  # seconds
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    cpu_threshold: float = 80.0     # percent
    memory_threshold: float = 85.0  # percent
    disk_threshold: float = 90.0    # percent
    
    # Alerting
    enable_alerting: bool = False
    alert_webhook_url: str = ""
    alert_email: str = ""

@dataclass
class MessageQueueConfig:
    """Message queue configuration settings"""
      # Celery settings
    broker_url: str = "redis://localhost:6380/2"
    result_backend: str = "redis://localhost:6380/3"
    
    # Queue settings
    task_default_queue: str = "default"
    task_routes: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "langgraph.tasks.high_priority": {"queue": "high"},
        "langgraph.tasks.normal_priority": {"queue": "normal"},
        "langgraph.tasks.low_priority": {"queue": "low"}
    })
    
    # Worker settings
    worker_concurrency: int = 4
    worker_prefetch_multiplier: int = 4
    worker_max_tasks_per_child: int = 1000
    
    # Task settings
    task_serializer: str = "json"
    result_serializer: str = "json"
    task_compression: str = "gzip"
    
    # Retry settings
    task_default_retry_delay: int = 60    # seconds
    task_max_retries: int = 3
    
    # Monitoring
    worker_send_task_events: bool = True
    task_send_sent_event: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    
    # General performance
    enable_async: bool = True
    worker_processes: int = 1
    worker_threads: int = 4
    
    # Connection pooling
    enable_connection_pooling: bool = True
    max_pool_size: int = 20
    pool_timeout: int = 30
    
    # Caching
    enable_response_caching: bool = True
    cache_ttl: int = 300  # seconds
    
    # Compression
    enable_gzip: bool = True
    gzip_threshold: int = 1000  # bytes
    
    # Optimization features
    enable_lazy_loading: bool = True
    enable_preloading: bool = True
    preload_models: bool = True

@dataclass
class ComponentConfig:
    """Component activation configuration"""
    
    # Infrastructure components
    enable_api_gateway: bool = True
    enable_message_queue: bool = True
    enable_database_pool: bool = True
    enable_cache_manager: bool = True
    enable_rate_limiting: bool = True
    enable_hot_reload: bool = True
    
    # Application components
    enable_streamlit: bool = True
    enable_cli_service: bool = True
    enable_content_creation: bool = True
    enable_voice_features: bool = False
    
    # Advanced features
    enable_analytics: bool = True
    enable_monitoring: bool = True
    enable_error_tracking: bool = True
    enable_performance_profiling: bool = False

@dataclass
class LangGraphIntegratedConfig:
    """Main configuration class that combines all component configurations"""
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    message_queue: MessageQueueConfig = field(default_factory=MessageQueueConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    components: ComponentConfig = field(default_factory=ComponentConfig)
    
    # Environment settings
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    # System information
    config_version: str = "1.0.0"
    last_updated: str = ""
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set environment from env var
        self.environment = os.getenv('LANGGRAPH_ENV', self.environment)
        self.debug = os.getenv('LANGGRAPH_DEBUG', str(self.debug)).lower() == 'true'
        
        # Apply environment-specific settings
        self._apply_environment_settings()
        
        # Load from environment variables
        self._load_from_environment()
        
        # Set last updated
        from datetime import datetime
        self.last_updated = datetime.now().isoformat()
    
    def _apply_environment_settings(self):
        """Apply environment-specific configuration settings"""
        if self.environment == "production":
            # Production optimizations
            self.debug = False
            self.monitoring.log_level = "WARNING"
            self.security.enable_security_headers = True
            self.security.enable_csrf_protection = True
            self.performance.worker_processes = max(1, os.cpu_count() // 2)
            self.cache.enable_compression = True
            
        elif self.environment == "development":
            # Development optimizations
            self.debug = True
            self.monitoring.log_level = "DEBUG"
            self.performance.enable_performance_profiling = True
            self.components.enable_analytics = False
            
        elif self.environment == "staging":
            # Staging optimizations
            self.debug = False
            self.monitoring.log_level = "INFO"
            self.performance.worker_processes = 2
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Database settings
        if os.getenv('DATABASE_URL'):
            self.database.primary_db_url = os.getenv('DATABASE_URL')
        
        # Redis settings
        self.redis.host = os.getenv('REDIS_HOST', self.redis.host)
        self.redis.port = int(os.getenv('REDIS_PORT', str(self.redis.port)))
        self.redis.password = os.getenv('REDIS_PASSWORD', self.redis.password)
        
        # Security settings
        self.security.jwt_secret = os.getenv('JWT_SECRET', self.security.jwt_secret)
        self.security.api_key = os.getenv('API_KEY', self.security.api_key)
        
        # Service ports
        self.services.adapter_port = int(os.getenv('ADAPTER_PORT', str(self.services.adapter_port)))
        self.services.streamlit_port = int(os.getenv('STREAMLIT_PORT', str(self.services.streamlit_port)))
        self.services.gateway_port = int(os.getenv('GATEWAY_PORT', str(self.services.gateway_port)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def save_to_file(self, file_path: str, format: str = "auto"):
        """Save configuration to file"""
        path = Path(file_path)
        
        if format == "auto":
            format = path.suffix.lower().lstrip('.')
        
        if format in ["json"]:
            with open(file_path, 'w') as f:
                f.write(self.to_json())
        elif format in ["yaml", "yml"]:
            with open(file_path, 'w') as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'LangGraphIntegratedConfig':
        """Load configuration from file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if path.suffix.lower() in ['.json']:
                data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LangGraphIntegratedConfig':
        """Create configuration from dictionary"""
        # Extract component configurations
        config = cls()
        
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'redis' in data:
            config.redis = RedisConfig(**data['redis'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'security' in data:
            config.security = SecurityConfig(**data['security'])
        if 'services' in data:
            config.services = ServiceConfig(**data['services'])
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        if 'message_queue' in data:
            config.message_queue = MessageQueueConfig(**data['message_queue'])
        if 'performance' in data:
            config.performance = PerformanceConfig(**data['performance'])
        if 'components' in data:
            config.components = ComponentConfig(**data['components'])
        
        # Update other fields
        config.environment = data.get('environment', config.environment)
        config.debug = data.get('debug', config.debug)
        config.config_version = data.get('config_version', config.config_version)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate ports
        ports = [
            self.services.adapter_port,
            self.services.streamlit_port,
            self.services.cli_service_port,
            self.services.gateway_port,
            self.redis.port
        ]
        
        if len(set(ports)) != len(ports):
            errors.append("Port conflicts detected - services cannot use the same port")
        
        # Validate security settings
        if self.environment == "production":
            if self.security.jwt_secret == "langgraph-101-jwt-secret-change-in-production":
                errors.append("JWT secret must be changed for production")
            
            if self.security.api_key == "langgraph-101-api-key":
                errors.append("API key must be changed for production")
        
        # Validate database settings
        if self.database.primary_db_type not in ["sqlite", "postgresql", "mysql"]:
            errors.append(f"Unsupported database type: {self.database.primary_db_type}")
        
        # Validate performance settings
        if self.performance.worker_processes < 1:
            errors.append("Worker processes must be at least 1")
        
        return errors

def load_config(config_path: Optional[str] = None) -> LangGraphIntegratedConfig:
    """Load configuration from file or create default"""
    if config_path and os.path.exists(config_path):
        try:
            return LangGraphIntegratedConfig.load_from_file(config_path)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
    
    # Check for default config files
    default_files = [
        'langgraph_config.yaml',
        'langgraph_config.yml',
        'langgraph_config.json',
        'config.yaml',
        'config.yml',
        'config.json'
    ]
    
    for default_file in default_files:
        if os.path.exists(default_file):
            try:
                return LangGraphIntegratedConfig.load_from_file(default_file)
            except Exception as e:
                print(f"Warning: Failed to load config from {default_file}: {e}")
    
    # Return default configuration
    return LangGraphIntegratedConfig()

def create_sample_config(output_path: str = "langgraph_config.yaml"):
    """Create a sample configuration file"""
    config = LangGraphIntegratedConfig()
    config.save_to_file(output_path)
    print(f"Sample configuration saved to: {output_path}")

def main():
    """Main function for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Configuration Manager")
    parser.add_argument('--create-sample', action='store_true', help="Create sample configuration file")
    parser.add_argument('--validate', help="Validate configuration file")
    parser.add_argument('--output', default="langgraph_config.yaml", help="Output file path")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config(args.output)
    elif args.validate:
        try:
            config = LangGraphIntegratedConfig.load_from_file(args.validate)
            errors = config.validate()
            
            if errors:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("✅ Configuration validation passed")
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
    else:
        print("Use --create-sample to create a sample configuration file")
        print("Use --validate <file> to validate a configuration file")

if __name__ == "__main__":
    main()
