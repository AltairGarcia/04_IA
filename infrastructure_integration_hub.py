#!/usr/bin/env python3
"""
LangGraph 101 - Infrastructure Integration Hub
============================================

Central integration hub that connects all new infrastructure components
with existing LangGraph 101 applications. This provides:

- Seamless integration with minimal code changes to existing apps
- Progressive enhancement without breaking existing functionality
- Centralized management of all infrastructure components
- Configuration-driven component activation
- Health monitoring and diagnostics

Components Integrated:
- API Gateway with load balancing and routing
- Message Queue System for async processing
- Database Connection Pooling for optimized DB access
- Enhanced Rate Limiting with multiple algorithms
- Configuration Hot Reloading for runtime updates
- Redis Cache Manager for performance optimization

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import uuid
from functools import wraps

# FastAPI and async components
from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

# Import our new infrastructure components
from api_gateway import APIGateway, APIGatewayConfig, ServiceEndpoint, RouteConfig
from message_queue_system import MessageQueueSystem, MessageQueueConfig, TaskPriority, create_example_tasks
from database_connection_pool import DatabaseConnectionPool, DatabaseConfig
from cache_manager import CacheManager, CacheConfig
from enhanced_rate_limiting import EnhancedRateLimiter, RateLimitConfig
from config_hot_reload import ConfigHotReloader, HotReloadConfig

# Import existing application components (with fallbacks for missing components)
try:
    from config import load_config
    from agent import create_agent, invoke_agent
    from tools import get_tools
    from personas import get_persona_by_name, get_all_personas
    from content_creation import ContentCreator
    from database import Database
    from history import get_history_manager
    from memory_manager import get_memory_manager
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for infrastructure integration"""
    
    # Component activation flags
    enable_api_gateway: bool = True
    enable_message_queue: bool = True
    enable_connection_pool: bool = True
    enable_cache_manager: bool = True
    enable_rate_limiting: bool = True
    enable_hot_reload: bool = True
    
    # Service configuration
    service_name: str = "langgraph-101"
    service_version: str = "1.0.0"
    
    # API Gateway settings
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    
    # Backend service settings
    backend_host: str = "127.0.0.1"
    backend_port: int = 8001
    
    # Security settings
    jwt_secret: str = "langgraph-101-jwt-secret"
    api_key: str = "langgraph-101-api-key"
    
    # Performance settings
    worker_processes: int = 1
    enable_monitoring: bool = True
    
    def __post_init__(self):
        # Load from environment variables
        self.enable_api_gateway = os.getenv('ENABLE_API_GATEWAY', str(self.enable_api_gateway)).lower() == 'true'
        self.enable_message_queue = os.getenv('ENABLE_MESSAGE_QUEUE', str(self.enable_message_queue)).lower() == 'true'
        self.enable_connection_pool = os.getenv('ENABLE_CONNECTION_POOL', str(self.enable_connection_pool)).lower() == 'true'
        self.enable_cache_manager = os.getenv('ENABLE_CACHE_MANAGER', str(self.enable_cache_manager)).lower() == 'true'
        self.enable_rate_limiting = os.getenv('ENABLE_RATE_LIMITING', str(self.enable_rate_limiting)).lower() == 'true'
        self.enable_hot_reload = os.getenv('ENABLE_HOT_RELOAD', str(self.enable_hot_reload)).lower() == 'true'
        
        self.gateway_host = os.getenv('GATEWAY_HOST', self.gateway_host)
        self.gateway_port = int(os.getenv('GATEWAY_PORT', str(self.gateway_port)))
        self.backend_host = os.getenv('BACKEND_HOST', self.backend_host)
        self.backend_port = int(os.getenv('BACKEND_PORT', str(self.backend_port)))
        
        self.jwt_secret = os.getenv('JWT_SECRET', self.jwt_secret)
        self.api_key = os.getenv('API_KEY', self.api_key)

class InfrastructureHub:
    """Central hub for managing all infrastructure components"""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        
        # Infrastructure components
        self.api_gateway = None
        self.message_queue = None
        self.connection_pool = None
        self.cache_manager = None
        self.rate_limiter = None
        self.hot_reloader = None
        
        # Application components
        self.backend_service = None
        self.health_status = {}
        
        # State management
        self.is_running = False
        self.startup_tasks = []
        self.shutdown_tasks = []
        self._lock = threading.RLock()
        
        logger.info(f"Infrastructure Hub initialized with config: {self.config}")
    
    async def initialize(self):
        """Initialize all enabled infrastructure components"""
        logger.info("Starting infrastructure component initialization...")
        
        try:
            # Initialize in dependency order
            await self._initialize_cache_manager()
            await self._initialize_connection_pool()
            await self._initialize_message_queue()
            await self._initialize_rate_limiter()
            await self._initialize_hot_reloader()
            await self._initialize_backend_service()
            await self._initialize_api_gateway()
            
            # Start monitoring
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            self.is_running = True
            logger.info("Infrastructure initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Infrastructure initialization failed: {e}")
            await self.shutdown()
            raise
    
    async def _initialize_cache_manager(self):
        """Initialize Redis cache manager"""
        if not self.config.enable_cache_manager:
            logger.info("Cache manager disabled by configuration")
            return
        
        try:
            cache_config = CacheConfig()
            self.cache_manager = CacheManager(cache_config)
            await self.cache_manager.initialize()
            
            # Test cache connectivity
            await self.cache_manager.set("test_key", "test_value", ttl=10)
            test_value = await self.cache_manager.get("test_key")
            
            if test_value == "test_value":
                logger.info("Cache manager initialized and tested successfully")
                self.health_status['cache_manager'] = 'healthy'
            else:
                raise Exception("Cache test failed")
                
        except Exception as e:
            logger.warning(f"Cache manager initialization failed: {e}")
            self.cache_manager = None
            self.health_status['cache_manager'] = 'unavailable'
    
    async def _initialize_connection_pool(self):
        """Initialize database connection pool"""
        if not self.config.enable_connection_pool:
            logger.info("Connection pool disabled by configuration")
            return
        
        try:
            db_config = DatabaseConfig()
            self.connection_pool = DatabaseConnectionPool(db_config)
            
            # Test database connectivity
            async with self.connection_pool.get_connection() as conn:
                if hasattr(conn, 'execute'):
                    await conn.execute("SELECT 1")
                else:
                    # For sync connections
                    conn.execute("SELECT 1")
            
            logger.info("Database connection pool initialized successfully")
            self.health_status['connection_pool'] = 'healthy'
            
        except Exception as e:
            logger.warning(f"Connection pool initialization failed: {e}")
            self.connection_pool = None
            self.health_status['connection_pool'] = 'unavailable'
    
    async def _initialize_message_queue(self):
        """Initialize message queue system"""
        if not self.config.enable_message_queue:
            logger.info("Message queue disabled by configuration")
            return
        
        try:
            mq_config = MessageQueueConfig()
            self.message_queue = MessageQueueSystem(mq_config)
            
            # Create example tasks for LangGraph operations
            await self._register_langgraph_tasks()
            
            # Start monitoring
            self.message_queue.start_monitoring()
            
            logger.info("Message queue system initialized successfully")
            self.health_status['message_queue'] = 'healthy'
            
        except Exception as e:
            logger.warning(f"Message queue initialization failed: {e}")
            self.message_queue = None
            self.health_status['message_queue'] = 'unavailable'
    
    async def _register_langgraph_tasks(self):
        """Register LangGraph-specific async tasks"""
        if not self.message_queue:
            return
        
        @self.message_queue.register_task('process_agent_message', priority=TaskPriority.HIGH)
        def process_agent_message(self, message: str, persona_name: str, session_id: str):
            """Process agent message asynchronously"""
            try:
                logger.info(f"Processing agent message for session {session_id}")
                
                # Simulate agent processing (replace with actual agent logic)
                time.sleep(2)  # Simulate processing time
                
                result = {
                    'session_id': session_id,
                    'response': f"Processed: {message[:50]}...",
                    'persona': persona_name,
                    'processed_at': datetime.utcnow().isoformat(),
                    'processing_time': 2.0
                }
                
                logger.info(f"Agent message processed for session {session_id}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to process agent message: {e}")
                raise
        
        @self.message_queue.register_task('create_content', priority=TaskPriority.NORMAL)
        def create_content(self, topic: str, content_type: str, options: Dict[str, Any] = None):
            """Create content asynchronously"""
            try:
                logger.info(f"Creating {content_type} content for topic: {topic}")
                
                # Simulate content creation
                time.sleep(3)
                
                result = {
                    'topic': topic,
                    'content_type': content_type,
                    'content': f"Generated {content_type} content about {topic}",
                    'created_at': datetime.utcnow().isoformat(),
                    'word_count': 500,
                    'options': options or {}
                }
                
                logger.info(f"Content created for topic: {topic}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to create content: {e}")
                raise
        
        @self.message_queue.register_task('export_conversation', priority=TaskPriority.LOW)
        def export_conversation(self, conversation_id: str, format_type: str):
            """Export conversation asynchronously"""
            try:
                logger.info(f"Exporting conversation {conversation_id} to {format_type}")
                
                # Simulate export process
                time.sleep(1)
                
                result = {
                    'conversation_id': conversation_id,
                    'format': format_type,
                    'file_path': f'/tmp/export_{conversation_id}.{format_type}',
                    'exported_at': datetime.utcnow().isoformat(),
                    'size_bytes': 1024
                }
                
                logger.info(f"Conversation {conversation_id} exported successfully")
                return result
                
            except Exception as e:
                logger.error(f"Failed to export conversation: {e}")
                raise
        
        logger.info("LangGraph tasks registered with message queue")
    
    async def _initialize_rate_limiter(self):
        """Initialize enhanced rate limiter"""
        if not self.config.enable_rate_limiting:
            logger.info("Rate limiter disabled by configuration")
            return
        
        try:
            rate_config = RateLimitConfig()
            self.rate_limiter = EnhancedRateLimiter(rate_config)
            
            # Configure rate limits for different endpoints
            await self._configure_rate_limits()
            
            logger.info("Enhanced rate limiter initialized successfully")
            self.health_status['rate_limiter'] = 'healthy'
            
        except Exception as e:
            logger.warning(f"Rate limiter initialization failed: {e}")
            self.rate_limiter = None
            self.health_status['rate_limiter'] = 'unavailable'
    
    async def _configure_rate_limits(self):
        """Configure rate limits for different operations"""
        if not self.rate_limiter:
            return
        
        # Configure different rate limits
        rate_configs = {
            'chat_endpoint': {'requests_per_minute': 60, 'requests_per_hour': 1000},
            'content_creation': {'requests_per_minute': 10, 'requests_per_hour': 100},
            'export_operations': {'requests_per_minute': 5, 'requests_per_hour': 50},
            'health_checks': {'requests_per_minute': 300, 'requests_per_hour': 5000}
        }
        
        for endpoint, limits in rate_configs.items():
            self.rate_limiter.configure_endpoint_limits(endpoint, limits)
        
        logger.info("Rate limits configured for all endpoints")
    
    async def _initialize_hot_reloader(self):
        """Initialize configuration hot reloader"""
        if not self.config.enable_hot_reload:
            logger.info("Hot reloader disabled by configuration")
            return
        
        try:
            reload_config = HotReloadConfig()
            self.hot_reloader = ConfigHotReloader(reload_config)
            
            # Register configuration change callbacks
            await self._register_config_callbacks()
            
            # Start watching for changes
            self.hot_reloader.start_watching()
            
            logger.info("Configuration hot reloader initialized successfully")
            self.health_status['hot_reloader'] = 'healthy'
            
        except Exception as e:
            logger.warning(f"Hot reloader initialization failed: {e}")
            self.hot_reloader = None
            self.health_status['hot_reloader'] = 'unavailable'
    
    async def _register_config_callbacks(self):
        """Register callbacks for configuration changes"""
        if not self.hot_reloader:
            return
        
        def on_rate_limit_change(old_value, new_value):
            """Handle rate limit configuration changes"""
            logger.info(f"Rate limit configuration changed: {old_value} -> {new_value}")
            if self.rate_limiter:
                asyncio.create_task(self._configure_rate_limits())
        
        def on_cache_config_change(old_value, new_value):
            """Handle cache configuration changes"""
            logger.info(f"Cache configuration changed: {old_value} -> {new_value}")
            # Could reload cache configuration here
        
        # Register callbacks
        self.hot_reloader.register_callback('rate_limiting.*', on_rate_limit_change)
        self.hot_reloader.register_callback('cache.*', on_cache_config_change)
        
        logger.info("Configuration change callbacks registered")
    
    async def _initialize_backend_service(self):
        """Initialize the backend service that wraps existing LangGraph functionality"""
        try:
            self.backend_service = LangGraphBackendService(
                self.config,
                infrastructure_hub=self
            )
            
            logger.info("Backend service initialized successfully")
            self.health_status['backend_service'] = 'healthy'
            
        except Exception as e:
            logger.error(f"Backend service initialization failed: {e}")
            self.backend_service = None
            self.health_status['backend_service'] = 'unavailable'
            raise
    
    async def _initialize_api_gateway(self):
        """Initialize API gateway"""
        if not self.config.enable_api_gateway:
            logger.info("API gateway disabled by configuration")
            return
        
        try:
            gateway_config = APIGatewayConfig()
            gateway_config.host = self.config.gateway_host
            gateway_config.port = self.config.gateway_port
            gateway_config.jwt_secret = self.config.jwt_secret
            
            self.api_gateway = APIGateway(gateway_config)
            
            # Register backend service
            if self.backend_service:
                backend_url = f"http://{self.config.backend_host}:{self.config.backend_port}"
                self.api_gateway.register_service(
                    'langgraph-backend',
                    [backend_url],
                    health_check_path='/health'
                )
                
                # Register routes
                await self._register_gateway_routes()
            
            logger.info("API gateway initialized successfully")
            self.health_status['api_gateway'] = 'healthy'
            
        except Exception as e:
            logger.warning(f"API gateway initialization failed: {e}")
            self.api_gateway = None
            self.health_status['api_gateway'] = 'unavailable'
    
    async def _register_gateway_routes(self):
        """Register routes with the API gateway"""
        if not self.api_gateway:
            return
        
        # Define routes
        routes = [
            {
                'path': '/api/chat',
                'service_name': 'langgraph-backend',
                'methods': ['POST'],
                'auth_required': True,
                'rate_limit': 60,
                'cache_ttl': None
            },
            {
                'path': '/api/content/*',
                'service_name': 'langgraph-backend',
                'methods': ['GET', 'POST'],
                'auth_required': True,
                'rate_limit': 30,
                'cache_ttl': 300
            },
            {
                'path': '/health',
                'service_name': 'langgraph-backend',
                'methods': ['GET'],
                'auth_required': False,
                'rate_limit': 300,
                'cache_ttl': 60
            },
            {
                'path': '/api/personas',
                'service_name': 'langgraph-backend',
                'methods': ['GET'],
                'auth_required': False,
                'rate_limit': 100,
                'cache_ttl': 600
            }
        ]
        
        for route in routes:
            self.api_gateway.register_route(**route)
        
        logger.info(f"Registered {len(routes)} routes with API gateway")
    
    async def _start_monitoring(self):
        """Start monitoring and health checking"""
        async def monitoring_loop():
            while self.is_running:
                try:
                    # Update health status
                    await self._update_health_status()
                    
                    # Log metrics periodically
                    if self.message_queue:
                        metrics = self.message_queue.get_metrics()
                        logger.debug(f"Message queue metrics: {asdict(metrics)}")
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(5)
        
        # Start monitoring task
        asyncio.create_task(monitoring_loop())
        logger.info("Monitoring started")
    
    async def _update_health_status(self):
        """Update health status of all components"""
        # Check cache manager
        if self.cache_manager:
            try:
                await self.cache_manager.set("health_check", "ok", ttl=5)
                self.health_status['cache_manager'] = 'healthy'
            except:
                self.health_status['cache_manager'] = 'unhealthy'
        
        # Check message queue
        if self.message_queue:
            try:
                health = self.message_queue.health_check()
                self.health_status['message_queue'] = health.get('overall_status', 'unknown')
            except:
                self.health_status['message_queue'] = 'unhealthy'
        
        # Check API gateway
        if self.api_gateway:
            try:
                health = await self.api_gateway.get_health_status()
                self.health_status['api_gateway'] = health.get('status', 'unknown')
            except:
                self.health_status['api_gateway'] = 'unhealthy'
    
    def get_component(self, name: str):
        """Get infrastructure component by name"""
        components = {
            'api_gateway': self.api_gateway,
            'message_queue': self.message_queue,
            'connection_pool': self.connection_pool,
            'cache_manager': self.cache_manager,
            'rate_limiter': self.rate_limiter,
            'hot_reloader': self.hot_reloader,
            'backend_service': self.backend_service
        }
        return components.get(name)
    
    async def submit_async_task(self, task_name: str, *args, **kwargs) -> Optional[str]:
        """Submit an asynchronous task"""
        if not self.message_queue:
            logger.warning("Message queue not available, cannot submit async task")
            return None
        
        try:
            task_id = self.message_queue.submit_task(task_name, *args, **kwargs)
            logger.info(f"Submitted async task '{task_name}' with ID: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to submit async task '{task_name}': {e}")
            return None
    
    async def get_task_result(self, task_id: str):
        """Get result of an async task"""
        if not self.message_queue:
            return None
        
        return self.message_queue.get_task_result(task_id)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        overall_healthy = all(
            status in ['healthy', 'unavailable'] 
            for status in self.health_status.values()
        )
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'components': self.health_status,
            'is_running': self.is_running
        }
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Starting infrastructure shutdown...")
        self.is_running = False
        
        # Shutdown in reverse dependency order
        if self.api_gateway:
            await self.api_gateway.shutdown()
        
        if self.hot_reloader:
            self.hot_reloader.stop_watching()
        
        if self.message_queue:
            self.message_queue.stop_monitoring()
            self.message_queue.shutdown()
        
        if self.connection_pool:
            await self.connection_pool.close_all()
        
        if self.cache_manager:
            await self.cache_manager.close()
        
        logger.info("Infrastructure shutdown completed")


class LangGraphBackendService:
    """Backend service that wraps existing LangGraph functionality with infrastructure integration"""
    
    def __init__(self, config: IntegrationConfig, infrastructure_hub: InfrastructureHub):
        self.config = config
        self.hub = infrastructure_hub
        self.app = FastAPI(
            title="LangGraph 101 Backend Service",
            version=config.service_version,
            description="Backend service with integrated infrastructure"
        )
        
        # Application components (with error handling for missing imports)
        self.agent = None
        self.content_creator = None
        self.tools = None
        self.initialized = False
        
        # Setup FastAPI app
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("LangGraph Backend Service created")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add custom middleware for infrastructure integration
        self.app.add_middleware(InfrastructureMiddleware, hub=self.hub)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "langgraph-backend",
                "version": self.config.service_version,
                "infrastructure": self.hub.get_health_status()
            }
        
        @self.app.post("/api/chat")
        async def chat_endpoint(request: Request, background_tasks: BackgroundTasks):
            """Enhanced chat endpoint with infrastructure integration"""
            try:
                data = await request.json()
                message = data.get('message', '')
                persona_name = data.get('persona', 'default')
                session_id = data.get('session_id', str(uuid.uuid4()))
                async_processing = data.get('async', False)
                
                # Check rate limiting
                await self._check_rate_limit(request, 'chat_endpoint')
                
                # Check cache for similar requests
                cache_key = f"chat:{hash(message + persona_name)}"
                cached_response = await self._get_cached_response(cache_key)
                if cached_response:
                    return cached_response
                
                if async_processing and self.hub.message_queue:
                    # Submit async task
                    task_id = await self.hub.submit_async_task(
                        'process_agent_message',
                        message, persona_name, session_id
                    )
                    
                    return {
                        "task_id": task_id,
                        "status": "processing",
                        "session_id": session_id,
                        "message": "Request submitted for async processing"
                    }
                else:
                    # Process synchronously
                    response = await self._process_chat_message(message, persona_name, session_id)
                    
                    # Cache the response
                    await self._cache_response(cache_key, response, ttl=300)
                    
                    return response
                
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/chat/result/{task_id}")
        async def get_chat_result(task_id: str):
            """Get result of async chat processing"""
            result = await self.hub.get_task_result(task_id)
            if result:
                return {
                    "task_id": task_id,
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "execution_time": result.execution_time
                }
            else:
                raise HTTPException(status_code=404, detail="Task not found")
        
        @self.app.post("/api/content/create")
        async def content_creation_endpoint(request: Request):
            """Enhanced content creation with async processing"""
            try:
                data = await request.json()
                topic = data.get('topic', '')
                content_type = data.get('type', 'article')
                async_processing = data.get('async', True)  # Default to async for content creation
                
                await self._check_rate_limit(request, 'content_creation')
                
                if async_processing and self.hub.message_queue:
                    task_id = await self.hub.submit_async_task(
                        'create_content',
                        topic, content_type, data.get('options', {})
                    )
                    
                    return {
                        "task_id": task_id,
                        "status": "processing",
                        "topic": topic,
                        "content_type": content_type
                    }
                else:
                    # Process synchronously (fallback)
                    result = await self._create_content_sync(topic, content_type, data.get('options', {}))
                    return result
                
            except Exception as e:
                logger.error(f"Content creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/personas")
        async def get_personas():
            """Get available personas"""
            try:
                cache_key = "personas:all"
                cached = await self._get_cached_response(cache_key)
                if cached:
                    return cached
                
                # Get personas (with fallback)
                try:
                    personas = get_all_personas()
                    response = {"personas": personas}
                except:
                    response = {"personas": {"default": {"name": "Default", "description": "Default persona"}}}
                
                await self._cache_response(cache_key, response, ttl=600)
                return response
                
            except Exception as e:
                logger.error(f"Get personas error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/status")
        async def get_status():
            """Get comprehensive system status"""
            return {
                "service": {
                    "name": "langgraph-backend",
                    "version": self.config.service_version,
                    "initialized": self.initialized,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "infrastructure": self.hub.get_health_status(),
                "config": {
                    "api_gateway_enabled": self.config.enable_api_gateway,
                    "message_queue_enabled": self.config.enable_message_queue,
                    "cache_enabled": self.config.enable_cache_manager,
                    "rate_limiting_enabled": self.config.enable_rate_limiting
                }
            }
    
    async def _check_rate_limit(self, request: Request, endpoint: str):
        """Check rate limiting for endpoint"""
        if not self.hub.rate_limiter:
            return
        
        client_ip = self._get_client_ip(request)
        try:
            is_allowed = await self.hub.rate_limiter.is_allowed(
                key=f"{endpoint}:{client_ip}",
                endpoint=endpoint
            )
            if not is_allowed:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except Exception as e:
            logger.warning(f"Rate limiting error: {e}")
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if not self.hub.cache_manager:
            return None
        
        try:
            cached = await self.hub.cache_manager.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for key: {cache_key}")
                return json.loads(cached) if isinstance(cached, str) else cached
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any], ttl: int):
        """Cache response"""
        if not self.hub.cache_manager:
            return
        
        try:
            await self.hub.cache_manager.set(
                cache_key,
                json.dumps(response) if not isinstance(response, str) else response,
                ttl=ttl
            )
            logger.debug(f"Cached response for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _process_chat_message(self, message: str, persona_name: str, session_id: str) -> Dict[str, Any]:
        """Process chat message (placeholder implementation)"""
        # This would integrate with actual LangGraph agent
        response = f"Echo: {message} (from {persona_name})"
        
        return {
            "response": response,
            "session_id": session_id,
            "persona": persona_name,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": 0.1
        }
    
    async def _create_content_sync(self, topic: str, content_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create content synchronously (placeholder implementation)"""
        content = f"Generated {content_type} about {topic}"
        
        return {
            "topic": topic,
            "content_type": content_type,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "word_count": len(content.split()),
            "options": options
        }
    
    def run(self):
        """Run the backend service"""
        uvicorn.run(
            self.app,
            host=self.config.backend_host,
            port=self.config.backend_port,
            workers=1  # Single worker for now
        )


class InfrastructureMiddleware(BaseHTTPMiddleware):
    """Middleware for infrastructure integration"""
    
    def __init__(self, app, hub: InfrastructureHub):
        super().__init__(app)
        self.hub = hub
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Log request metrics
            execution_time = time.time() - start_time
            logger.debug(f"Request {request_id} completed in {execution_time:.3f}s")
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{execution_time:.3f}"
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Request {request_id} failed after {execution_time:.3f}s: {e}")
            raise


# Integration utilities
def create_integrated_app(config: Optional[IntegrationConfig] = None) -> InfrastructureHub:
    """Create fully integrated LangGraph application with all infrastructure components"""
    return InfrastructureHub(config)

async def run_integrated_system(config: Optional[IntegrationConfig] = None):
    """Run the complete integrated system"""
    hub = create_integrated_app(config)
    
    try:
        await hub.initialize()
        
        # Start backend service
        if hub.backend_service:
            backend_task = asyncio.create_task(
                _run_backend_service(hub.backend_service, hub.config)
            )
        
        # Start API gateway
        if hub.api_gateway:
            gateway_task = asyncio.create_task(
                _run_api_gateway(hub.api_gateway)
            )
        
        # Wait for shutdown signal
        await _wait_for_shutdown()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await hub.shutdown()

async def _run_backend_service(service: LangGraphBackendService, config: IntegrationConfig):
    """Run backend service"""
    server_config = uvicorn.Config(
        service.app,
        host=config.backend_host,
        port=config.backend_port,
        workers=1,
        access_log=False
    )
    server = uvicorn.Server(server_config)
    await server.serve()

async def _run_api_gateway(gateway: APIGateway):
    """Run API gateway"""
    await gateway.startup()
    # Gateway runs in its own process/thread
    
async def _wait_for_shutdown():
    """Wait for shutdown signal"""
    import signal
    
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await shutdown_event.wait()


def main():
    """Main entry point for integrated system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangGraph 101 Integrated System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--gateway-port', type=int, default=8000, help='API Gateway port')
    parser.add_argument('--backend-port', type=int, default=8001, help='Backend service port')
    parser.add_argument('--disable-gateway', action='store_true', help='Disable API Gateway')
    parser.add_argument('--disable-queue', action='store_true', help='Disable Message Queue')
    parser.add_argument('--disable-cache', action='store_true', help='Disable Cache Manager')
    
    args = parser.parse_args()
    
    # Create configuration
    config = IntegrationConfig()
    config.gateway_port = args.gateway_port
    config.backend_port = args.backend_port
    config.enable_api_gateway = not args.disable_gateway
    config.enable_message_queue = not args.disable_queue
    config.enable_cache_manager = not args.disable_cache
    
    logger.info(f"Starting LangGraph 101 Integrated System")
    logger.info(f"API Gateway: {'enabled' if config.enable_api_gateway else 'disabled'} (port {config.gateway_port})")
    logger.info(f"Backend Service: port {config.backend_port}")
    logger.info(f"Message Queue: {'enabled' if config.enable_message_queue else 'disabled'}")
    logger.info(f"Cache Manager: {'enabled' if config.enable_cache_manager else 'disabled'}")
    
    try:
        asyncio.run(run_integrated_system(config))
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


if __name__ == '__main__':
    main()
