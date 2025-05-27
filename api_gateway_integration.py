#!/usr/bin/env python3
"""
LangGraph 101 - API Gateway Integration Adapter
==============================================

Integration adapter that connects existing LangGraph 101 applications
with the new API Gateway infrastructure, providing:
- Seamless integration with minimal code changes
- Authentication and authorization handling
- Request routing and load balancing
- Monitoring and metrics collection
- Progressive enhancement capabilities

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
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import uuid

# FastAPI and async components
from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware

# Import our infrastructure components
from api_gateway import APIGateway, APIGatewayConfig, ServiceEndpoint, RouteConfig
from message_queue_system import MessageQueueSystem, TaskPriority, create_example_tasks
from database_connection_pool import DatabaseConnectionPool
from cache_manager import CacheManager
from enhanced_rate_limiting import EnhancedRateLimiter

# Import existing application components
from config import load_config
from agent import create_agent, invoke_agent
from tools import get_tools
from personas import get_persona_by_name
from content_creation import ContentCreator
from history import get_history_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for API Gateway integration"""
    
    # Gateway settings
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    
    # Service endpoints
    streamlit_host: str = "localhost"
    streamlit_port: int = 8501
    
    # Backend service
    backend_host: str = "localhost"
    backend_port: int = 8002
    
    # Integration features
    enable_async_processing: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_rate_limiting: bool = True
    
    # Authentication
    jwt_secret: str = "your-jwt-secret-key"
    api_key: str = "langgraph-101-api-key"
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls(
            gateway_host=os.getenv('GATEWAY_HOST', '0.0.0.0'),
            gateway_port=int(os.getenv('GATEWAY_PORT', 8000)),
            streamlit_host=os.getenv('STREAMLIT_HOST', 'localhost'),
            streamlit_port=int(os.getenv('STREAMLIT_PORT', 8501)),
            backend_host=os.getenv('BACKEND_HOST', 'localhost'),
            backend_port=int(os.getenv('BACKEND_PORT', 8002)),
            enable_async_processing=os.getenv('ENABLE_ASYNC_PROCESSING', 'true').lower() == 'true',
            enable_caching=os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            enable_monitoring=os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
            enable_rate_limiting=os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            jwt_secret=os.getenv('JWT_SECRET', 'your-jwt-secret-key'),
            api_key=os.getenv('API_KEY', 'langgraph-101-api-key')
        )

class LangGraphBackendService:
    """Backend service that wraps existing LangGraph functionality"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.agent = None
        self.content_creator = None
        self.history_manager = None
        self.tools = None
        self.app = FastAPI(title="LangGraph 101 Backend Service")
        
        # Initialize components
        self._initialize_components()
        self._setup_routes()
        self._setup_middleware()
        
        logger.info("LangGraph Backend Service initialized")
    
    def _initialize_components(self):
        """Initialize LangGraph components"""
        try:
            # Load configuration
            app_config = load_config()
            
            # Initialize tools and agent
            self.tools = get_tools()
            self.agent = create_agent(tools=self.tools)
            
            # Initialize content creator
            self.content_creator = ContentCreator()
            
            # Initialize history manager
            self.history_manager = get_history_manager()
            
            logger.info("LangGraph components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph components: {e}")
            raise
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "langgraph-backend",
                "version": "1.0.0"
            }
        
        @self.app.post("/api/chat")
        async def chat_endpoint(request: Request):
            """Main chat endpoint"""
            try:
                data = await request.json()
                message = data.get('message', '')
                persona_name = data.get('persona', 'default')
                session_id = data.get('session_id', str(uuid.uuid4()))
                
                # Get persona
                persona = get_persona_by_name(persona_name)
                if not persona:
                    raise HTTPException(status_code=400, detail=f"Persona '{persona_name}' not found")
                
                # Process message with agent
                result = await self._process_chat_message(message, persona, session_id)
                
                return {
                    "response": result,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "persona": persona_name
                }
                
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/content/create")
        async def content_creation_endpoint(request: Request):
            """Content creation endpoint"""
            try:
                data = await request.json()
                topic = data.get('topic', '')
                content_type = data.get('type', 'blog_post')
                options = data.get('options', {})
                
                # Create content
                result = await self._create_content(topic, content_type, options)
                
                return {
                    "content": result,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": content_type,
                    "topic": topic
                }
                
            except Exception as e:
                logger.error(f"Content creation endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/personas")
        async def list_personas():
            """List available personas"""
            try:
                from config import get_available_personas
                personas = get_available_personas()
                return {"personas": personas}
            except Exception as e:
                logger.error(f"List personas error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/history/{session_id}")
        async def get_conversation_history(session_id: str):
            """Get conversation history"""
            try:
                history = self.history_manager.get_conversation_history(session_id)
                return {"history": history, "session_id": session_id}
            except Exception as e:
                logger.error(f"Get history error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/export")
        async def export_conversation_endpoint(request: Request):
            """Export conversation endpoint"""
            try:
                data = await request.json()
                session_id = data.get('session_id', '')
                format_type = data.get('format', 'json')
                
                # Export conversation
                from export import export_conversation
                exported_data = export_conversation(session_id, format_type)
                
                return {
                    "data": exported_data,
                    "format": format_type,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Export endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_chat_message(self, message: str, persona: Any, session_id: str) -> str:
        """Process chat message with LangGraph agent"""
        try:
            # Update agent state with persona
            state = {
                "messages": [{"role": "user", "content": message}],
                "persona": persona.name if hasattr(persona, 'name') else 'default',
                "session_id": session_id
            }
            
            # Invoke agent
            result = await asyncio.to_thread(invoke_agent, self.agent, state)
            
            # Extract response
            if isinstance(result, dict) and 'messages' in result:
                messages = result['messages']
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and 'content' in last_message:
                        response = last_message['content']
                    else:
                        response = str(last_message)
                else:
                    response = "No response generated"
            else:
                response = str(result)
            
            # Save to history
            self.history_manager.save_interaction(session_id, message, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return f"Error processing message: {str(e)}"
    
    async def _create_content(self, topic: str, content_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Create content using ContentCreator"""
        try:
            # Create content using existing ContentCreator
            result = await asyncio.to_thread(
                self.content_creator.create_content,
                topic=topic,
                content_type=content_type,
                **options
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating content: {e}")
            raise
    
    def run(self):
        """Run the backend service"""
        logger.info(f"Starting LangGraph Backend Service on {self.config.backend_host}:{self.config.backend_port}")
        uvicorn.run(
            self.app,
            host=self.config.backend_host,
            port=self.config.backend_port,
            log_level="info"
        )

class IntegratedAPIGateway:
    """Enhanced API Gateway with LangGraph 101 integration"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.gateway = None
        self.message_queue = None
        self.cache_manager = None
        self.rate_limiter = None
        
        # Initialize components
        self._initialize_infrastructure()
        self._setup_services_and_routes()
        
        logger.info("Integrated API Gateway initialized")
    
    def _initialize_infrastructure(self):
        """Initialize infrastructure components"""
        # Initialize API Gateway
        gateway_config = APIGatewayConfig()
        gateway_config.host = self.config.gateway_host
        gateway_config.port = self.config.gateway_port
        gateway_config.jwt_secret = self.config.jwt_secret
        
        self.gateway = APIGateway(gateway_config)
        
        # Initialize message queue if enabled
        if self.config.enable_async_processing:
            try:
                self.message_queue = MessageQueueSystem()
                create_example_tasks(self.message_queue)
                logger.info("Message queue system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize message queue: {e}")
        
        # Initialize cache manager if enabled
        if self.config.enable_caching:
            try:
                self.cache_manager = CacheManager()
                logger.info("Cache manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize cache manager: {e}")
        
        # Initialize rate limiter if enabled
        if self.config.enable_rate_limiting:
            try:
                self.rate_limiter = EnhancedRateLimiter()
                logger.info("Enhanced rate limiter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize rate limiter: {e}")
    
    def _setup_services_and_routes(self):
        """Setup services and routes in the gateway"""
        
        # Register backend service
        backend_url = f"http://{self.config.backend_host}:{self.config.backend_port}"
        self.gateway.register_service(
            'langgraph-backend',
            [backend_url],
            weight=1,
            health_check_path='/health'
        )
        
        # Register Streamlit service (if running separately)
        streamlit_url = f"http://{self.config.streamlit_host}:{self.config.streamlit_port}"
        self.gateway.register_service(
            'streamlit-frontend',
            [streamlit_url],
            weight=1,
            health_check_path='/_stcore/health'
        )
        
        # Register routes
        self._register_api_routes()
        self._register_frontend_routes()
    
    def _register_api_routes(self):
        """Register API routes"""
        
        # Chat API routes
        self.gateway.register_route(
            '/api/chat',
            'langgraph-backend',
            methods=['POST'],
            auth_required=True,
            rate_limit=100,  # 100 requests per hour
            cache_ttl=None,  # Don't cache chat responses
            timeout=60.0
        )
        
        # Content creation routes
        self.gateway.register_route(
            '/api/content/*',
            'langgraph-backend',
            methods=['GET', 'POST'],
            auth_required=True,
            rate_limit=50,  # 50 requests per hour
            cache_ttl=300,  # Cache for 5 minutes
            timeout=120.0
        )
        
        # Personas and configuration routes
        self.gateway.register_route(
            '/api/personas',
            'langgraph-backend',
            methods=['GET'],
            auth_required=False,
            rate_limit=200,
            cache_ttl=600  # Cache for 10 minutes
        )
        
        # History routes
        self.gateway.register_route(
            '/api/history/*',
            'langgraph-backend',
            methods=['GET', 'POST'],
            auth_required=True,
            rate_limit=150,
            cache_ttl=60  # Cache for 1 minute
        )
        
        # Export routes
        self.gateway.register_route(
            '/api/export',
            'langgraph-backend',
            methods=['POST'],
            auth_required=True,
            rate_limit=20,  # Lower limit for export operations
            cache_ttl=None
        )
        
        # Health and monitoring routes
        self.gateway.register_route(
            '/health',
            'langgraph-backend',
            methods=['GET'],
            auth_required=False,
            rate_limit=1000,
            cache_ttl=30
        )
    
    def _register_frontend_routes(self):
        """Register frontend routes"""
        
        # Streamlit frontend routes
        self.gateway.register_route(
            '/',
            'streamlit-frontend',
            methods=['GET'],
            auth_required=False,
            rate_limit=500,
            cache_ttl=60
        )
        
        # Static assets and Streamlit internal routes
        self.gateway.register_route(
            '/static/*',
            'streamlit-frontend',
            methods=['GET'],
            auth_required=False,
            rate_limit=1000,
            cache_ttl=3600  # Cache static assets for 1 hour
        )
        
        self.gateway.register_route(
            '/_stcore/*',
            'streamlit-frontend',
            methods=['GET', 'POST'],
            auth_required=False,
            rate_limit=1000,
            cache_ttl=None
        )
    
    def add_custom_routes(self):
        """Add custom integration routes"""
        
        @self.gateway.app.get("/integration/status")
        async def integration_status():
            """Get integration status"""
            return {
                "status": "active",
                "components": {
                    "api_gateway": "running",
                    "message_queue": "running" if self.message_queue else "disabled",
                    "cache_manager": "running" if self.cache_manager else "disabled",
                    "rate_limiter": "running" if self.rate_limiter else "disabled"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.gateway.app.post("/integration/async-task")
        async def submit_async_task(request: Request):
            """Submit async task through message queue"""
            if not self.message_queue:
                raise HTTPException(status_code=503, detail="Message queue not available")
            
            try:
                data = await request.json()
                task_type = data.get('task_type', 'default')
                task_data = data.get('data', {})
                priority = TaskPriority(data.get('priority', 'normal'))
                
                # Submit task
                task_id = self.message_queue.submit_task(
                    task_type,
                    task_data,
                    priority=priority
                )
                
                return {
                    "task_id": task_id,
                    "status": "submitted",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error submitting async task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.gateway.app.get("/integration/cache/stats")
        async def cache_stats():
            """Get cache statistics"""
            if not self.cache_manager:
                raise HTTPException(status_code=503, detail="Cache manager not available")
            
            try:
                stats = await self.cache_manager.get_stats()
                return {
                    "cache_stats": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """Run the integrated gateway"""
        # Add custom routes
        self.add_custom_routes()
        
        # Start the gateway
        logger.info(f"Starting Integrated API Gateway on {self.config.gateway_host}:{self.config.gateway_port}")
        self.gateway.run()

def create_streamlit_client_adapter():
    """Create adapter for Streamlit to use the API Gateway"""
    
    class StreamlitAPIClient:
        """API client for Streamlit to communicate through gateway"""
        
        def __init__(self, gateway_url: str, api_key: str):
            self.gateway_url = gateway_url.rstrip('/')
            self.api_key = api_key
            self.session = None
        
        def _get_headers(self) -> Dict[str, str]:
            """Get headers for API requests"""
            return {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key,
                'User-Agent': 'LangGraph101-Streamlit/1.0'
            }
        
        async def chat(self, message: str, persona: str = 'default', session_id: str = None) -> Dict[str, Any]:
            """Send chat message through gateway"""
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gateway_url}/api/chat",
                    json={
                        'message': message,
                        'persona': persona,
                        'session_id': session_id or str(uuid.uuid4())
                    },
                    headers=self._get_headers(),
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()
        
        async def create_content(self, topic: str, content_type: str = 'blog_post', options: Dict[str, Any] = None) -> Dict[str, Any]:
            """Create content through gateway"""
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gateway_url}/api/content/create",
                    json={
                        'topic': topic,
                        'type': content_type,
                        'options': options or {}
                    },
                    headers=self._get_headers(),
                    timeout=120.0
                )
                response.raise_for_status()
                return response.json()
        
        async def get_personas(self) -> Dict[str, Any]:
            """Get available personas through gateway"""
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.gateway_url}/api/personas",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        
        async def get_history(self, session_id: str) -> Dict[str, Any]:
            """Get conversation history through gateway"""
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.gateway_url}/api/history/{session_id}",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        
        async def export_conversation(self, session_id: str, format_type: str = 'json') -> Dict[str, Any]:
            """Export conversation through gateway"""
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gateway_url}/api/export",
                    json={
                        'session_id': session_id,
                        'format': format_type
                    },
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
    
    return StreamlitAPIClient

def create_deployment_scripts():
    """Create deployment scripts for the integrated system"""
    
    # Docker Compose configuration
    docker_compose_content = """version: '3.8'

services:
  # API Gateway
  api-gateway:
    build: .
    command: python api_gateway_integration.py --mode gateway
    ports:
      - "8000:8000"
    environment:
      - GATEWAY_HOST=0.0.0.0
      - GATEWAY_PORT=8000
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET:-your-jwt-secret}
      - API_KEY=${API_KEY:-langgraph-101-api-key}
    depends_on:
      - redis
      - backend
    networks:
      - langgraph-network
  
  # Backend Service
  backend:
    build: .
    command: python api_gateway_integration.py --mode backend
    ports:
      - "8002:8002"
    environment:
      - BACKEND_HOST=0.0.0.0
      - BACKEND_PORT=8002
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    networks:
      - langgraph-network
  
  # Streamlit Frontend
  streamlit:
    build: .
    command: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - GATEWAY_URL=http://api-gateway:8000
      - API_KEY=${API_KEY:-langgraph-101-api-key}
    depends_on:
      - api-gateway
    networks:
      - langgraph-network
  
  # Redis for caching and message queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - langgraph-network
  
  # Celery Worker for async tasks
  celery-worker:
    build: .
    command: python message_queue_system.py worker
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    networks:
      - langgraph-network

networks:
  langgraph-network:
    driver: bridge
"""
    
    # Startup script
    startup_script = """#!/bin/bash
# LangGraph 101 - Integrated System Startup Script

echo "Starting LangGraph 101 Integrated System..."

# Check if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "Starting with Docker Compose..."
    docker-compose up -d
else
    echo "Starting services manually..."
    
    # Start Redis (if available)
    if command -v redis-server &> /dev/null; then
        echo "Starting Redis..."
        redis-server --daemonize yes
    fi
    
    # Start backend service
    echo "Starting backend service..."
    python api_gateway_integration.py --mode backend &
    BACKEND_PID=$!
    
    # Wait for backend to start
    sleep 5
    
    # Start API Gateway
    echo "Starting API Gateway..."
    python api_gateway_integration.py --mode gateway &
    GATEWAY_PID=$!
    
    # Wait for gateway to start
    sleep 5
    
    # Start Streamlit
    echo "Starting Streamlit frontend..."
    GATEWAY_URL=http://localhost:8000 streamlit run streamlit_app.py &
    STREAMLIT_PID=$!
    
    echo "All services started!"
    echo "API Gateway: http://localhost:8000"
    echo "Streamlit UI: http://localhost:8501"
    echo "Backend API: http://localhost:8002"
    
    # Wait for interruption
    trap "kill $BACKEND_PID $GATEWAY_PID $STREAMLIT_PID" EXIT
    wait
fi
"""
    
    return docker_compose_content, startup_script

def main():
    """Main function for running the integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangGraph 101 API Gateway Integration')
    parser.add_argument('--mode', choices=['gateway', 'backend', 'both'], default='both',
                       help='Run mode: gateway, backend, or both')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = IntegrationConfig.from_env()
    
    try:
        if args.mode == 'backend':
            # Run backend service only
            backend = LangGraphBackendService(config)
            backend.run()
        
        elif args.mode == 'gateway':
            # Run API gateway only
            gateway = IntegratedAPIGateway(config)
            gateway.run()
        
        else:
            # Run both in separate processes
            import multiprocessing
            
            def run_backend():
                backend = LangGraphBackendService(config)
                backend.run()
            
            def run_gateway():
                # Wait for backend to start
                time.sleep(5)
                gateway = IntegratedAPIGateway(config)
                gateway.run()
            
            # Start processes
            backend_process = multiprocessing.Process(target=run_backend)
            gateway_process = multiprocessing.Process(target=run_gateway)
            
            backend_process.start()
            gateway_process.start()
            
            try:
                backend_process.join()
                gateway_process.join()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                backend_process.terminate()
                gateway_process.terminate()
                backend_process.join()
                gateway_process.join()
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running integration: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
