#!/usr/bin/env python3
"""
LangGraph 101 - API Gateway System
=================================

Comprehensive API gateway providing centralized request routing, authentication,
rate limiting, load balancing, and monitoring for the LangGraph 101 platform.

Features:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Caching and optimization
- Security filtering and validation
- Monitoring and analytics
- Circuit breaker pattern
- Health checks and service discovery
- WebSocket support

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import ipaddress
import hashlib
import hmac
from urllib.parse import urlparse, urljoin
import re

# Third-party imports
import aiohttp
try:
    from aioredis_compat import Redis, from_url, get_redis_client
    aioredis_available = True
except ImportError:
    import aioredis
    from aioredis import Redis
    from aioredis import from_url
    aioredis_available = True
    
    def get_redis_client(url="redis://localhost:6379", **kwargs):
        return from_url(url, **kwargs)
from fastapi import FastAPI, Request, Response, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import httpx
import jwt
from cryptography.fernet import Fernet
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadBalancerStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class HealthStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    url: str
    weight: int = 1
    max_connections: int = 100
    timeout: float = 30.0
    health_check_path: str = "/health"
    health_check_interval: int = 30
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    circuit_breaker_failures: int = 0
    circuit_breaker_last_failure: Optional[datetime] = None

@dataclass
class RouteConfig:
    """Route configuration"""
    path: str
    methods: List[str]
    service_name: str
    strip_path: bool = True
    preserve_host: bool = False
    auth_required: bool = True
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None
    timeout: float = 30.0
    retry_count: int = 3
    transform_request: Optional[str] = None
    transform_response: Optional[str] = None

@dataclass
class GatewayMetrics:
    """Gateway metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    blocked_requests: int = 0
    average_response_time: float = 0.0
    active_connections: int = 0
    bytes_transferred: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

class APIGatewayConfig:
    """Configuration for API Gateway"""
    
    def __init__(self):
        # Server configuration
        self.host = os.getenv('GATEWAY_HOST', '0.0.0.0')
        self.port = int(os.getenv('GATEWAY_PORT', 8000))
        self.workers = int(os.getenv('GATEWAY_WORKERS', 1))
        
        # Security configuration
        self.jwt_secret = os.getenv('JWT_SECRET', 'your-secret-key')
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.api_key_header = os.getenv('API_KEY_HEADER', 'X-API-Key')
        self.encryption_key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        
        # Rate limiting
        self.default_rate_limit = int(os.getenv('DEFAULT_RATE_LIMIT', 1000))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', 3600))
        
        # Caching
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.default_cache_ttl = int(os.getenv('DEFAULT_CACHE_TTL', 300))
        
        # Redis configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Load balancing
        self.load_balancer_strategy = LoadBalancerStrategy(
            os.getenv('LOAD_BALANCER_STRATEGY', 'round_robin')
        )
        
        # Circuit breaker
        self.circuit_breaker_threshold = int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', 5))
        self.circuit_breaker_timeout = int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', 60))
        
        # Health checks
        self.health_check_enabled = os.getenv('HEALTH_CHECK_ENABLED', 'true').lower() == 'true'
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', 30))
        
        # Monitoring
        self.metrics_enabled = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        self.detailed_logging = os.getenv('DETAILED_LOGGING', 'false').lower() == 'true'

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int, timeout: int):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self._lock = threading.RLock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise HTTPException(status_code=503, detail="Service unavailable (circuit breaker open)")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        return datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.timeout)
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Reset after 3 successful calls
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class LoadBalancer:
    """Load balancer implementation"""
    
    def __init__(self, strategy: LoadBalancerStrategy):
        self.strategy = strategy
        self.current_index = 0
        self._lock = threading.RLock()
    
    def select_endpoint(self, endpoints: List[ServiceEndpoint], request_ip: str = None) -> Optional[ServiceEndpoint]:
        """Select an endpoint based on load balancing strategy"""
        healthy_endpoints = [ep for ep in endpoints if ep.health_status == HealthStatus.HEALTHY]
        
        if not healthy_endpoints:
            # Fallback to degraded endpoints if no healthy ones
            healthy_endpoints = [ep for ep in endpoints if ep.health_status == HealthStatus.DEGRADED]
        
        if not healthy_endpoints:
            return None
        
        with self._lock:
            if self.strategy == LoadBalancerStrategy.ROUND_ROBIN:
                return self._round_robin(healthy_endpoints)
            elif self.strategy == LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin(healthy_endpoints)
            elif self.strategy == LoadBalancerStrategy.LEAST_CONNECTIONS:
                return self._least_connections(healthy_endpoints)
            elif self.strategy == LoadBalancerStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time(healthy_endpoints)
            elif self.strategy == LoadBalancerStrategy.IP_HASH:
                return self._ip_hash(healthy_endpoints, request_ip)
            elif self.strategy == LoadBalancerStrategy.RANDOM:
                return self._random(healthy_endpoints)
            else:
                return healthy_endpoints[0]
    
    def _round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin selection"""
        endpoint = endpoints[self.current_index % len(endpoints)]
        self.current_index += 1
        return endpoint
    
    def _weighted_round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round robin selection"""
        total_weight = sum(ep.weight for ep in endpoints)
        weighted_index = self.current_index % total_weight
        
        current_weight = 0
        for endpoint in endpoints:
            current_weight += endpoint.weight
            if weighted_index < current_weight:
                self.current_index += 1
                return endpoint
        
        return endpoints[0]
    
    def _least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections selection"""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _least_response_time(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least response time selection"""
        return min(endpoints, key=lambda ep: ep.average_response_time)
    
    def _ip_hash(self, endpoints: List[ServiceEndpoint], request_ip: str) -> ServiceEndpoint:
        """IP hash selection"""
        if not request_ip:
            return endpoints[0]
        
        hash_value = hash(request_ip)
        index = hash_value % len(endpoints)
        return endpoints[index]
    
    def _random(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random selection"""
        import random
        return random.choice(endpoints)

class RequestTransformer:
    """Request/response transformation handler"""
    
    def __init__(self):
        self.transformations = {}
    
    def register_transformation(self, name: str, transform_func: Callable):
        """Register a transformation function"""
        self.transformations[name] = transform_func
    
    def transform_request(self, request_data: Dict[str, Any], transformation: str) -> Dict[str, Any]:
        """Apply request transformation"""
        if transformation in self.transformations:
            return self.transformations[transformation](request_data)
        return request_data
    
    def transform_response(self, response_data: Dict[str, Any], transformation: str) -> Dict[str, Any]:
        """Apply response transformation"""
        if transformation in self.transformations:
            return self.transformations[transformation](response_data)
        return response_data

class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self, config: Optional[APIGatewayConfig] = None):
        self.config = config or APIGatewayConfig()
        self.app = FastAPI(title="LangGraph 101 API Gateway", version="1.0.0")
        self.services = {}  # service_name -> List[ServiceEndpoint]
        self.routes = {}    # path_pattern -> RouteConfig
        self.load_balancer = LoadBalancer(self.config.load_balancer_strategy)
        self.circuit_breakers = {}  # service_name -> CircuitBreaker
        self.transformer = RequestTransformer()
        self.metrics = GatewayMetrics()
        self.redis_client = None
        self.http_client = None
        self.health_check_task = None
        self._lock = threading.RLock()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_websocket()
        
        logger.info("API Gateway initialized successfully")
    
    async def startup(self):
        """Initialize async components"""
        # Initialize Redis connection
        try:
            self.redis_client = aioredis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for caching")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
        )
        
        # Start health check task
        if self.config.health_check_enabled:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("API Gateway startup complete")
    
    async def shutdown(self):
        """Cleanup async components"""
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("API Gateway shutdown complete")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware
        self.app.add_middleware(GatewayMetricsMiddleware, gateway=self)
        self.app.add_middleware(SecurityMiddleware, gateway=self)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "LangGraph 101 API Gateway", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health():
            return await self.get_health_status()
        
        @self.app.get("/metrics")
        async def metrics():
            return asdict(self.metrics)
        
        @self.app.get("/services")
        async def services():
            return {name: [asdict(ep) for ep in endpoints] 
                   for name, endpoints in self.services.items()}
        
        @self.app.get("/routes")
        async def routes():
            return {path: asdict(config) for path, config in self.routes.items()}
        
        # Dynamic route handler
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        async def proxy_request(request: Request, path: str):
            return await self._handle_request(request, path)
    
    def _setup_websocket(self):
        """Setup WebSocket support"""
        
        @self.app.websocket("/ws/{path:path}")
        async def websocket_proxy(websocket: WebSocket, path: str):
            await self._handle_websocket(websocket, path)
    
    def register_service(self, name: str, endpoints: List[str], **kwargs):
        """Register a service with multiple endpoints"""
        service_endpoints = []
        
        for endpoint_url in endpoints:
            endpoint = ServiceEndpoint(
                url=endpoint_url,
                weight=kwargs.get('weight', 1),
                max_connections=kwargs.get('max_connections', 100),
                timeout=kwargs.get('timeout', 30.0),
                health_check_path=kwargs.get('health_check_path', '/health'),
                health_check_interval=kwargs.get('health_check_interval', 30)
            )
            service_endpoints.append(endpoint)
        
        self.services[name] = service_endpoints
        
        # Create circuit breaker for service
        self.circuit_breakers[name] = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        
        logger.info(f"Registered service '{name}' with {len(endpoints)} endpoints")
    
    def register_route(self, 
                      path: str, 
                      service_name: str, 
                      methods: List[str] = None,
                      **kwargs):
        """Register a route configuration"""
        if methods is None:
            methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        
        route_config = RouteConfig(
            path=path,
            methods=methods,
            service_name=service_name,
            strip_path=kwargs.get('strip_path', True),
            preserve_host=kwargs.get('preserve_host', False),
            auth_required=kwargs.get('auth_required', True),
            rate_limit=kwargs.get('rate_limit'),
            cache_ttl=kwargs.get('cache_ttl'),
            timeout=kwargs.get('timeout', 30.0),
            retry_count=kwargs.get('retry_count', 3),
            transform_request=kwargs.get('transform_request'),
            transform_response=kwargs.get('transform_response')
        )
        
        self.routes[path] = route_config
        logger.info(f"Registered route '{path}' -> service '{service_name}'")
    
    async def _handle_request(self, request: Request, path: str) -> Response:
        """Handle incoming HTTP request"""
        start_time = time.time()
        
        try:
            # Find matching route
            route_config = self._find_route(path, request.method)
            if not route_config:
                raise HTTPException(status_code=404, detail="Route not found")
            
            # Check rate limiting
            if route_config.rate_limit:
                await self._check_rate_limit(request, route_config.rate_limit)
            
            # Check authentication
            if route_config.auth_required:
                await self._check_authentication(request)
            
            # Check cache
            cache_key = None
            if self.config.cache_enabled and route_config.cache_ttl and request.method == "GET":
                cache_key = self._generate_cache_key(request, path)
                cached_response = await self._get_cached_response(cache_key)
                if cached_response:
                    self.metrics.cached_requests += 1
                    return JSONResponse(content=cached_response)
            
            # Get service endpoint
            service_endpoints = self.services.get(route_config.service_name)
            if not service_endpoints:
                raise HTTPException(status_code=502, detail="Service not available")
            
            client_ip = self._get_client_ip(request)
            endpoint = self.load_balancer.select_endpoint(service_endpoints, client_ip)
            if not endpoint:
                raise HTTPException(status_code=503, detail="No healthy endpoints available")
            
            # Proxy request with circuit breaker
            circuit_breaker = self.circuit_breakers.get(route_config.service_name)
            response = await circuit_breaker.call(
                self._proxy_request,
                request, endpoint, route_config, path
            )
            
            # Cache response if configured
            if cache_key and route_config.cache_ttl:
                await self._cache_response(cache_key, response, route_config.cache_ttl)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time, endpoint)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling request to {path}: {e}")
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _proxy_request(self, 
                           request: Request, 
                           endpoint: ServiceEndpoint, 
                           route_config: RouteConfig, 
                           path: str) -> Response:
        """Proxy request to backend service"""
        # Prepare target URL
        target_path = path
        if route_config.strip_path and route_config.path != "/":
            target_path = path.replace(route_config.path.rstrip('/'), '', 1)
        
        target_url = urljoin(endpoint.url, target_path.lstrip('/'))
        
        # Prepare headers
        headers = dict(request.headers)
        if not route_config.preserve_host:
            headers['host'] = urlparse(endpoint.url).netloc
        
        # Remove hop-by-hop headers
        hop_by_hop = ['connection', 'keep-alive', 'proxy-authenticate', 
                     'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']
        for header in hop_by_hop:
            headers.pop(header, None)
        
        # Prepare request data
        request_data = {
            'method': request.method,
            'url': target_url,
            'headers': headers,
            'params': dict(request.query_params),
            'timeout': route_config.timeout
        }
        
        # Add body for non-GET requests
        if request.method != 'GET':
            try:
                body = await request.body()
                if body:
                    request_data['content'] = body
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")
        
        # Apply request transformation
        if route_config.transform_request:
            request_data = self.transformer.transform_request(request_data, route_config.transform_request)
        
        # Update endpoint connection count
        endpoint.current_connections += 1
        
        try:
            # Make request with retries
            for attempt in range(route_config.retry_count + 1):
                try:
                    response = await self.http_client.request(**request_data)
                    
                    # Update endpoint metrics
                    endpoint.total_requests += 1
                    response_time = response.elapsed.total_seconds() if response.elapsed else 0
                    endpoint.average_response_time = (
                        (endpoint.average_response_time * (endpoint.total_requests - 1) + response_time) / 
                        endpoint.total_requests
                    )
                    
                    # Prepare response
                    response_headers = dict(response.headers)
                    
                    # Remove hop-by-hop headers
                    for header in hop_by_hop:
                        response_headers.pop(header, None)
                    
                    # Apply response transformation
                    response_content = response.content
                    if route_config.transform_response and response.headers.get('content-type', '').startswith('application/json'):
                        try:
                            json_content = response.json()
                            transformed = self.transformer.transform_response(json_content, route_config.transform_response)
                            response_content = json.dumps(transformed).encode()
                            response_headers['content-length'] = str(len(response_content))
                        except Exception as e:
                            logger.warning(f"Failed to transform response: {e}")
                    
                    return Response(
                        content=response_content,
                        status_code=response.status_code,
                        headers=response_headers
                    )
                    
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt < route_config.retry_count:
                        logger.warning(f"Request attempt {attempt + 1} failed: {e}, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        endpoint.failed_requests += 1
                        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
                
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500:
                        endpoint.failed_requests += 1
                    raise HTTPException(status_code=e.response.status_code, detail="Upstream error")
        
        finally:
            endpoint.current_connections = max(0, endpoint.current_connections - 1)
    
    async def _handle_websocket(self, websocket: WebSocket, path: str):
        """Handle WebSocket connection"""
        await websocket.accept()
        
        try:
            # Find matching route
            route_config = self._find_route(path, "GET")  # Use GET for WebSocket route matching
            if not route_config:
                await websocket.close(code=404, reason="Route not found")
                return
            
            # Get service endpoint
            service_endpoints = self.services.get(route_config.service_name)
            if not service_endpoints:
                await websocket.close(code=502, reason="Service not available")
                return
            
            endpoint = self.load_balancer.select_endpoint(service_endpoints)
            if not endpoint:
                await websocket.close(code=503, reason="No healthy endpoints available")
                return
            
            # Convert HTTP endpoint to WebSocket URL
            ws_url = endpoint.url.replace('http://', 'ws://').replace('https://', 'wss://')
            target_path = path
            if route_config.strip_path and route_config.path != "/":
                target_path = path.replace(route_config.path.rstrip('/'), '', 1)
            
            target_url = urljoin(ws_url, target_path.lstrip('/'))
            
            # Create upstream WebSocket connection
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(target_url) as ws_upstream:
                    # Proxy messages in both directions
                    async def client_to_upstream():
                        try:
                            async for message in websocket.iter_text():
                                await ws_upstream.send_str(message)
                        except WebSocketDisconnect:
                            pass
                    
                    async def upstream_to_client():
                        try:
                            async for msg in ws_upstream:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    await websocket.send_text(msg.data)
                                elif msg.type == aiohttp.WSMsgType.BINARY:
                                    await websocket.send_bytes(msg.data)
                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    break
                        except Exception:
                            pass
                    
                    # Run both directions concurrently
                    await asyncio.gather(
                        client_to_upstream(),
                        upstream_to_client(),
                        return_exceptions=True
                    )
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if not websocket.client_state.DISCONNECTED:
                await websocket.close()
    
    def _find_route(self, path: str, method: str) -> Optional[RouteConfig]:
        """Find matching route configuration"""
        # Try exact match first
        if path in self.routes:
            route = self.routes[path]
            if method in route.methods:
                return route
        
        # Try pattern matching
        for route_path, route_config in self.routes.items():
            if method not in route_config.methods:
                continue
            
            # Convert route pattern to regex
            pattern = route_path.replace('*', '.*').replace('{', '(?P<').replace('}', '>[^/]+)')
            if re.match(f"^{pattern}$", path):
                return route_config
        
        return None
    
    async def _check_rate_limit(self, request: Request, rate_limit: int):
        """Check rate limiting"""
        if not self.redis_client:
            return  # Skip if Redis not available
        
        client_ip = self._get_client_ip(request)
        key = f"rate_limit:{client_ip}"
        
        try:
            current = await self.redis_client.get(key)
            if current is None:
                await self.redis_client.setex(key, self.config.rate_limit_window, 1)
            else:
                current_count = int(current)
                if current_count >= rate_limit:
                    self.metrics.blocked_requests += 1
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                await self.redis_client.incr(key)
        
        except Exception as e:
            logger.warning(f"Rate limiting error: {e}")
    
    async def _check_authentication(self, request: Request):
        """Check authentication"""
        # Check for JWT token
        auth_header = request.headers.get('authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
                return
            except jwt.InvalidTokenError:
                pass
        
        # Check for API key
        api_key = request.headers.get(self.config.api_key_header)
        if api_key:
            # Validate API key (implement your validation logic)
            if self._validate_api_key(api_key):
                return
        
        raise HTTPException(status_code=401, detail="Authentication required")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key (implement your logic)"""
        # Placeholder - implement your API key validation
        return api_key == "valid-api-key"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _generate_cache_key(self, request: Request, path: str) -> str:
        """Generate cache key for request"""
        key_parts = [
            request.method,
            path,
            str(sorted(request.query_params.items())),
            request.headers.get('authorization', ''),
        ]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(f"cache:{cache_key}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Response, ttl: int):
        """Cache response"""
        if not self.redis_client:
            return
        
        try:
            # Only cache successful JSON responses
            if response.status_code == 200 and response.headers.get('content-type', '').startswith('application/json'):
                response_data = {
                    'body': response.body.decode() if hasattr(response, 'body') else None,
                    'headers': dict(response.headers),
                    'status_code': response.status_code
                }
                await self.redis_client.setex(
                    f"cache:{cache_key}",
                    ttl,
                    json.dumps(response_data)
                )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _update_metrics(self, success: bool, execution_time: float, endpoint: ServiceEndpoint = None):
        """Update gateway metrics"""
        with self._lock:
            self.metrics.total_requests += 1
            
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
            
            # Update average response time
            total_responses = self.metrics.successful_requests + self.metrics.failed_requests
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (total_responses - 1) + execution_time) / 
                total_responses
            )
            
            self.metrics.last_updated = datetime.utcnow()
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """Perform health checks on all service endpoints"""
        for service_name, endpoints in self.services.items():
            for endpoint in endpoints:
                try:
                    health_url = urljoin(endpoint.url, endpoint.health_check_path)
                    response = await self.http_client.get(health_url, timeout=10.0)
                    
                    if response.status_code == 200:
                        endpoint.health_status = HealthStatus.HEALTHY
                    elif 200 <= response.status_code < 300:
                        endpoint.health_status = HealthStatus.DEGRADED
                    else:
                        endpoint.health_status = HealthStatus.UNHEALTHY
                
                except Exception:
                    endpoint.health_status = HealthStatus.UNHEALTHY
                
                endpoint.last_health_check = datetime.utcnow()
        
        logger.debug("Health checks completed")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        service_health = {}
        overall_healthy = True
        
        for service_name, endpoints in self.services.items():
            healthy_count = sum(1 for ep in endpoints if ep.health_status == HealthStatus.HEALTHY)
            total_count = len(endpoints)
            
            service_health[service_name] = {
                'healthy_endpoints': healthy_count,
                'total_endpoints': total_count,
                'status': 'healthy' if healthy_count > 0 else 'unhealthy'
            }
            
            if healthy_count == 0:
                overall_healthy = False
        
        return {
            'status': 'healthy' if overall_healthy else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'services': service_health,
            'metrics': asdict(self.metrics)
        }
    
    def run(self):
        """Run the API Gateway"""
        # Setup startup and shutdown events
        @self.app.on_event("startup")
        async def startup_event():
            await self.startup()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown()
        
        # Run with uvicorn
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            access_log=self.config.detailed_logging
        )


class GatewayMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting gateway metrics"""
    
    def __init__(self, app: ASGIApp, gateway: APIGateway):
        super().__init__(app)
        self.gateway = gateway
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Update active connections
        self.gateway.metrics.active_connections += 1
        
        try:
            response = await call_next(request)
            
            # Update metrics
            execution_time = time.time() - start_time
            content_length = int(response.headers.get('content-length', 0))
            self.gateway.metrics.bytes_transferred += content_length
            
            return response
        
        finally:
            self.gateway.metrics.active_connections = max(0, self.gateway.metrics.active_connections - 1)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for the gateway"""
    
    def __init__(self, app: ASGIApp, gateway: APIGateway):
        super().__init__(app)
        self.gateway = gateway
    
    async def dispatch(self, request: Request, call_next):
        # Add security headers
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


# Example usage and configuration
def create_example_gateway():
    """Create example gateway with sample services"""
    config = APIGatewayConfig()
    gateway = APIGateway(config)
    
    # Register services
    gateway.register_service(
        'auth-service',
        ['http://localhost:8001', 'http://localhost:8002'],
        weight=2,
        health_check_path='/health'
    )
    
    gateway.register_service(
        'content-service',
        ['http://localhost:8003', 'http://localhost:8004'],
        weight=1,
        health_check_path='/status'
    )
    
    gateway.register_service(
        'ai-service',
        ['http://localhost:8005'],
        weight=3,
        timeout=60.0
    )
    
    # Register routes
    gateway.register_route('/api/auth/*', 'auth-service', auth_required=False, rate_limit=100)
    gateway.register_route('/api/content/*', 'content-service', cache_ttl=300)
    gateway.register_route('/api/ai/*', 'ai-service', timeout=60.0, retry_count=2)
    
    return gateway


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangGraph 101 API Gateway')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create and configure gateway
    config = APIGatewayConfig()
    config.host = args.host
    config.port = args.port
    config.workers = args.workers
    
    gateway = create_example_gateway()
    gateway.config = config
    
    try:
        logger.info(f"Starting API Gateway on {args.host}:{args.port}")
        gateway.run()
    except KeyboardInterrupt:
        logger.info("Gateway stopped by user")


if __name__ == '__main__':
    main()
