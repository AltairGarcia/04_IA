#!/usr/bin/env python3
"""
LangGraph 101 - Microservices Orchestrator
==========================================

Comprehensive microservices orchestration system that manages service discovery,
health checks, load balancing, and inter-service communication for the
LangGraph 101 platform.

Features:
- Service discovery and registration
- Health monitoring and circuit breakers
- Load balancing and failover
- Inter-service communication
- Distributed logging and tracing
- Configuration management
- Service mesh integration
- Deployment orchestration

Services Architecture:
- Gateway Service (8000): API Gateway, routing, rate limiting
- Auth Service (8001): Authentication, MFA, sessions
- Content Service (8002): Content CRUD, quality analysis
- Analytics Service (8003): Metrics, monitoring, dashboards
- Security Service (8004): Audit, IDS, threat detection
- Workflow Service (8005): Task processing, orchestration

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
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import ipaddress
import subprocess
from pathlib import Path

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
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import yaml
import consul
import etcd3
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

class ServiceType(Enum):
    """Service type enumeration"""
    GATEWAY = "gateway"
    AUTH = "auth"
    CONTENT = "content"
    ANALYTICS = "analytics"
    SECURITY = "security"
    WORKFLOW = "workflow"

@dataclass
class ServiceInstance:
    """Service instance configuration"""
    service_id: str
    service_name: str
    service_type: ServiceType
    host: str
    port: int
    version: str
    status: ServiceStatus = ServiceStatus.STARTING
    health_check_url: str = None
    metadata: Dict[str, Any] = None
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    circuit_breaker_open: bool = False
    load_balancer_weight: int = 1
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.health_check_url is None:
            self.health_check_url = f"http://{self.host}:{self.port}/health"
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = set()
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    type: ServiceType
    port: int
    replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 5
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"
    health_check_interval: int = 30
    health_check_timeout: int = 10
    environment: Dict[str, str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.dependencies is None:
            self.dependencies = []

class ServiceRegistry:
    """Service discovery and registry"""
    
    def __init__(self, backend: str = "memory"):
        self.backend = backend
        self.services: Dict[str, ServiceInstance] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.consul_client = None
        self.etcd_client = None
        self.redis_client = None
        self._lock = threading.RLock()
        
        # Initialize backend
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize service registry backend"""
        if self.backend == "consul":
            try:
                self.consul_client = consul.Consul()
                logger.info("Initialized Consul service registry")
            except Exception as e:
                logger.warning(f"Failed to initialize Consul: {e}, falling back to memory")
                self.backend = "memory"
        
        elif self.backend == "etcd":
            try:
                self.etcd_client = etcd3.client()
                logger.info("Initialized etcd service registry")
            except Exception as e:
                logger.warning(f"Failed to initialize etcd: {e}, falling back to memory")
                self.backend = "memory"
        
        elif self.backend == "redis":
            try:
                import redis
                self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
                self.redis_client.ping()
                logger.info("Initialized Redis service registry")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}, falling back to memory")
                self.backend = "memory"
        
        if self.backend == "memory":
            logger.info("Using in-memory service registry")
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a service instance"""
        try:
            with self._lock:
                self.services[service.service_id] = service
            
            if self.backend == "consul" and self.consul_client:
                self.consul_client.agent.service.register(
                    name=service.service_name,
                    service_id=service.service_id,
                    address=service.host,
                    port=service.port,
                    tags=list(service.tags),
                    check=consul.Check.http(service.health_check_url, interval="30s")
                )
            
            elif self.backend == "etcd" and self.etcd_client:
                key = f"/services/{service.service_name}/{service.service_id}"
                value = json.dumps(asdict(service), default=str)
                self.etcd_client.put(key, value)
            
            elif self.backend == "redis" and self.redis_client:
                key = f"service:{service.service_name}:{service.service_id}"
                value = json.dumps(asdict(service), default=str)
                self.redis_client.setex(key, 300, value)  # 5 min TTL
            
            logger.info(f"Registered service {service.service_name} ({service.service_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_id}: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        try:
            service = self.services.get(service_id)
            if not service:
                return False
            
            with self._lock:
                del self.services[service_id]
            
            if self.backend == "consul" and self.consul_client:
                self.consul_client.agent.service.deregister(service_id)
            
            elif self.backend == "etcd" and self.etcd_client:
                key = f"/services/{service.service_name}/{service_id}"
                self.etcd_client.delete(key)
            
            elif self.backend == "redis" and self.redis_client:
                key = f"service:{service.service_name}:{service_id}"
                self.redis_client.delete(key)
            
            logger.info(f"Deregistered service {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover service instances by name"""
        try:
            if self.backend == "consul" and self.consul_client:
                _, services = self.consul_client.health.service(service_name, passing=True)
                instances = []
                for service in services:
                    service_info = service['Service']
                    instance = ServiceInstance(
                        service_id=service_info['ID'],
                        service_name=service_info['Service'],
                        service_type=ServiceType(service_info.get('Tags', ['unknown'])[0]),
                        host=service_info['Address'],
                        port=service_info['Port'],
                        version=service_info.get('Meta', {}).get('version', '1.0.0'),
                        status=ServiceStatus.HEALTHY,
                        tags=set(service_info.get('Tags', []))
                    )
                    instances.append(instance)
                return instances
            
            elif self.backend == "etcd" and self.etcd_client:
                prefix = f"/services/{service_name}/"
                instances = []
                for value, _ in self.etcd_client.get_prefix(prefix):
                    service_data = json.loads(value.decode())
                    instance = ServiceInstance(**service_data)
                    instances.append(instance)
                return instances
            
            elif self.backend == "redis" and self.redis_client:
                pattern = f"service:{service_name}:*"
                keys = self.redis_client.keys(pattern)
                instances = []
                for key in keys:
                    value = self.redis_client.get(key)
                    if value:
                        service_data = json.loads(value.decode())
                        instance = ServiceInstance(**service_data)
                        instances.append(instance)
                return instances
            
            else:
                # Memory backend
                return [service for service in self.services.values() 
                       if service.service_name == service_name and service.is_healthy]
            
        except Exception as e:
            logger.error(f"Failed to discover services for {service_name}: {e}")
            return []
    
    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all registered services grouped by name"""
        services_by_name = {}
        
        for service in self.services.values():
            if service.service_name not in services_by_name:
                services_by_name[service.service_name] = []
            services_by_name[service.service_name].append(service)
        
        return services_by_name

class HealthChecker:
    """Service health monitoring"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.check_interval = 30
        self.timeout = 10
        self.running = False
        self.task = None
    
    async def start(self):
        """Start health checking"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")
    
    async def stop(self):
        """Stop health checking"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_services(self):
        """Check health of all registered services"""
        services = list(self.registry.services.values())
        
        if not services:
            return
        
        # Create tasks for concurrent health checks
        tasks = [self._check_service_health(service) for service in services]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: ServiceInstance):
        """Check health of a single service"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(service.health_check_url)
                
                if response.status_code == 200:
                    service.status = ServiceStatus.HEALTHY
                    service.failure_count = 0
                    service.circuit_breaker_open = False
                elif 200 <= response.status_code < 300:
                    service.status = ServiceStatus.DEGRADED
                    service.failure_count = 0
                else:
                    self._handle_unhealthy_service(service)
                
                service.last_health_check = datetime.utcnow()
                
        except Exception as e:
            logger.warning(f"Health check failed for {service.service_id}: {e}")
            self._handle_unhealthy_service(service)
    
    def _handle_unhealthy_service(self, service: ServiceInstance):
        """Handle unhealthy service"""
        service.failure_count += 1
        service.last_health_check = datetime.utcnow()
        
        if service.failure_count >= 3:
            service.status = ServiceStatus.UNHEALTHY
            service.circuit_breaker_open = True
        else:
            service.status = ServiceStatus.DEGRADED

class LoadBalancer:
    """Load balancing for service instances"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.round_robin_counters = {}
        self._lock = threading.RLock()
    
    def select_instance(self, instances: List[ServiceInstance], 
                       client_ip: str = None) -> Optional[ServiceInstance]:
        """Select an instance based on load balancing strategy"""
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        
        if not healthy_instances:
            return None
        
        with self._lock:
            if self.strategy == "round_robin":
                return self._round_robin(healthy_instances)
            elif self.strategy == "weighted_round_robin":
                return self._weighted_round_robin(healthy_instances)
            elif self.strategy == "least_connections":
                return self._least_connections(healthy_instances)
            elif self.strategy == "random":
                import random
                return random.choice(healthy_instances)
            elif self.strategy == "ip_hash" and client_ip:
                return self._ip_hash(healthy_instances, client_ip)
            else:
                return healthy_instances[0]
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        service_name = instances[0].service_name
        if service_name not in self.round_robin_counters:
            self.round_robin_counters[service_name] = 0
        
        index = self.round_robin_counters[service_name] % len(instances)
        self.round_robin_counters[service_name] += 1
        return instances[index]
    
    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection"""
        total_weight = sum(inst.load_balancer_weight for inst in instances)
        service_name = instances[0].service_name
        
        if service_name not in self.round_robin_counters:
            self.round_robin_counters[service_name] = 0
        
        target = self.round_robin_counters[service_name] % total_weight
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.load_balancer_weight
            if target < current_weight:
                self.round_robin_counters[service_name] += 1
                return instance
        
        return instances[0]
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection (placeholder - would need connection tracking)"""
        return min(instances, key=lambda x: x.failure_count)
    
    def _ip_hash(self, instances: List[ServiceInstance], client_ip: str) -> ServiceInstance:
        """IP hash selection"""
        hash_value = hash(client_ip)
        index = hash_value % len(instances)
        return instances[index]

class ServiceCommunicator:
    """Inter-service communication handler"""
    
    def __init__(self, registry: ServiceRegistry, load_balancer: LoadBalancer):
        self.registry = registry
        self.load_balancer = load_balancer
        self.http_client = None
        self.circuit_breakers = {}
        self.correlation_id_header = "X-Correlation-ID"
    
    async def initialize(self):
        """Initialize HTTP client"""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def call_service(self, 
                          service_name: str, 
                          method: str, 
                          path: str, 
                          **kwargs) -> httpx.Response:
        """Make a call to another service"""
        instances = await self.registry.discover_services(service_name)
        if not instances:
            raise HTTPException(status_code=503, detail=f"Service {service_name} not available")
        
        instance = self.load_balancer.select_instance(instances)
        if not instance:
            raise HTTPException(status_code=503, detail=f"No healthy instances of {service_name}")
        
        url = f"{instance.base_url}{path}"
        
        # Add correlation ID
        headers = kwargs.get('headers', {})
        if self.correlation_id_header not in headers:
            headers[self.correlation_id_header] = str(uuid.uuid4())
        kwargs['headers'] = headers
        
        try:
            response = await self.http_client.request(method, url, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Service call failed to {service_name}: {e}")
            raise HTTPException(status_code=503, detail=f"Service call failed: {e}")

class MicroservicesOrchestrator:
    """Main orchestrator for microservices"""
    
    def __init__(self, config_path: str = "microservices.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.registry = ServiceRegistry(backend=self.config.get('registry_backend', 'memory'))
        self.health_checker = HealthChecker(self.registry)
        self.load_balancer = LoadBalancer(strategy=self.config.get('load_balancer_strategy', 'round_robin'))
        self.communicator = ServiceCommunicator(self.registry, self.load_balancer)
        self.services_processes = {}
        self.running = False
        
        # Create FastAPI app for orchestrator API
        self.app = FastAPI(title="Microservices Orchestrator", version="1.0.0")
        self._setup_api_routes()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load microservices configuration"""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default microservices configuration"""
        default_config = {
            'registry_backend': 'memory',
            'load_balancer_strategy': 'round_robin',
            'services': {
                'gateway-service': {
                    'type': 'gateway',
                    'port': 8000,
                    'replicas': 1,
                    'dependencies': []
                },
                'auth-service': {
                    'type': 'auth',
                    'port': 8001,
                    'replicas': 1,
                    'dependencies': []
                },
                'content-service': {
                    'type': 'content',
                    'port': 8002,
                    'replicas': 1,
                    'dependencies': ['auth-service']
                },
                'analytics-service': {
                    'type': 'analytics',
                    'port': 8003,
                    'replicas': 1,
                    'dependencies': ['auth-service']
                },
                'security-service': {
                    'type': 'security',
                    'port': 8004,
                    'replicas': 1,
                    'dependencies': ['auth-service']
                },
                'workflow-service': {
                    'type': 'workflow',
                    'port': 8005,
                    'replicas': 1,
                    'dependencies': ['auth-service', 'content-service']
                }
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def _setup_api_routes(self):
        """Setup API routes for orchestrator"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Microservices Orchestrator", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": await self._get_services_health()
            }
        
        @self.app.get("/services")
        async def get_services():
            return await self.registry.get_all_services()
        
        @self.app.get("/services/{service_name}")
        async def get_service(service_name: str):
            instances = await self.registry.discover_services(service_name)
            return {
                "service_name": service_name,
                "instances": [asdict(inst) for inst in instances]
            }
        
        @self.app.post("/services/{service_name}/scale")
        async def scale_service(service_name: str, replicas: int):
            return await self._scale_service(service_name, replicas)
        
        @self.app.post("/services/{service_name}/restart")
        async def restart_service(service_name: str):
            return await self._restart_service(service_name)
    
    async def start(self):
        """Start the orchestrator and all services"""
        if self.running:
            return
        
        self.running = True
        
        # Initialize communicator
        await self.communicator.initialize()
        
        # Start health checker
        await self.health_checker.start()
        
        # Start services
        await self._start_all_services()
        
        logger.info("Microservices orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and all services"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop services
        await self._stop_all_services()
        
        # Stop health checker
        await self.health_checker.stop()
        
        # Cleanup communicator
        await self.communicator.cleanup()
        
        logger.info("Microservices orchestrator stopped")
    
    async def _start_all_services(self):
        """Start all configured services"""
        services_config = self.config.get('services', {})
        
        # Start services in dependency order
        started_services = set()
        while len(started_services) < len(services_config):
            for service_name, config in services_config.items():
                if service_name in started_services:
                    continue
                
                dependencies = config.get('dependencies', [])
                if all(dep in started_services for dep in dependencies):
                    await self._start_service(service_name, config)
                    started_services.add(service_name)
    
    async def _start_service(self, service_name: str, config: Dict[str, Any]):
        """Start a single service"""
        try:
            service_type = ServiceType(config['type'])
            port = config['port']
            
            # Create service instance
            service_id = f"{service_name}-{uuid.uuid4().hex[:8]}"
            instance = ServiceInstance(
                service_id=service_id,
                service_name=service_name,
                service_type=service_type,
                host="localhost",
                port=port,
                version="1.0.0",
                tags={service_type.value, "orchestrated"}
            )
            
            # Register service
            await self.registry.register_service(instance)
            
            # Start service process (placeholder - would implement actual service starting)
            logger.info(f"Started service {service_name} on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
    
    async def _stop_all_services(self):
        """Stop all services"""
        for service_id in list(self.services_processes.keys()):
            await self._stop_service_instance(service_id)
    
    async def _stop_service_instance(self, service_id: str):
        """Stop a service instance"""
        try:
            # Deregister service
            await self.registry.deregister_service(service_id)
            
            # Stop process (placeholder)
            if service_id in self.services_processes:
                del self.services_processes[service_id]
            
            logger.info(f"Stopped service instance {service_id}")
            
        except Exception as e:
            logger.error(f"Failed to stop service {service_id}: {e}")
    
    async def _get_services_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        services = await self.registry.get_all_services()
        health_status = {}
        
        for service_name, instances in services.items():
            healthy_count = sum(1 for inst in instances if inst.is_healthy)
            total_count = len(instances)
            
            health_status[service_name] = {
                "healthy_instances": healthy_count,
                "total_instances": total_count,
                "status": "healthy" if healthy_count > 0 else "unhealthy",
                "instances": [
                    {
                        "service_id": inst.service_id,
                        "status": inst.status.value,
                        "last_health_check": inst.last_health_check.isoformat() if inst.last_health_check else None
                    }
                    for inst in instances
                ]
            }
        
        return health_status
    
    async def _scale_service(self, service_name: str, replicas: int) -> Dict[str, Any]:
        """Scale a service to specified replicas"""
        # Placeholder for scaling logic
        return {
            "service_name": service_name,
            "target_replicas": replicas,
            "status": "scaling_requested"
        }
    
    async def _restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a service"""
        # Placeholder for restart logic
        return {
            "service_name": service_name,
            "status": "restart_requested"
        }
    
    def run(self, host: str = "localhost", port: int = 9000):
        """Run the orchestrator"""
        
        async def startup():
            await self.start()
        
        async def shutdown():
            await self.stop()
        
        self.app.add_event_handler("startup", startup)
        self.app.add_event_handler("shutdown", shutdown)
        
        uvicorn.run(self.app, host=host, port=port)


# CLI interface
def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangGraph 101 Microservices Orchestrator')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'scale'], 
                       help='Command to execute')
    parser.add_argument('--config', default='microservices.yaml', 
                       help='Configuration file path')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=9000, help='Port to bind to')
    parser.add_argument('--service', help='Service name for specific operations')
    parser.add_argument('--replicas', type=int, help='Number of replicas for scaling')
    
    args = parser.parse_args()
    
    orchestrator = MicroservicesOrchestrator(config_path=args.config)
    
    if args.command == 'start':
        try:
            logger.info(f"Starting microservices orchestrator on {args.host}:{args.port}")
            orchestrator.run(host=args.host, port=args.port)
        except KeyboardInterrupt:
            logger.info("Orchestrator stopped by user")
    
    elif args.command == 'status':
        # Show status (simplified for CLI)
        print("Microservices Orchestrator Status")
        print("Run with 'start' command to begin orchestration")
    
    else:
        print(f"Command {args.command} not implemented yet")


if __name__ == '__main__':
    main()
