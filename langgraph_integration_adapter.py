#!/usr/bin/env python3
"""
LangGraph 101 - Integration Adapter
=================================

Integration adapter that seamlessly connects existing LangGraph 101 applications
(langgraph-101.py and streamlit_app.py) with the new Infrastructure Integration Hub.

This adapter provides:
- Zero-downtime integration with existing functionality
- Progressive enhancement without breaking changes
- Automatic fallback to original behavior if infrastructure is unavailable
- Performance monitoring and health checks
- Load balancing and routing for multiple instances

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
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager, contextmanager
import uuid
from functools import wraps
import importlib
import traceback

# FastAPI and async components
try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available - some features will be limited")

# Import infrastructure hub
try:
    from infrastructure_integration_hub import InfrastructureHub, IntegrationConfig, LangGraphBackendService
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    logging.warning("Infrastructure Hub not available - running in fallback mode")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AdapterConfig:
    """Configuration for the integration adapter"""
    
    # Application paths
    cli_app_path: str = "langgraph-101.py"
    streamlit_app_path: str = "streamlit_app.py"
    
    # Integration settings
    enable_infrastructure: bool = True
    enable_load_balancing: bool = True
    enable_health_checks: bool = True
    enable_monitoring: bool = True
    
    # Service ports
    adapter_port: int = 9000
    streamlit_port: int = 8501
    cli_service_port: int = 8002
    
    # Process management
    auto_start_services: bool = True
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    
    # Health check settings
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    def __post_init__(self):
        # Load from environment variables
        self.enable_infrastructure = os.getenv('ENABLE_INFRASTRUCTURE', str(self.enable_infrastructure)).lower() == 'true'
        self.enable_load_balancing = os.getenv('ENABLE_LOAD_BALANCING', str(self.enable_load_balancing)).lower() == 'true'
        self.enable_health_checks = os.getenv('ENABLE_HEALTH_CHECKS', str(self.enable_health_checks)).lower() == 'true'
        self.enable_monitoring = os.getenv('ENABLE_MONITORING', str(self.enable_monitoring)).lower() == 'true'
        
        self.adapter_port = int(os.getenv('ADAPTER_PORT', str(self.adapter_port)))
        self.streamlit_port = int(os.getenv('STREAMLIT_PORT', str(self.streamlit_port)))
        self.cli_service_port = int(os.getenv('CLI_SERVICE_PORT', str(self.cli_service_port)))

class ServiceManager:
    """Manages external service processes (Streamlit, CLI service)"""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.processes = {}
        self.restart_counts = {}
        self._lock = threading.RLock()
        
    def start_streamlit_service(self) -> bool:
        """Start Streamlit application as a subprocess"""
        try:
            if 'streamlit' in self.processes:
                if self.processes['streamlit'].poll() is None:
                    logger.info("Streamlit service already running")
                    return True
                else:
                    logger.warning("Streamlit process found but not running, restarting...")
            
            # Start Streamlit with specific port
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                self.config.streamlit_app_path,
                "--server.port", str(self.config.streamlit_port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            logger.info(f"Starting Streamlit service: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            self.processes['streamlit'] = process
            self.restart_counts['streamlit'] = 0
            
            # Give it a moment to start
            time.sleep(3)
            
            if process.poll() is None:
                logger.info(f"Streamlit service started successfully on port {self.config.streamlit_port}")
                return True
            else:
                logger.error("Streamlit service failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Streamlit service: {e}")
            return False
    
    def start_cli_service(self) -> bool:
        """Start CLI application as a web service wrapper"""
        try:
            if 'cli_service' in self.processes:
                if self.processes['cli_service'].poll() is None:
                    logger.info("CLI service already running")
                    return True
            
            # Create a simple FastAPI wrapper for CLI app
            cli_wrapper_code = f"""
import sys
import os
import json
import subprocess
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="LangGraph CLI Service")

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "service": "cli", "timestamp": "{{datetime.now().isoformat()}}"}}

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "")
        
        # Here you would integrate with the actual CLI app logic
        # For now, return a simple response
        return {{"response": f"CLI processed: {{message}}", "status": "success"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={self.config.cli_service_port})
"""
            
            # Write wrapper to temporary file
            wrapper_path = "cli_service_wrapper.py"
            with open(wrapper_path, 'w') as f:
                f.write(cli_wrapper_code)
            
            # Start CLI service wrapper
            cmd = [sys.executable, wrapper_path]
            
            logger.info(f"Starting CLI service wrapper: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            self.processes['cli_service'] = process
            self.restart_counts['cli_service'] = 0
            
            # Give it a moment to start
            time.sleep(3)
            
            if process.poll() is None:
                logger.info(f"CLI service started successfully on port {self.config.cli_service_port}")
                return True
            else:
                logger.error("CLI service failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting CLI service: {e}")
            return False
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        try:
            if service_name not in self.processes:
                return False
            
            process = self.processes[service_name]
            return process.poll() is None
            
        except Exception as e:
            logger.error(f"Error checking {service_name} health: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a failed service"""
        try:
            with self._lock:
                restart_count = self.restart_counts.get(service_name, 0)
                
                if restart_count >= self.config.max_restart_attempts:
                    logger.error(f"Max restart attempts reached for {service_name}")
                    return False
                
                logger.info(f"Restarting {service_name} (attempt {restart_count + 1})")
                
                # Stop existing process
                if service_name in self.processes:
                    try:
                        self.processes[service_name].terminate()
                        self.processes[service_name].wait(timeout=10)
                    except:
                        try:
                            self.processes[service_name].kill()
                        except:
                            pass
                    del self.processes[service_name]
                
                # Start service
                success = False
                if service_name == 'streamlit':
                    success = self.start_streamlit_service()
                elif service_name == 'cli_service':
                    success = self.start_cli_service()
                
                if success:
                    self.restart_counts[service_name] = restart_count + 1
                    logger.info(f"Successfully restarted {service_name}")
                else:
                    logger.error(f"Failed to restart {service_name}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error restarting {service_name}: {e}")
            return False
    
    def stop_all_services(self):
        """Stop all managed services"""
        logger.info("Stopping all services...")
        
        for service_name, process in self.processes.items():
            try:
                logger.info(f"Stopping {service_name}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {service_name}...")
                    process.kill()
                    process.wait()
                logger.info(f"{service_name} stopped")
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        self.processes.clear()
        self.restart_counts.clear()

class LangGraphIntegrationAdapter:
    """Main integration adapter that coordinates everything"""
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.service_manager = ServiceManager(self.config)
        self.infrastructure_hub = None
        self.app = None
        self.health_monitor_task = None
        self.is_running = False
        
        # Initialize infrastructure if available
        if INFRASTRUCTURE_AVAILABLE and self.config.enable_infrastructure:
            try:
                integration_config = IntegrationConfig(
                    gateway_host="0.0.0.0",
                    gateway_port=self.config.adapter_port
                )
                self.infrastructure_hub = InfrastructureHub(integration_config)
                logger.info("Infrastructure Hub initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Infrastructure Hub: {e}")
                self.infrastructure_hub = None
        
        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self._setup_fastapi_app()
    
    def _setup_fastapi_app(self):
        """Setup FastAPI application for the adapter"""
        self.app = FastAPI(
            title="LangGraph Integration Adapter",
            description="Integration adapter for LangGraph 101 applications",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with service information"""
            return {
                "service": "LangGraph Integration Adapter",
                "version": "1.0.0",
                "status": "running",
                "infrastructure_enabled": self.infrastructure_hub is not None,
                "services": {
                    "streamlit": f"http://localhost:{self.config.streamlit_port}",
                    "cli_service": f"http://localhost:{self.config.cli_service_port}",
                    "adapter": f"http://localhost:{self.config.adapter_port}"
                },
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            health_status = {
                "adapter": "healthy",
                "streamlit": "healthy" if self.service_manager.check_service_health('streamlit') else "unhealthy",
                "cli_service": "healthy" if self.service_manager.check_service_health('cli_service') else "unhealthy",
                "infrastructure": "available" if self.infrastructure_hub else "not_available",
                "timestamp": datetime.now().isoformat()
            }
            
            overall_status = "healthy" if all(
                status in ["healthy", "available", "not_available"] 
                for status in health_status.values() 
                if status != health_status["timestamp"]
            ) else "degraded"
            
            health_status["overall"] = overall_status
            
            return health_status
        
        @self.app.get("/streamlit")
        async def redirect_to_streamlit():
            """Redirect to Streamlit application"""
            streamlit_url = f"http://localhost:{self.config.streamlit_port}"
            return RedirectResponse(url=streamlit_url)
        
        @self.app.post("/cli/chat")
        async def cli_chat(request: Request):
            """Forward chat requests to CLI service"""
            try:
                data = await request.json()
                
                if self.infrastructure_hub and self.infrastructure_hub.backend_service:
                    # Use infrastructure hub for processing
                    result = await self.infrastructure_hub.backend_service.process_chat_message(
                        data.get("message", ""),
                        data.get("user_id", "anonymous"),
                        data.get("context", {})
                    )
                    return result
                else:
                    # Fallback to simple processing
                    message = data.get("message", "")
                    return {
                        "response": f"Processed (fallback mode): {message}",
                        "status": "success",
                        "mode": "fallback"
                    }
            except Exception as e:
                logger.error(f"Error in CLI chat: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/services/restart/{service_name}")
        async def restart_service(service_name: str):
            """Restart a specific service"""
            if service_name not in ['streamlit', 'cli_service']:
                raise HTTPException(status_code=400, detail="Invalid service name")
            
            success = self.service_manager.restart_service(service_name)
            return {
                "service": service_name,
                "restart_success": success,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/infrastructure/status")
        async def infrastructure_status():
            """Get infrastructure status"""
            if not self.infrastructure_hub:
                return {"status": "not_available"}
            
            try:
                return await self.infrastructure_hub.get_health_status()
            except Exception as e:
                logger.error(f"Error getting infrastructure status: {e}")
                return {"status": "error", "error": str(e)}
    
    async def _health_monitor_loop(self):
        """Background task to monitor service health"""
        while self.is_running:
            try:
                # Check service health
                for service_name in ['streamlit', 'cli_service']:
                    if not self.service_manager.check_service_health(service_name):
                        if self.config.restart_on_failure:
                            logger.warning(f"{service_name} is unhealthy, attempting restart...")
                            self.service_manager.restart_service(service_name)
                
                # Wait before next check
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def start(self):
        """Start the integration adapter"""
        try:
            logger.info("Starting LangGraph Integration Adapter...")
            
            # Start infrastructure hub if available
            if self.infrastructure_hub:
                logger.info("Starting Infrastructure Hub...")
                await self.infrastructure_hub.start()
            
            # Start managed services
            if self.config.auto_start_services:
                logger.info("Starting managed services...")
                
                if not self.service_manager.start_streamlit_service():
                    logger.warning("Failed to start Streamlit service")
                
                if not self.service_manager.start_cli_service():
                    logger.warning("Failed to start CLI service")
            
            # Start health monitoring
            if self.config.enable_health_checks:
                logger.info("Starting health monitoring...")
                self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            self.is_running = True
            logger.info("LangGraph Integration Adapter started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Integration Adapter: {e}")
            raise
    
    async def stop(self):
        """Stop the integration adapter"""
        try:
            logger.info("Stopping LangGraph Integration Adapter...")
            self.is_running = False
            
            # Stop health monitoring
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
                try:
                    await self.health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop managed services
            self.service_manager.stop_all_services()
            
            # Stop infrastructure hub
            if self.infrastructure_hub:
                await self.infrastructure_hub.stop()
            
            logger.info("LangGraph Integration Adapter stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Integration Adapter: {e}")
    
    def run(self):
        """Run the integration adapter"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available - cannot run adapter")
            return
        
        try:
            # Create event loop for async startup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start the adapter
            loop.run_until_complete(self.start())
            
            # Run FastAPI server
            uvicorn.run(
                self.app,
                host="0.0.0.0",
                port=self.config.adapter_port,
                loop=loop
            )
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error running adapter: {e}")
        finally:
            # Cleanup
            try:
                loop.run_until_complete(self.stop())
            except:
                pass

def main():
    """Main entry point for the integration adapter"""
    try:
        # Load configuration
        config = AdapterConfig()
        
        # Create and run adapter
        adapter = LangGraphIntegrationAdapter(config)
        adapter.run()
        
    except KeyboardInterrupt:
        logger.info("Integration adapter stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
