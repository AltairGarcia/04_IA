#!/usr/bin/env python3
"""
Integrated Startup Script for LangGraph 101 Platform - Phase 4
============================================================

This script initializes and starts all infrastructure components and services
for the enhanced LangGraph 101 platform with Phase 4 streaming capabilities,
FastAPI bridge, and enhanced Streamlit frontend.

Phase 4 Features:
- Streaming LangGraph integration with WebSocket support
- Production FastAPI bridge with real-time capabilities
- Enhanced Streamlit frontend with live chat
- Multi-agent orchestration and workflow management
- Comprehensive security and monitoring
"""

import asyncio
import json
import logging
import os
import sys
import signal
import subprocess
import threading
import time
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import infrastructure components
from database_connection_pool import DatabaseConnectionPool
from config_hot_reload import ConfigHotReload
from enhanced_rate_limiting import EnhancedRateLimiter
from message_queue_system import MessageQueueSystem
from cache_manager import CacheManager
from api_gateway_integration import (
    IntegrationConfig, 
    LangGraphBackendService, 
    IntegratedAPIGateway,
    integration_logger
)

# Import Phase 4 components
try:
    from fastapi_streaming_bridge import create_production_app as create_streaming_bridge
    PHASE4_STREAMING_AVAILABLE = True
except ImportError:
    PHASE4_STREAMING_AVAILABLE = False
    logging.warning("Phase 4 streaming bridge not available")

# Import existing components
from config import load_config
from core.database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class IntegratedPlatformManager:
    """Manages the complete integrated platform lifecycle with Phase 4 support."""
    
    def __init__(self):
        self.config = None
        self.integration_config = None
        self.db_pool = None
        self.config_hot_reload = None
        self.rate_limiter = None
        self.message_queue = None
        self.cache_manager = None
        self.backend_service = None
        self.api_gateway = None
        self.streaming_bridge = None  # Phase 4: FastAPI streaming bridge
        self.streamlit_process = None  # Phase 4: Streamlit process
        self.services_running = False
        self.shutdown_event = threading.Event()
        self.phase4_enabled = PHASE4_STREAMING_AVAILABLE
        
    async def initialize_infrastructure(self) -> bool:
        """Initialize all infrastructure components."""
        try:
            logger.info("🚀 Initializing LangGraph 101 Integrated Platform...")
            
            # Load base configuration
            logger.info("📋 Loading base configuration...")
            self.config = load_config()
            
            # Load integration configuration
            logger.info("🔧 Loading integration configuration...")
            self.integration_config = IntegrationConfig()
            
            # Initialize database connection pool
            logger.info("🗄️ Initializing database connection pool...")
            self.db_pool = DatabaseConnectionPool(
                config={
                    'default': {
                        'type': 'sqlite',
                        'database': self.config.get('database', {}).get('path', 'langgraph_101.db'),
                        'pool_size': 10,
                        'max_overflow': 20,
                        'timeout': 30
                    }
                }
            )
            await self.db_pool.initialize()
            
            # Initialize configuration hot reload
            logger.info("🔄 Setting up configuration hot reload...")
            self.config_hot_reload = ConfigHotReload()
            self.config_hot_reload.register_callback(self._on_config_change)
            self.config_hot_reload.start_monitoring()
            
            # Initialize cache manager
            logger.info("🗃️ Setting up cache manager...")
            self.cache_manager = CacheManager(
                redis_url=self.integration_config.REDIS_URL,
                default_ttl=3600
            )
            await self.cache_manager.initialize()
            
            # Initialize rate limiter
            logger.info("⚡ Setting up enhanced rate limiter...")
            self.rate_limiter = EnhancedRateLimiter(
                redis_url=self.integration_config.REDIS_URL
            )
            await self.rate_limiter.initialize()
            
            # Initialize message queue system
            logger.info("📬 Setting up message queue system...")
            self.message_queue = MessageQueueSystem(
                broker_url=self.integration_config.CELERY_BROKER_URL,
                result_backend=self.integration_config.CELERY_RESULT_BACKEND
            )
            await self.message_queue.initialize()
            
            # Initialize backend service
            logger.info("🔧 Setting up LangGraph backend service...")
            self.backend_service = LangGraphBackendService(
                db_pool=self.db_pool,
                cache_manager=self.cache_manager,
                message_queue=self.message_queue
            )
              # Initialize API gateway
            logger.info("🌐 Setting up integrated API gateway...")
            self.api_gateway = IntegratedAPIGateway(
                rate_limiter=self.rate_limiter,
                cache_manager=self.cache_manager,
                message_queue=self.message_queue,
                config=self.integration_config
            )
            
            # Register backend service with gateway
            await self.api_gateway.register_service(
                "langgraph-backend",
                f"http://localhost:{self.integration_config.BACKEND_PORT}",
                health_check_path="/health"
            )
            
            # Phase 4: Initialize streaming bridge if available
            if self.phase4_enabled:
                logger.info("🌊 Setting up Phase 4 streaming bridge...")
                self.streaming_bridge = create_streaming_bridge()
                logger.info("✅ Phase 4 streaming bridge initialized")
            
            logger.info("✅ All infrastructure components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize infrastructure: {e}")
            return False
      async def start_services(self) -> bool:
        """Start all services including Phase 4 components."""
        try:
            logger.info("🚀 Starting all services...")
            
            # Start message queue workers
            logger.info("📬 Starting message queue workers...")
            await self.message_queue.start_workers()
            
            # Start backend service
            logger.info("🔧 Starting backend service...")
            backend_config = uvicorn.Config(
                app=self.backend_service.app,
                host=self.integration_config.BACKEND_HOST,
                port=self.integration_config.BACKEND_PORT,
                log_level="info"
            )
            self.backend_server = uvicorn.Server(backend_config)
            
            # Start API gateway
            logger.info("🌐 Starting API gateway...")
            gateway_config = uvicorn.Config(
                app=self.api_gateway.app,
                host=self.integration_config.GATEWAY_HOST,
                port=self.integration_config.GATEWAY_PORT,
                log_level="info"
            )
            self.gateway_server = uvicorn.Server(gateway_config)
            
            # Phase 4: Start streaming bridge
            streaming_server = None
            if self.phase4_enabled and self.streaming_bridge:
                logger.info("🌊 Starting Phase 4 streaming bridge...")
                streaming_config = uvicorn.Config(
                    app=self.streaming_bridge,
                    host="localhost",
                    port=8001,  # Different port for streaming bridge
                    log_level="info"
                )
                streaming_server = uvicorn.Server(streaming_config)
            
            # Start servers in background tasks
            backend_task = asyncio.create_task(self.backend_server.serve())
            gateway_task = asyncio.create_task(self.gateway_server.serve())
            
            # Start streaming server if available
            if streaming_server:
                streaming_task = asyncio.create_task(streaming_server.serve())
                self.streaming_server = streaming_server
            
            # Wait a moment for servers to start
            await asyncio.sleep(3)
            
            # Phase 4: Start Streamlit frontend
            if self.phase4_enabled:
                await self._start_streamlit_phase4()
            
            # Verify services are healthy
            if await self._verify_service_health():
                self.services_running = True
                logger.info("✅ All services started successfully!")
                
                # Print service information
                self._print_service_info()
                
                return True
            else:
                logger.error("❌ Service health check failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start services: {e}")
            return False
      async def _start_streamlit_phase4(self):
        """Start the Phase 4 enhanced Streamlit frontend."""
        try:
            logger.info("🎨 Starting Phase 4 enhanced Streamlit frontend...")
            
            # Set environment variables for integration
            os.environ['LANGGRAPH_PHASE4_MODE'] = 'true'
            os.environ['LANGGRAPH_STREAMING_API_URL'] = 'http://localhost:8001'
            os.environ['LANGGRAPH_API_BASE_URL'] = f"http://localhost:{self.integration_config.GATEWAY_PORT}"
            
            # Start Streamlit in background
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "streamlit_app_phase4.py",
                "--server.port", "8502",  # Different port for Phase 4
                "--server.address", "localhost",
                "--server.headless", "true"
            ]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait a moment for Streamlit to start
            await asyncio.sleep(3)
            
            if self.streamlit_process.poll() is None:
                logger.info("✅ Phase 4 Streamlit frontend started on http://localhost:8502")
            else:
                logger.error("❌ Failed to start Streamlit frontend")
                
        except Exception as e:
            logger.error(f"❌ Error starting Streamlit: {e}")

    async def _verify_service_health(self) -> bool:
        """Verify all services are healthy including Phase 4 components."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Check backend service
                try:
                    async with session.get(
                        f"http://{self.integration_config.BACKEND_HOST}:{self.integration_config.BACKEND_PORT}/health",
                        timeout=5
                    ) as response:
                        if response.status != 200:
                            logger.error("Backend service health check failed")
                            return False
                except Exception as e:
                    logger.error(f"Backend service not responding: {e}")
                    return False
                
                # Check API gateway
                try:
                    async with session.get(
                        f"http://{self.integration_config.GATEWAY_HOST}:{self.integration_config.GATEWAY_PORT}/health",
                        timeout=5
                    ) as response:
                        if response.status != 200:
                            logger.error("API gateway health check failed")
                            return False
                except Exception as e:
                    logger.error(f"API gateway not responding: {e}")
                    return False
                
                # Phase 4: Check streaming bridge
                if self.phase4_enabled:
                    try:
                        async with session.get(
                            "http://localhost:8001/health",
                            timeout=5
                        ) as response:
                            if response.status != 200:
                                logger.warning("Streaming bridge health check failed")
                            else:
                                logger.info("✅ Streaming bridge health check passed")
                    except Exception as e:
                        logger.warning(f"Streaming bridge not responding: {e}")
            
            return True
            
        except ImportError:
            logger.warning("aiohttp not available, skipping HTTP health checks")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
      def _print_service_info(self):
        """Print information about running services including Phase 4."""
        logger.info("\n" + "="*70)
        logger.info("🎉 LangGraph 101 Integrated Platform - Phase 4 RUNNING!")
        logger.info("="*70)
        logger.info(f"🌐 API Gateway: http://{self.integration_config.GATEWAY_HOST}:{self.integration_config.GATEWAY_PORT}")
        logger.info(f"🔧 Backend Service: http://{self.integration_config.BACKEND_HOST}:{self.integration_config.BACKEND_PORT}")
        
        # Phase 4 services
        if self.phase4_enabled:
            logger.info("🌊 Phase 4 Streaming Bridge: http://localhost:8001")
            logger.info("🎨 Phase 4 Streamlit Frontend: http://localhost:8502")
        
        logger.info(f"📬 Message Queue: {self.integration_config.CELERY_BROKER_URL}")
        logger.info(f"🗃️ Cache: {self.integration_config.REDIS_URL}")
        
        logger.info("\n📋 Available Endpoints:")
        logger.info("  • /api/v1/chat - Chat with LangGraph agent")
        logger.info("  • /api/v1/content - Content creation")
        logger.info("  • /api/v1/personas - Persona management")
        logger.info("  • /api/v1/history - Conversation history")
        logger.info("  • /api/v1/export - Export conversations")
        logger.info("  • /health - Health status")
        logger.info("  • /metrics - Performance metrics")
        
        if self.phase4_enabled:
            logger.info("\n🌊 Phase 4 Streaming Endpoints:")
            logger.info("  • /stream/chat - Real-time streaming chat")
            logger.info("  • /ws/chat - WebSocket chat connection")
            logger.info("  • /stream/agents - Multi-agent orchestration")
            logger.info("  • /stream/workflows - Workflow management")
            logger.info("  • /admin/system - System administration")
        
        logger.info("\n🔐 Security Features Active:")
        logger.info("  • Enhanced rate limiting")
        logger.info("  • Request validation")
        logger.info("  • Authentication middleware")
        logger.info("  • CORS protection")
        if self.phase4_enabled:
            logger.info("  • WebSocket security")
            logger.info("  • Real-time monitoring")
        logger.info("="*70)
    
    def _on_config_change(self, config_path: str, config_data: Dict[str, Any]):
        """Handle configuration changes."""
        logger.info(f"🔄 Configuration changed: {config_path}")
        # Implement configuration reload logic here
        # This could update service configurations dynamically
      async def shutdown(self):
        """Gracefully shutdown all services including Phase 4 components."""
        logger.info("🛑 Shutting down integrated platform...")
        
        self.services_running = False
        self.shutdown_event.set()
        
        try:
            # Stop Streamlit process
            if self.streamlit_process and self.streamlit_process.poll() is None:
                logger.info("🛑 Stopping Streamlit frontend...")
                self.streamlit_process.terminate()
                try:
                    self.streamlit_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.streamlit_process.kill()
            
            # Stop servers
            if hasattr(self, 'streaming_server'):
                logger.info("🛑 Stopping streaming bridge...")
                self.streaming_server.should_exit = True
            if hasattr(self, 'backend_server'):
                logger.info("🛑 Stopping backend service...")
                self.backend_server.should_exit = True
            if hasattr(self, 'gateway_server'):
                logger.info("🛑 Stopping API gateway...")
                self.gateway_server.should_exit = True
            
            # Stop message queue
            if self.message_queue:
                await self.message_queue.stop_workers()
                await self.message_queue.shutdown()
            
            # Stop configuration monitoring
            if self.config_hot_reload:
                self.config_hot_reload.stop_monitoring()
            
            # Close database pool
            if self.db_pool:
                await self.db_pool.close_all()
            
            # Close cache manager
            if self.cache_manager:
                await self.cache_manager.close()
            
            logger.info("✅ Platform shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def run_forever(self):
        """Run the platform until interrupted."""
        try:
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Run the main loop
            asyncio.run(self._main_loop())
            
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            logger.info("👋 Platform stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}")
        self.shutdown_event.set()
    
    async def _main_loop(self):
        """Main application loop."""
        # Initialize and start services
        if not await self.initialize_infrastructure():
            sys.exit(1)
        
        if not await self.start_services():
            sys.exit(1)
        
        # Wait for shutdown signal
        while not self.shutdown_event.is_set():
            await asyncio.sleep(1)
        
        # Shutdown gracefully
        await self.shutdown()


class Phase4StreamlitIntegration:
    """Helper class for Phase 4 Streamlit integration."""
    
    @staticmethod
    def start_streamlit_app():
        """Start the Phase 4 Streamlit application in integrated mode."""
        import subprocess
        import sys
        
        # Set environment variables for Phase 4 integration
        os.environ['LANGGRAPH_PHASE4_MODE'] = 'true'
        os.environ['LANGGRAPH_STREAMING_API_URL'] = 'http://localhost:8001'
        os.environ['LANGGRAPH_API_BASE_URL'] = 'http://localhost:8000'
        
        # Start Phase 4 Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app_phase4.py",
            "--server.port", "8502",
            "--server.address", "localhost"
        ]
        
        logger.info("🎨 Starting Phase 4 Streamlit web interface...")
        subprocess.run(cmd)


def start_streaming_bridge_only():
    """Start only the streaming bridge for development."""
    try:
        if not PHASE4_STREAMING_AVAILABLE:
            logger.error("❌ Phase 4 streaming bridge not available")
            sys.exit(1)
        
        logger.info("🌊 Starting streaming bridge only...")
        app = create_streaming_bridge()
        
        uvicorn.run(
            app,
            host="localhost",
            port=8001,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start streaming bridge: {e}")
        sys.exit(1)


def run_security_validation() -> bool:
    """Run comprehensive security validation tests."""
    try:
        logger.info("🔒 Starting security validation...")
        
        # Import security test components
        from phase2_security_test_suite import SecurityTestSuite
        
        # Create test suite
        test_suite = SecurityTestSuite()
        
        # Run all security tests
        results = test_suite.run_all_tests()
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"security_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📋 Security validation report saved to {report_file}")
        
        # Check if all tests passed
        total_tests = results.get('total_tests', 0)
        passed_tests = results.get('passed_tests', 0)
        
        success = passed_tests == total_tests and total_tests > 0
        
        if success:
            logger.info(f"✅ All {total_tests} security tests passed")
        else:
            logger.error(f"❌ {total_tests - passed_tests} out of {total_tests} security tests failed")
        
        return success
        
    except ImportError:
        logger.error("❌ Security test suite not available")
        return False
    except Exception as e:
        logger.error(f"❌ Security validation failed: {e}")
        return False


class StreamlitIntegration:
    """Helper class for integrating with Streamlit."""
    
    @staticmethod
    def start_streamlit_app():
        """Start the Streamlit application in integrated mode."""
        import subprocess
        import sys
        
        # Set environment variable to indicate integrated mode
        os.environ['LANGGRAPH_INTEGRATED_MODE'] = 'true'
        os.environ['LANGGRAPH_API_BASE_URL'] = f"http://localhost:{IntegrationConfig().GATEWAY_PORT}"
        
        # Start Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app_integrated.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        logger.info("🌐 Starting Streamlit web interface...")
        subprocess.run(cmd)


def main():
    """Main entry point with Phase 4 support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph 101 Integrated Platform - Phase 4")
    parser.add_argument(
        '--mode', 
        choices=['full', 'phase4', 'backend-only', 'gateway-only', 'streaming-only', 'streamlit-phase4'], 
        default='phase4',
        help='Startup mode (default: phase4 for full Phase 4 stack)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--security-test',
        action='store_true',
        help='Run security validation tests'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run security tests if requested
    if args.security_test:
        logger.info("🔒 Running security validation tests...")
        result = run_security_validation()
        if result:
            logger.info("✅ Security tests passed")
        else:
            logger.error("❌ Security tests failed")
            sys.exit(1)
        return
    
    platform = IntegratedPlatformManager()
    
    if args.mode in ['full', 'phase4']:
        # Start the complete Phase 4 platform
        platform.run_forever()
    elif args.mode == 'streamlit-phase4':
        # Start just the Phase 4 Streamlit interface (assumes backend is running)
        Phase4StreamlitIntegration.start_streamlit_app()
    elif args.mode == 'streaming-only':
        # Start just the streaming bridge
        start_streaming_bridge_only()
    else:
        logger.error(f"Mode '{args.mode}' not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
