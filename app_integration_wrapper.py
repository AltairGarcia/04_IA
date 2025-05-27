#!/usr/bin/env python3
"""
LangGraph 101 - Application Integration Wrapper
==============================================

This wrapper provides seamless integration between existing LangGraph 101 applications
and the new infrastructure components. It acts as a bridge that:

- Preserves existing application functionality
- Gradually enables new infrastructure features
- Provides fallback mechanisms for reliability
- Monitors performance and health
- Enables progressive enhancement

Features:
- Zero-downtime integration
- Automatic fallback to original behavior
- Performance monitoring and optimization
- Health checks and diagnostics
- Configuration-driven feature activation

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
from dataclasses import dataclass
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationState:
    """Manages the state of infrastructure integration"""
    
    def __init__(self):
        self.infrastructure_available = False
        self.components_loaded = {}
        self.fallback_mode = False
        self.performance_metrics = {}
        self.last_health_check = None
        
        # Try to load infrastructure components
        self._check_infrastructure_availability()
    
    def _check_infrastructure_availability(self):
        """Check if infrastructure components are available"""
        try:
            # Try to import infrastructure hub
            from infrastructure_integration_hub import InfrastructureHub
            self.components_loaded['infrastructure_hub'] = True
            logger.info("Infrastructure Hub available")
        except ImportError:
            self.components_loaded['infrastructure_hub'] = False
            logger.warning("Infrastructure Hub not available")
        
        try:
            # Try to import integration adapter
            from langgraph_integration_adapter import LangGraphIntegrationAdapter
            self.components_loaded['integration_adapter'] = True
            logger.info("Integration Adapter available")
        except ImportError:
            self.components_loaded['integration_adapter'] = False
            logger.warning("Integration Adapter not available")
        
        try:
            # Try to import integrated config
            from integrated_config import load_config
            self.components_loaded['integrated_config'] = True
            logger.info("Integrated Config available")
        except ImportError:
            self.components_loaded['integrated_config'] = False
            logger.warning("Integrated Config not available")
        
        # Determine overall infrastructure availability
        self.infrastructure_available = any(self.components_loaded.values())
        
        if self.infrastructure_available:
            logger.info("Infrastructure components available - enhanced mode enabled")
        else:
            logger.warning("Infrastructure components not available - fallback mode enabled")
            self.fallback_mode = True

# Global integration state
integration_state = IntegrationState()

def with_infrastructure_fallback(fallback_func: Optional[Callable] = None):
    """Decorator that provides automatic fallback to original functionality"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Try enhanced functionality if infrastructure is available
                if integration_state.infrastructure_available and not integration_state.fallback_mode:
                    return func(*args, **kwargs)
                else:
                    # Use fallback functionality
                    if fallback_func:
                        logger.debug(f"Using fallback for {func.__name__}")
                        return fallback_func(*args, **kwargs)
                    else:
                        # Original function behavior
                        return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in enhanced function {func.__name__}: {e}")
                
                # Automatic fallback on error
                if fallback_func:
                    logger.info(f"Falling back to original functionality for {func.__name__}")
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed for {func.__name__}: {fallback_error}")
                        raise
                else:
                    raise
        return wrapper
    return decorator

class EnhancedLangGraphApp:
    """Enhanced wrapper for the main LangGraph application"""
    
    def __init__(self):
        self.original_app = None
        self.infrastructure_hub = None
        self.config = None
        self.performance_monitor = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize infrastructure components if available"""
        try:
            if integration_state.components_loaded.get('integrated_config'):
                from integrated_config import load_config
                self.config = load_config()
                logger.info("Integrated configuration loaded")
            
            if integration_state.components_loaded.get('infrastructure_hub'):
                from infrastructure_integration_hub import InfrastructureHub, IntegrationConfig
                
                if self.config:
                    # Use integrated config
                    integration_config = IntegrationConfig(
                        enable_api_gateway=self.config.components.enable_api_gateway,
                        enable_message_queue=self.config.components.enable_message_queue,
                        enable_connection_pool=self.config.components.enable_database_pool,
                        enable_cache_manager=self.config.components.enable_cache_manager,
                        enable_rate_limiting=self.config.components.enable_rate_limiting,
                        gateway_port=self.config.services.gateway_port,
                        backend_port=self.config.services.cli_service_port
                    )
                else:
                    # Use default config
                    integration_config = IntegrationConfig()
                
                self.infrastructure_hub = InfrastructureHub(integration_config)
                logger.info("Infrastructure Hub initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            integration_state.fallback_mode = True
    
    @with_infrastructure_fallback()
    def process_message(self, message: str, user_id: str = "anonymous", context: Dict = None) -> Dict[str, Any]:
        """Enhanced message processing with infrastructure support"""
        start_time = time.time()
        context = context or {}
        
        try:
            # Use infrastructure hub if available
            if self.infrastructure_hub and self.infrastructure_hub.backend_service:
                result = asyncio.run(
                    self.infrastructure_hub.backend_service.process_chat_message(
                        message, user_id, context
                    )
                )
            else:
                # Fallback to original processing
                result = self._process_message_fallback(message, user_id, context)
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self._record_performance_metric('message_processing', processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Fallback to original processing
            return self._process_message_fallback(message, user_id, context)
    
    def _process_message_fallback(self, message: str, user_id: str, context: Dict) -> Dict[str, Any]:
        """Fallback message processing using original functionality"""
        try:
            # Import original components
            from agent import create_agent, invoke_agent
            from config import load_config
            
            # Load original config
            original_config = load_config()
            
            # Create agent
            agent = create_agent(original_config)
            
            # Process message
            response = invoke_agent(agent, message)
            
            return {
                'response': response,
                'status': 'success',
                'mode': 'fallback',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            return {
                'response': f"Error processing message: {str(e)}",
                'status': 'error',
                'mode': 'fallback',
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _record_performance_metric(self, metric_name: str, value: float):
        """Record performance metric"""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {}
        
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # Keep only last 100 metrics per type
        if len(self.performance_metrics[metric_name]) > 100:
            self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for metric_name, metrics in getattr(self, 'performance_metrics', {}).items():
            if metrics:
                values = [m['value'] for m in metrics]
                summary[metric_name] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'last': values[-1] if values else 0
                }
        
        return summary
    
    async def start_infrastructure(self):
        """Start infrastructure components"""
        if self.infrastructure_hub:
            try:
                await self.infrastructure_hub.start()
                logger.info("Infrastructure started successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to start infrastructure: {e}")
                integration_state.fallback_mode = True
                return False
        return False
    
    async def stop_infrastructure(self):
        """Stop infrastructure components"""
        if self.infrastructure_hub:
            try:
                await self.infrastructure_hub.stop()
                logger.info("Infrastructure stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping infrastructure: {e}")

class StreamlitAppWrapper:
    """Enhanced wrapper for Streamlit application"""
    
    def __init__(self):
        self.enhanced_app = EnhancedLangGraphApp()
        self.session_state_initialized = False
    
    def initialize_session_state(self, st):
        """Initialize Streamlit session state with enhanced features"""
        if not self.session_state_initialized:
            # Initialize enhanced features in session state
            if 'enhanced_app' not in st.session_state:
                st.session_state.enhanced_app = self.enhanced_app
            
            if 'infrastructure_enabled' not in st.session_state:
                st.session_state.infrastructure_enabled = integration_state.infrastructure_available
            
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            
            self.session_state_initialized = True
    
    def enhance_chat_interface(self, st, original_chat_func: Callable):
        """Enhance the chat interface with infrastructure features"""
        
        @with_infrastructure_fallback(original_chat_func)
        def enhanced_chat(*args, **kwargs):
            # Use enhanced message processing
            if hasattr(st.session_state, 'enhanced_app'):
                return st.session_state.enhanced_app.process_message(*args, **kwargs)
            else:
                return original_chat(*args, **kwargs)
        
        return enhanced_chat
    
    def add_infrastructure_status_sidebar(self, st):
        """Add infrastructure status to sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("🏗️ Infrastructure Status")
            
            if integration_state.infrastructure_available:
                st.success("✅ Enhanced Mode")
                
                # Show component status
                for component, loaded in integration_state.components_loaded.items():
                    icon = "✅" if loaded else "❌"
                    st.write(f"{icon} {component.replace('_', ' ').title()}")
            else:
                st.warning("⚠️ Fallback Mode")
                st.write("Infrastructure components not available")
            
            # Performance metrics
            if hasattr(st.session_state, 'enhanced_app'):
                perf_summary = st.session_state.enhanced_app.get_performance_summary()
                if perf_summary:
                    st.markdown("**Performance:**")
                    for metric, stats in perf_summary.items():
                        st.write(f"• {metric}: {stats['avg']:.3f}s avg")

class CLIAppWrapper:
    """Enhanced wrapper for CLI application"""
    
    def __init__(self):
        self.enhanced_app = EnhancedLangGraphApp()
        self.original_main = None
    
    def wrap_main_function(self, original_main: Callable):
        """Wrap the original main function with enhancements"""
        
        @with_infrastructure_fallback(original_main)
        def enhanced_main(*args, **kwargs):
            try:
                # Start infrastructure if available
                if self.enhanced_app.infrastructure_hub:
                    print("🚀 Starting enhanced LangGraph with infrastructure...")
                    asyncio.run(self.enhanced_app.start_infrastructure())
                
                # Run original main with enhancements
                return self._run_enhanced_cli(*args, **kwargs)
                
            except KeyboardInterrupt:
                print("\\n⏹️  Shutting down...")
                if self.enhanced_app.infrastructure_hub:
                    asyncio.run(self.enhanced_app.stop_infrastructure())
            except Exception as e:
                logger.error(f"Error in enhanced CLI: {e}")
                # Fallback to original
                return original_main(*args, **kwargs)
        
        return enhanced_main
    
    def _run_enhanced_cli(self, *args, **kwargs):
        """Run enhanced CLI with infrastructure features"""
        # Import original modules
        from ui import print_welcome, print_colored, Colors, get_user_input
        
        # Enhanced welcome message
        print_colored("🚀 LangGraph 101 - Enhanced Edition", Colors.CYAN, bold=True)
        if integration_state.infrastructure_available:
            print_colored("✅ Infrastructure Mode: Enhanced features enabled", Colors.GREEN)
        else:
            print_colored("⚠️  Fallback Mode: Basic features only", Colors.YELLOW)
        
        print_welcome()
        
        # Main interaction loop with enhancements
        while True:
            try:
                user_input = get_user_input()
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break
                elif user_input.lower() == 'status':
                    self._show_infrastructure_status()
                    continue
                elif user_input.lower() == 'performance':
                    self._show_performance_metrics()
                    continue
                
                # Process message with enhanced features
                result = self.enhanced_app.process_message(user_input)
                
                # Display response
                if result.get('status') == 'success':
                    print_colored(result.get('response', ''), Colors.WHITE)
                    
                    # Show mode indicator
                    mode = result.get('mode', 'enhanced')
                    if mode == 'fallback':
                        print_colored("(fallback mode)", Colors.YELLOW)
                else:
                    print_colored(f"Error: {result.get('response', 'Unknown error')}", Colors.RED)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print_colored(f"Error: {e}", Colors.RED)
        
        print_colored("👋 Goodbye!", Colors.CYAN)
    
    def _show_infrastructure_status(self):
        """Show infrastructure status in CLI"""
        from ui import print_colored, Colors
        
        print_colored("\\n🏗️ Infrastructure Status:", Colors.YELLOW, bold=True)
        print_colored(f"Available: {'Yes' if integration_state.infrastructure_available else 'No'}", Colors.GREEN if integration_state.infrastructure_available else Colors.RED)
        print_colored(f"Fallback Mode: {'Yes' if integration_state.fallback_mode else 'No'}", Colors.YELLOW if integration_state.fallback_mode else Colors.GREEN)
        
        print_colored("\\nComponents:", Colors.CYAN)
        for component, loaded in integration_state.components_loaded.items():
            status = "✅" if loaded else "❌"
            print_colored(f"  {status} {component.replace('_', ' ').title()}", Colors.WHITE)
        print()
    
    def _show_performance_metrics(self):
        """Show performance metrics in CLI"""
        from ui import print_colored, Colors
        
        perf_summary = self.enhanced_app.get_performance_summary()
        
        print_colored("\\n📊 Performance Metrics:", Colors.YELLOW, bold=True)
        if perf_summary:
            for metric, stats in perf_summary.items():
                print_colored(f"{metric}:", Colors.CYAN)
                print_colored(f"  Average: {stats['avg']:.3f}s", Colors.WHITE)
                print_colored(f"  Min: {stats['min']:.3f}s", Colors.WHITE)
                print_colored(f"  Max: {stats['max']:.3f}s", Colors.WHITE)
                print_colored(f"  Count: {stats['count']}", Colors.WHITE)
        else:
            print_colored("No metrics available yet", Colors.YELLOW)
        print()

# Global instances for easy access
enhanced_app = EnhancedLangGraphApp()
streamlit_wrapper = StreamlitAppWrapper()
cli_wrapper = CLIAppWrapper()

def enhance_langgraph_app():
    """Main function to enhance LangGraph application"""
    logger.info("Enhancing LangGraph application with infrastructure integration")
    
    # Check if we're running in Streamlit
    try:
        import streamlit as st
        # We're in Streamlit
        streamlit_wrapper.initialize_session_state(st)
        streamlit_wrapper.add_infrastructure_status_sidebar(st)
        logger.info("Streamlit enhancement initialized")
        return streamlit_wrapper
    except:
        # We're in CLI mode
        logger.info("CLI enhancement initialized")
        return cli_wrapper

def get_enhanced_app() -> EnhancedLangGraphApp:
    """Get the enhanced application instance"""
    return enhanced_app

def get_integration_status() -> Dict[str, Any]:
    """Get current integration status"""
    return {
        'infrastructure_available': integration_state.infrastructure_available,
        'fallback_mode': integration_state.fallback_mode,
        'components_loaded': integration_state.components_loaded,
        'performance_metrics': enhanced_app.get_performance_summary(),
        'last_health_check': integration_state.last_health_check
    }

# Auto-enhance on import
if __name__ != "__main__":
    enhance_langgraph_app()
