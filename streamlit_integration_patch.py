#!/usr/bin/env python3
"""
LangGraph 101 - Streamlit Integration Patch
==========================================

This file patches the existing Streamlit application to integrate with
the new infrastructure while maintaining full backward compatibility.

The patch:
- Adds infrastructure integration without breaking existing functionality
- Provides enhanced features when infrastructure is available
- Falls back gracefully to original behavior when infrastructure is unavailable
- Adds monitoring and performance tracking
- Enhances the UI with infrastructure status information

Author: GitHub Copilot
Date: 2024
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import integration wrapper
try:
    from app_integration_wrapper import streamlit_wrapper, get_enhanced_app, get_integration_status
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    print("⚠️  Integration wrapper not available - running in original mode")

def patch_streamlit_app():
    """Apply integration patches to Streamlit application"""
    
    if not INTEGRATION_AVAILABLE:
        return
    
    # Initialize session state with enhanced features
    streamlit_wrapper.initialize_session_state(st)
    
    # Add infrastructure status to sidebar
    streamlit_wrapper.add_infrastructure_status_sidebar(st)
    
    # Enhanced page configuration
    if hasattr(st, 'set_page_config') and 'page_config_set' not in st.session_state:
        st.set_page_config(
            page_title="LangGraph 101 - Enhanced",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/langgraph-101',
                'Report a bug': 'https://github.com/your-repo/langgraph-101/issues',
                'About': "LangGraph 101 - Enhanced with Infrastructure Integration"
            }
        )
        st.session_state.page_config_set = True

def enhance_chat_functionality():
    """Enhance chat functionality with infrastructure features"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    def enhanced_chat_processor(message: str, user_id: str = "streamlit_user", context: dict = None):
        """Enhanced chat message processor"""
        try:
            enhanced_app = get_enhanced_app()
            result = enhanced_app.process_message(message, user_id, context or {})
            
            # Store in session state for metrics
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({
                'message': message,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            st.error(f"Error in enhanced chat processing: {e}")
            return {
                'response': f"Error: {str(e)}",
                'status': 'error',
                'mode': 'fallback'
            }
    
    return enhanced_chat_processor

def add_infrastructure_monitoring_tab():
    """Add infrastructure monitoring tab to the application"""
    
    if not INTEGRATION_AVAILABLE:
        return
    
    # Create monitoring tab
    with st.expander("🔧 Infrastructure Monitoring", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Status")
            integration_status = get_integration_status()
            
            # Infrastructure availability
            if integration_status['infrastructure_available']:
                st.success("✅ Infrastructure Available")
            else:
                st.warning("⚠️ Fallback Mode")
            
            # Component status
            st.write("**Components:**")
            for component, loaded in integration_status['components_loaded'].items():
                icon = "✅" if loaded else "❌"
                st.write(f"{icon} {component.replace('_', ' ').title()}")
        
        with col2:
            st.subheader("Performance Metrics")
            perf_metrics = integration_status['performance_metrics']
            
            if perf_metrics:
                for metric, stats in perf_metrics.items():
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=f"{stats['avg']:.3f}s",
                        delta=f"Last: {stats['last']:.3f}s"
                    )
            else:
                st.info("No performance metrics available yet")
        
        # Chat history analysis
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.subheader("Chat Analytics")
            
            total_messages = len(st.session_state.chat_history)
            success_rate = sum(1 for chat in st.session_state.chat_history 
                             if chat['result'].get('status') == 'success') / total_messages * 100
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Total Messages", total_messages)
            with col4:
                st.metric("Success Rate", f"{success_rate:.1f}%")

def add_enhanced_sidebar_features():
    """Add enhanced features to the sidebar"""
    
    if not INTEGRATION_AVAILABLE:
        return
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("🚀 Enhanced Features")
        
        # Infrastructure controls
        if st.button("🔄 Refresh Infrastructure"):
            # Trigger infrastructure refresh
            st.rerun()
        
        # Performance mode toggle
        perf_mode = st.checkbox("📊 Performance Monitoring", value=False)
        if perf_mode:
            st.session_state.show_performance = True
        else:
            st.session_state.show_performance = False
        
        # Cache controls
        if st.button("🗑️ Clear Cache"):
            # Clear session cache
            for key in list(st.session_state.keys()):
                if key.startswith('cache_'):
                    del st.session_state[key]
            st.success("Cache cleared!")

def enhance_error_handling():
    """Enhance error handling with better user experience"""
    
    if not INTEGRATION_AVAILABLE:
        return
    
    # Custom error handler for the session
    def custom_error_handler(error: Exception, context: str = ""):
        """Custom error handler with enhanced logging"""
        error_msg = f"Error in {context}: {str(error)}"
        
        # Log error
        st.error(f"🚨 {error_msg}")
        
        # Store error for monitoring
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append({
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        # Show error details in expander
        with st.expander("Error Details", expanded=False):
            st.code(f"Error Type: {type(error).__name__}")
            st.code(f"Error Message: {str(error)}")
            if context:
                st.code(f"Context: {context}")
    
    return custom_error_handler

def add_performance_indicators():
    """Add performance indicators to the UI"""
    
    if not INTEGRATION_AVAILABLE or not st.session_state.get('show_performance', False):
        return
    
    # Performance indicators container
    perf_container = st.container()
    
    with perf_container:
        st.markdown("---")
        st.subheader("⚡ Performance Indicators")
        
        # Real-time metrics
        integration_status = get_integration_status()
        perf_metrics = integration_status['performance_metrics']
        
        if perf_metrics:
            cols = st.columns(len(perf_metrics))
            
            for i, (metric, stats) in enumerate(perf_metrics.items()):
                with cols[i]:
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=f"{stats['last']:.3f}s",
                        delta=f"Avg: {stats['avg']:.3f}s"
                    )

def patch_chat_interface():
    """Patch the chat interface with enhanced functionality"""
    
    if not INTEGRATION_AVAILABLE:
        return None
    
    # Enhanced chat processor
    enhanced_processor = enhance_chat_functionality()
    
    def enhanced_chat_ui():
        """Enhanced chat UI with infrastructure integration"""
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Chat input
            if prompt := st.chat_input("Type your message here..."):
                # Add user message to chat history
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process with enhanced functionality
                with st.chat_message("assistant"):
                    with st.spinner("Processing with enhanced infrastructure..."):
                        result = enhanced_processor(prompt)
                    
                    # Display response
                    if result.get('status') == 'success':
                        response = result.get('response', '')
                        st.markdown(response)
                        
                        # Show mode indicator
                        mode = result.get('mode', 'enhanced')
                        if mode == 'fallback':
                            st.caption("⚠️ Response generated in fallback mode")
                        else:
                            st.caption("✅ Response generated with enhanced infrastructure")
                    else:
                        st.error(f"Error: {result.get('response', 'Unknown error')}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result.get('response', 'Error occurred')
                    })
            
            # Display chat history
            if 'messages' in st.session_state:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
    
    return enhanced_chat_ui

def initialize_enhanced_streamlit():
    """Initialize all Streamlit enhancements"""
    
    # Apply basic patches
    patch_streamlit_app()
    
    # Add enhanced features
    add_enhanced_sidebar_features()
    
    # Add monitoring
    add_infrastructure_monitoring_tab()
    
    # Add performance indicators
    add_performance_indicators()
    
    # Setup error handling
    error_handler = enhance_error_handling()
    if error_handler:
        st.session_state.error_handler = error_handler
    
    # Return enhanced chat interface
    return patch_chat_interface()

# Main execution
if __name__ == "__main__":
    # This would typically be imported by the main streamlit_app.py
    st.title("🚀 LangGraph 101 - Enhanced Edition")
    
    # Initialize enhancements
    enhanced_chat = initialize_enhanced_streamlit()
    
    if enhanced_chat:
        # Use enhanced chat interface
        enhanced_chat()
    else:
        # Fallback to basic interface
        st.info("Running in basic mode - enhanced features not available")
        
        # Simple chat interface
        if prompt := st.chat_input("Type your message here..."):
            st.chat_message("user").markdown(prompt)
            st.chat_message("assistant").markdown(f"Echo: {prompt} (basic mode)")

# Export functions for use in main streamlit app
__all__ = [
    'patch_streamlit_app',
    'enhance_chat_functionality', 
    'add_infrastructure_monitoring_tab',
    'add_enhanced_sidebar_features',
    'enhance_error_handling',
    'add_performance_indicators',
    'patch_chat_interface',
    'initialize_enhanced_streamlit'
]
