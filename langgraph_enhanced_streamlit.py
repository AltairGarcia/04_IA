#!/usr/bin/env python3
"""
LangGraph 101 - Enhanced Streamlit Application
==============================================

This is the enhanced version of the Streamlit application that integrates with
the new infrastructure while maintaining full backward compatibility.

This application automatically detects and uses infrastructure components
when available, falling back gracefully to original functionality when not.

Features added:
- Infrastructure integration with monitoring sidebar
- Performance indicators and metrics
- Enhanced chat interface with infrastructure features
- Real-time system status monitoring
- Advanced caching and optimization

Usage:
    streamlit run langgraph_enhanced_streamlit.py

Author: GitHub Copilot
Date: 2024
"""

import sys
import os
import streamlit as st

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set page configuration first
st.set_page_config(
    page_title="LangGraph 101 - Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_integration_status():
    """Show integration status in sidebar"""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        try:
            from app_integration_wrapper import get_integration_status
            status = get_integration_status()
            
            if status.get('infrastructure_available', False):
                st.success("✅ Infrastructure Mode")
                st.info("Enhanced features enabled")
            else:
                st.warning("⚠️ Fallback Mode")
                st.info("Basic features active")
                
            # Show component status
            components = status.get('components', {})
            if components:
                st.markdown("#### Component Status")
                for comp, available in components.items():
                    icon = "✅" if available else "❌"
                    st.markdown(f"{icon} {comp.replace('_', ' ').title()}")
                    
        except Exception as e:
            st.error(f"Status check failed: {e}")

# Try to apply Streamlit integration patches
integration_applied = False

try:
    from streamlit_integration_patch import patch_streamlit_app
    from app_integration_wrapper import get_integration_status
    
    # Show enhanced header
    st.title("🚀 LangGraph 101 - Enhanced Edition")
    
    # Show integration status
    show_integration_status()
    
    # Apply Streamlit integration patch
    streamlit_app_function = patch_streamlit_app()
    
    if streamlit_app_function:
        st.success("✅ Infrastructure integration active")
        integration_applied = True
        
        # Run enhanced Streamlit app
        streamlit_app_function()
    else:
        st.warning("⚠️ Running in fallback mode")
    
except ImportError as e:
    st.warning(f"⚠️ Integration patches not available: {e}")
    st.info("Running original Streamlit application")

# Fallback to original application if integration fails or not available
if not integration_applied:
    try:
        # Load and execute original streamlit_app.py content
        with open('streamlit_app.py', 'r') as f:
            original_app_code = f.read()
        
        # Remove any streamlit configuration that might conflict
        lines = original_app_code.split('\n')
        filtered_lines = []
        skip_next = False
        
        for line in lines:
            if 'st.set_page_config' in line:
                skip_next = True
                continue
            if skip_next and line.strip() == ')':
                skip_next = False
                continue
            if not skip_next:
                filtered_lines.append(line)
        
        filtered_code = '\n'.join(filtered_lines)
        
        # Execute the original app code
        exec(filtered_code)
        
    except Exception as e:
        st.error(f"❌ Failed to load original application: {e}")
        st.exception(e)
