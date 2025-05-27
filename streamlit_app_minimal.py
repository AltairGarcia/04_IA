"""
Streamlit web interface for LangGraph 101 project - Python 3.13 Compatible Version.

This is a simplified version that handles missing dependencies gracefully.
"""
import streamlit as st
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Safe imports
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available. Install with: pip install pandas")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("Psutil not available. Install with: pip install psutil")

# Import safe_imports for dependency checking
try:
    from safe_imports import VOICE_INPUT_AVAILABLE, GOOGLE_AI_AVAILABLE
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    GOOGLE_AI_AVAILABLE = False

# Try to import core modules with fallbacks
try:
    from config import load_config
except ImportError:
    logger.warning("config.py not found, using fallback")
    def load_config():
        return {
            'api_key': os.getenv('GOOGLE_GEMINI_API_KEY', ''),
            'model_name': 'gemini-2.0-flash',
            'temperature': 0.7
        }

# Set page configuration
st.set_page_config(
    page_title="LangGraph 101 - AI Agent Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {
            'dependencies': {
                'plotly': PLOTLY_AVAILABLE,
                'pandas': PANDAS_AVAILABLE,
                'psutil': PSUTIL_AVAILABLE,
                'voice_input': VOICE_INPUT_AVAILABLE,
                'google_ai': GOOGLE_AI_AVAILABLE
            }
        }

def display_system_status():
    """Display system status and dependency information."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    status = st.session_state.system_status['dependencies']
    
    for dep_name, available in status.items():
        if available:
            st.sidebar.success(f"✅ {dep_name.replace('_', ' ').title()}")
        else:
            st.sidebar.error(f"❌ {dep_name.replace('_', ' ').title()}")

def display_analytics_dashboard():
    """Display a basic analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    if PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
        # Create sample data for demonstration
        import pandas as pd
        import plotly.express as px
        
        # Sample usage data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        usage_data = pd.DataFrame({
            'Date': dates,
            'Messages': [20 + i % 10 for i in range(len(dates))],
            'API Calls': [50 + i % 25 for i in range(len(dates))]
        })
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.messages))
        with col2:
            st.metric("System Health", "🟢 Healthy")
        with col3:
            st.metric("API Status", "🟢 Online" if GOOGLE_AI_AVAILABLE else "🔴 Offline")
        with col4:
            st.metric("Dependencies", f"{sum(st.session_state.system_status['dependencies'].values())}/{len(st.session_state.system_status['dependencies'])}")
        
        # Usage chart
        st.subheader("📈 Usage Trends")
        fig = px.line(usage_data, x='Date', y=['Messages', 'API Calls'], 
                     title='Daily Usage Statistics')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("📊 Analytics features require pandas and plotly. Please install them:")
        st.code("pip install pandas plotly")

def display_system_health():
    """Display system health monitoring."""
    st.header("🏥 System Health Monitor")
    
    if PSUTIL_AVAILABLE:
        import psutil
        
        # System metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_percent = psutil.cpu_percent(interval=1)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%", 
                     delta="Normal" if cpu_percent < 80 else "High")
        
        with col2:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent:.1f}%",
                     delta="Normal" if memory.percent < 80 else "High")
        
        with col3:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            st.metric("Disk Usage", f"{disk_percent:.1f}%",
                     delta="Normal" if disk_percent < 90 else "High")
        
        # System info
        st.subheader("💻 System Information")
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Platform:** {psutil.Platform if hasattr(psutil, 'Platform') else 'Unknown'}")
        st.write(f"**CPU Cores:** {psutil.cpu_count()}")
        st.write(f"**Total Memory:** {memory.total / (1024**3):.1f} GB")
        
    else:
        st.warning("🏥 Health monitoring requires psutil. Please install it:")
        st.code("pip install psutil")

def display_chat_interface():
    """Display a basic chat interface."""
    st.header("💬 AI Chat Interface")
    
    # Configuration status
    config = st.session_state.config
    api_key = config.get('api_key', '')
    
    if not api_key:
        st.error("🔑 Gemini API key not configured. Please set GOOGLE_GEMINI_API_KEY in your .env file.")
        return
    
    st.success(f"🔑 API Key configured (ends with: ...{api_key[-6:]})")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response (placeholder for now)
        with st.chat_message("assistant"):
            if GOOGLE_AI_AVAILABLE:
                response = f"I received your message: '{prompt}'. The Gemini API is configured and ready to use!"
            else:
                response = f"I received your message: '{prompt}'. However, Google AI is not available. Please install: pip install google-generativeai"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def display_dependency_manager():
    """Display dependency management interface."""
    st.header("📦 Dependency Manager")
    
    st.subheader("🔍 Missing Dependencies")
    
    missing_deps = []
    if not PLOTLY_AVAILABLE:
        missing_deps.append("plotly")
    if not PANDAS_AVAILABLE:
        missing_deps.append("pandas")
    if not PSUTIL_AVAILABLE:
        missing_deps.append("psutil")
    if not GOOGLE_AI_AVAILABLE:
        missing_deps.append("google-generativeai")
    if not VOICE_INPUT_AVAILABLE:
        missing_deps.append("speech libraries (Python 3.13 compatibility issues)")
    
    if missing_deps:
        st.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
        
        st.subheader("🛠️ Installation Commands")
        for dep in missing_deps:
            if dep != "speech libraries (Python 3.13 compatibility issues)":
                st.code(f"pip install {dep}")
        
        st.info("💡 **Note**: Voice input features are disabled in Python 3.13 due to compatibility issues with the `aifc` module. This is a known limitation.")
        
    else:
        st.success("✅ All core dependencies are available!")
    
    st.subheader("📋 Recommended Installation")
    st.code("""
# Install core dependencies for Python 3.13
pip install streamlit plotly pandas psutil google-generativeai

# Optional: Install additional packages
pip install langchain openai anthropic
""")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("🤖 LangGraph 101")
    st.sidebar.markdown("*AI Agent Platform*")
    
    # Display system status
    display_system_status()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🏥 Health", "📦 Dependencies"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_analytics_dashboard()
    
    with tab3:
        display_system_health()
    
    with tab4:
        display_dependency_manager()
    
    # Footer
    st.markdown("---")
    st.markdown("*LangGraph 101 - Python 3.13 Compatible Version*")
    
    # Show Python version info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Python:** {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    st.sidebar.markdown(f"**Streamlit:** {st.__version__}")

if __name__ == "__main__":
    main()
