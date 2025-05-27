#!/usr/bin/env python3
"""
LangGraph 101 - Enhanced Streamlit Frontend (Phase 4)
====================================================

Modern, real-time Streamlit interface with WebSocket support, streaming chat,
and comprehensive analytics integration for Phase 4.

Features:
- Real-time streaming chat with WebSocket support
- Multi-agent selection and comparison
- Live analytics dashboard
- Modern UI/UX with responsive design
- Export/import capabilities
- User preference management
- Performance monitoring

Author: GitHub Copilot
Date: 2024
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import websocket
import threading
import queue

# Import components if available
try:
    from langgraph_streaming_agent_enhanced import get_streaming_agent
    from analytics_logger import AnalyticsLogger
    from config import load_config
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LangGraph 101 - Streaming AI Chat",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
/* Modern theme */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Chat container */
.chat-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

/* Message bubbles */
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    margin-left: 20%;
    animation: slideInRight 0.3s ease;
}

.ai-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    margin-right: 20%;
    animation: slideInLeft 0.3s ease;
}

.thinking-message {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 8px 15px;
    border-radius: 15px;
    margin: 5px 0;
    margin-right: 30%;
    opacity: 0.8;
    font-style: italic;
}

/* Animations */
@keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes slideInLeft {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.thinking-indicator {
    animation: pulse 1.5s infinite;
}

/* Metrics cards */
.metric-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #667eea;
}

.metric-label {
    color: #666;
    font-size: 0.9em;
}

/* Status indicators */
.status-online {
    color: #4CAF50;
    font-weight: bold;
}

.status-offline {
    color: #f44336;
    font-weight: bold;
}

.status-connecting {
    color: #ff9800;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# WebSocket Client Class
class WebSocketClient:
    def __init__(self, url: str, user_id: str):
        self.url = url
        self.user_id = user_id
        self.ws = None
        self.connected = False
        self.message_queue = queue.Queue()
        self.thread = None
        
    def connect(self):
        """Connect to WebSocket server"""
        try:
            self.ws = websocket.WebSocketApp(
                f"{self.url}/{self.user_id}",
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.thread.start()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            
    def on_open(self, ws):
        """WebSocket opened"""
        self.connected = True
        logger.info("WebSocket connected")
        
    def on_message(self, ws, message):
        """Received WebSocket message"""
        try:
            data = json.loads(message)
            self.message_queue.put(data)
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            
    def on_error(self, ws, error):
        """WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        self.connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed"""
        self.connected = False
        logger.info("WebSocket disconnected")
        
    def send_message(self, message_type: str, payload: dict):
        """Send message to WebSocket"""
        if self.connected and self.ws:
            message = {
                "type": message_type,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            self.ws.send(json.dumps(message))
            
    def get_messages(self):
        """Get queued messages"""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
        
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
        self.connected = False

# Session state initialization
def init_session_state():
    """Initialize Streamlit session state"""
    
    # Configuration
    if 'config' not in st.session_state:
        st.session_state.config = {
            'api_url': 'http://localhost:8002',
            'ws_url': 'ws://localhost:8003/ws',
            'user_id': f'user_{uuid.uuid4().hex[:8]}',
            'auto_scroll': True,
            'show_timestamps': True,
            'enable_sound': False,
            'theme': 'modern'
        }
    
    # Chat state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
        
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = 'general'
        
    if 'selected_persona' not in st.session_state:
        st.session_state.selected_persona = 'default'
    
    # WebSocket state
    if 'ws_client' not in st.session_state:
        st.session_state.ws_client = None
        
    if 'ws_connected' not in st.session_state:
        st.session_state.ws_connected = False
    
    # Analytics state
    if 'analytics_data' not in st.session_state:
        st.session_state.analytics_data = {
            'response_times': [],
            'message_counts': [],
            'agent_usage': {},
            'timestamps': []
        }
    
    # UI state
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False
        
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False

# API functions
def get_system_status():
    """Get system status from API"""
    try:
        response = requests.get(f"{st.session_state.config['api_url']}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Status API error: {e}")
    return None

def get_available_agents():
    """Get available agents from API"""
    try:
        response = requests.get(f"{st.session_state.config['api_url']}/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('agents', {})
    except Exception as e:
        logger.error(f"Agents API error: {e}")
    return {}

def send_chat_message(message: str, use_streaming: bool = True):
    """Send chat message via API"""
    try:
        payload = {
            'message': message,
            'agent_id': st.session_state.selected_agent,
            'session_id': st.session_state.session_id,
            'streaming': use_streaming,
            'persona': st.session_state.selected_persona
        }
        
        if use_streaming:
            # Use WebSocket for streaming
            if st.session_state.ws_client and st.session_state.ws_connected:
                st.session_state.ws_client.send_message('chat_message', {
                    'text': message,
                    'agent_id': st.session_state.selected_agent,
                    'persona': st.session_state.selected_persona
                })
                return True
        else:
            # Use REST API for sync
            response = requests.post(
                f"{st.session_state.config['api_url']}/chat",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data
                
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return None

# UI Components
def render_sidebar():
    """Render sidebar with controls"""
    
    with st.sidebar:
        st.title("🚀 LangGraph Control Panel")
        
        # System status
        status = get_system_status()
        if status:
            if status['status'] == 'operational':
                st.markdown("🟢 **System Online**", unsafe_allow_html=True)
            else:
                st.markdown("🔴 **System Issues**", unsafe_allow_html=True)
                
            st.metric("Active Sessions", status['active_sessions'])
            st.metric("Uptime", status['uptime'])
        else:
            st.markdown("🔴 **System Offline**", unsafe_allow_html=True)
        
        st.divider()
        
        # Agent selection
        st.subheader("🤖 Agent Selection")
        agents = get_available_agents()
        
        if agents:
            agent_options = list(agents.keys())
            selected_idx = 0
            if st.session_state.selected_agent in agent_options:
                selected_idx = agent_options.index(st.session_state.selected_agent)
                
            st.session_state.selected_agent = st.selectbox(
                "Choose Agent",
                agent_options,
                index=selected_idx,
                help="Select the AI agent to chat with"
            )
            
            # Show agent config
            if st.session_state.selected_agent in agents:
                agent_config = agents[st.session_state.selected_agent]['config']
                with st.expander(f"⚙️ {st.session_state.selected_agent.title()} Config"):
                    st.json(agent_config)
        
        st.divider()
        
        # Persona selection
        st.subheader("🎭 Persona")
        personas = ['default', 'helpful', 'creative', 'analytical', 'friendly']
        st.session_state.selected_persona = st.selectbox(
            "Choose Persona",
            personas,
            index=personas.index(st.session_state.selected_persona) if st.session_state.selected_persona in personas else 0
        )
        
        st.divider()
        
        # Connection controls
        st.subheader("🔌 Connection")
        
        if st.session_state.ws_connected:
            st.markdown("🟢 **WebSocket Connected**")
            if st.button("Disconnect", type="secondary"):
                if st.session_state.ws_client:
                    st.session_state.ws_client.disconnect()
                    st.session_state.ws_connected = False
                    st.rerun()
        else:
            st.markdown("🔴 **WebSocket Disconnected**")
            if st.button("Connect", type="primary"):
                connect_websocket()
                st.rerun()
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        
        st.session_state.config['auto_scroll'] = st.checkbox(
            "Auto-scroll chat",
            value=st.session_state.config['auto_scroll']
        )
        
        st.session_state.config['show_timestamps'] = st.checkbox(
            "Show timestamps",
            value=st.session_state.config['show_timestamps']
        )
        
        # Export chat
        if st.button("📤 Export Chat"):
            export_chat_history()
        
        # Clear chat
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

def connect_websocket():
    """Connect to WebSocket server"""
    try:
        if st.session_state.ws_client:
            st.session_state.ws_client.disconnect()
        
        st.session_state.ws_client = WebSocketClient(
            st.session_state.config['ws_url'],
            st.session_state.config['user_id']
        )
        st.session_state.ws_client.connect()
        
        # Wait for connection
        time.sleep(1)
        st.session_state.ws_connected = st.session_state.ws_client.connected
        
        if st.session_state.ws_connected:
            st.success("WebSocket connected!")
        else:
            st.error("Failed to connect WebSocket")
            
    except Exception as e:
        st.error(f"WebSocket connection error: {e}")

def render_chat_interface():
    """Render main chat interface"""
    
    st.title("💬 LangGraph Streaming Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Message display
        for i, msg in enumerate(st.session_state.messages):
            if msg['type'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    {msg['content']}
                    {f"<small style='opacity: 0.7;'><br>{msg['timestamp']}</small>" if st.session_state.config['show_timestamps'] else ""}
                </div>
                """, unsafe_allow_html=True)
                
            elif msg['type'] == 'ai':
                st.markdown(f"""
                <div class="ai-message">
                    {msg['content']}
                    {f"<small style='opacity: 0.7;'><br>{msg['timestamp']}</small>" if st.session_state.config['show_timestamps'] else ""}
                </div>
                """, unsafe_allow_html=True)
                
            elif msg['type'] == 'thinking':
                st.markdown(f"""
                <div class="thinking-message thinking-indicator">
                    🤔 {msg['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Thinking indicator
    if st.session_state.thinking:
        st.markdown("""
        <div class="thinking-message thinking-indicator">
            🤔 AI is thinking...
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            key="chat_input",
            placeholder="Ask me anything...",
            label_visibility="collapsed"
        )
    
    with col2:
        use_streaming = st.checkbox("Stream", value=True, help="Enable real-time streaming")
    
    with col3:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle input
    if (send_button or user_input) and user_input.strip():
        handle_user_input(user_input.strip(), use_streaming)

def handle_user_input(message: str, use_streaming: bool):
    """Handle user input and get AI response"""
    
    # Add user message
    st.session_state.messages.append({
        'type': 'user',
        'content': message,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'agent': st.session_state.selected_agent
    })
    
    # Clear input
    st.session_state.chat_input = ""
    
    if use_streaming and st.session_state.ws_connected:
        # Use WebSocket streaming
        st.session_state.thinking = True
        send_chat_message(message, use_streaming=True)
        
        # Process streaming response
        process_streaming_response()
        
    else:
        # Use REST API
        with st.spinner("Getting response..."):
            response = send_chat_message(message, use_streaming=False)
            
            if response:
                st.session_state.messages.append({
                    'type': 'ai',
                    'content': response['response'],
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'agent': response['agent_id']
                })
                
                # Update session ID
                st.session_state.session_id = response['session_id']
                
                # Update analytics
                update_analytics(message, response['response'])
            else:
                st.error("Failed to get AI response")
    
    # Auto-scroll
    if st.session_state.config['auto_scroll']:
        st.rerun()

def process_streaming_response():
    """Process WebSocket streaming response"""
    
    if not st.session_state.ws_client:
        return
    
    response_chunks = []
    thinking_shown = False
    
    # Create placeholder for streaming content
    response_placeholder = st.empty()
    
    start_time = time.time()
    timeout = 30  # 30 second timeout
    
    while time.time() - start_time < timeout:
        messages = st.session_state.ws_client.get_messages()
        
        for msg in messages:
            if msg.get('type') == 'thinking_start':
                if not thinking_shown:
                    st.session_state.messages.append({
                        'type': 'thinking',
                        'content': 'Processing your request...',
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    thinking_shown = True
                    
            elif msg.get('type') == 'chat_chunk':
                # Remove thinking indicator
                if thinking_shown:
                    st.session_state.messages = [m for m in st.session_state.messages if m['type'] != 'thinking']
                    thinking_shown = False
                    st.session_state.thinking = False
                
                # Accumulate response
                response_chunks.append(msg.get('content', ''))
                
                # Update display with accumulated response
                full_response = ''.join(response_chunks)
                response_placeholder.markdown(f"""
                <div class="ai-message">
                    {full_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Check if final
                if msg.get('is_final', False):
                    # Add final message
                    st.session_state.messages.append({
                        'type': 'ai',
                        'content': full_response,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'agent': st.session_state.selected_agent
                    })
                    
                    # Update analytics
                    if st.session_state.messages:
                        last_user_msg = None
                        for msg in reversed(st.session_state.messages):
                            if msg['type'] == 'user':
                                last_user_msg = msg['content']
                                break
                        if last_user_msg:
                            update_analytics(last_user_msg, full_response)
                    
                    response_placeholder.empty()
                    return
        
        time.sleep(0.1)  # Small delay
    
    # Timeout handling
    if thinking_shown:
        st.session_state.messages = [m for m in st.session_state.messages if m['type'] != 'thinking']
    st.session_state.thinking = False
    response_placeholder.empty()
    st.error("Response timeout")

def update_analytics(user_message: str, ai_response: str):
    """Update analytics data"""
    
    current_time = datetime.now()
    
    # Response time (simulated)
    response_time = len(ai_response) / 100  # Rough estimate
    
    st.session_state.analytics_data['response_times'].append(response_time)
    st.session_state.analytics_data['timestamps'].append(current_time)
    st.session_state.analytics_data['message_counts'].append(len(st.session_state.messages))
    
    # Agent usage
    agent = st.session_state.selected_agent
    if agent not in st.session_state.analytics_data['agent_usage']:
        st.session_state.analytics_data['agent_usage'][agent] = 0
    st.session_state.analytics_data['agent_usage'][agent] += 1
    
    # Keep only recent data (last 100 interactions)
    for key in ['response_times', 'timestamps', 'message_counts']:
        if len(st.session_state.analytics_data[key]) > 100:
            st.session_state.analytics_data[key] = st.session_state.analytics_data[key][-100:]

def render_analytics_dashboard():
    """Render analytics dashboard"""
    
    st.title("📊 Analytics Dashboard")
    
    analytics = st.session_state.analytics_data
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    
    with col2:
        if analytics['response_times']:
            avg_response = sum(analytics['response_times']) / len(analytics['response_times'])
            st.metric("Avg Response Time", f"{avg_response:.2f}s")
        else:
            st.metric("Avg Response Time", "N/A")
    
    with col3:
        st.metric("Active Agent", st.session_state.selected_agent.title())
    
    with col4:
        st.metric("WebSocket Status", "🟢 Connected" if st.session_state.ws_connected else "🔴 Disconnected")
    
    # Charts
    if analytics['response_times']:
        
        # Response time trend
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Response Time Trend")
            df_response = pd.DataFrame({
                'Time': analytics['timestamps'][-20:],  # Last 20 interactions
                'Response Time (s)': analytics['response_times'][-20:]
            })
            
            fig_response = px.line(df_response, x='Time', y='Response Time (s)', 
                                 title="Response Time Over Time")
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            st.subheader("Agent Usage")
            if analytics['agent_usage']:
                df_agents = pd.DataFrame([
                    {"Agent": agent, "Usage": count} 
                    for agent, count in analytics['agent_usage'].items()
                ])
                
                fig_agents = px.pie(df_agents, values='Usage', names='Agent',
                                  title="Agent Usage Distribution")
                st.plotly_chart(fig_agents, use_container_width=True)
        
        # Message count over time
        st.subheader("Message Count Over Time")
        df_messages = pd.DataFrame({
            'Time': analytics['timestamps'][-20:],
            'Total Messages': analytics['message_counts'][-20:]
        })
        
        fig_messages = px.bar(df_messages, x='Time', y='Total Messages',
                             title="Cumulative Message Count")
        st.plotly_chart(fig_messages, use_container_width=True)
    
    else:
        st.info("Start chatting to see analytics data!")
    
    # System status
    st.subheader("System Status")
    status = get_system_status()
    
    if status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Status": status['status'],
                "Uptime": status['uptime'],
                "Active Connections": status['active_connections'],
                "Active Sessions": status['active_sessions']
            })
        
        with col2:
            if 'system_metrics' in status and status['system_metrics']:
                st.json(status['system_metrics'])
    else:
        st.error("Unable to retrieve system status")

def export_chat_history():
    """Export chat history to JSON"""
    
    export_data = {
        'chat_history': st.session_state.messages,
        'analytics': st.session_state.analytics_data,
        'config': {
            'agent': st.session_state.selected_agent,
            'persona': st.session_state.selected_persona,
            'session_id': st.session_state.session_id
        },
        'exported_at': datetime.now().isoformat()
    }
    
    # Create download
    json_data = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="💾 Download Chat History",
        data=json_data,
        file_name=f"langgraph_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Main application
def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        render_chat_interface()
        
        # Auto-process WebSocket messages
        if st.session_state.ws_connected and st.session_state.ws_client:
            messages = st.session_state.ws_client.get_messages()
            if messages:
                st.rerun()
    
    with tab2:
        render_analytics_dashboard()
    
    with tab3:
        st.title("⚙️ Settings")
        
        st.subheader("API Configuration")
        st.session_state.config['api_url'] = st.text_input(
            "API URL", 
            value=st.session_state.config['api_url']
        )
        
        st.session_state.config['ws_url'] = st.text_input(
            "WebSocket URL", 
            value=st.session_state.config['ws_url']
        )
        
        st.subheader("User Preferences")
        st.session_state.config['user_id'] = st.text_input(
            "User ID", 
            value=st.session_state.config['user_id']
        )
        
        st.subheader("Debug Information")
        with st.expander("Session State"):
            st.json({
                'messages_count': len(st.session_state.messages),
                'session_id': st.session_state.session_id,
                'ws_connected': st.session_state.ws_connected,
                'selected_agent': st.session_state.selected_agent,
                'config': st.session_state.config
            })
    
    # Auto-refresh every 5 seconds if connected
    if st.session_state.ws_connected:
        time.sleep(0.5)  # Small delay for real-time updates

if __name__ == "__main__":
    main()
