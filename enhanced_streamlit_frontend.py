#!/usr/bin/env python3
"""
LangGraph 101 - Enhanced Streamlit Frontend with Real-time Chat
==============================================================

Enhanced Streamlit frontend that provides real-time chat interface with WebSocket
support, streaming responses, analytics integration, and modern UI/UX.

Features:
- Real-time chat interface with WebSocket connections
- Streaming response display with typewriter effect
- Interactive analytics dashboard
- Modern responsive UI design
- Session management and persistence
- Multi-persona support with avatars
- Voice input/output capabilities
- Export and sharing functionality

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import json
import uuid
import time
import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import traceback

# Streamlit imports
import streamlit as st
from streamlit.runtime.caching import cache_data
from streamlit.runtime.state import SessionState
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# WebSocket client
try:
    import websocket
    import threading
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Server-Sent Events
try:
    import sseclient
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False

# Import existing components
from config import load_config
from personas import get_all_personas, get_persona_by_name
from history import get_history_manager
from memory_manager import get_memory_manager
from analytics_dashboard import render_analytics_dashboard
from content_dashboard import render_content_creation_dashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketClient:
    """WebSocket client for real-time communication"""
    
    def __init__(self, url: str, session_id: str):
        self.url = url
        self.session_id = session_id
        self.ws = None
        self.connected = False
        self.message_queue = []
        self.message_handlers = {}
        self.connection_thread = None
        
    def connect(self):
        """Connect to WebSocket server"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available")
            return False
        
        try:
            self.ws = websocket.WebSocketApp(
                f"{self.url}/ws/{self.session_id}",
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            self.connection_thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            self.connection_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.ws:
            self.ws.close()
        self.connected = False
    
    def send_message(self, message_type: str, payload: Dict[str, Any]):
        """Send message to WebSocket server"""
        if not self.connected or not self.ws:
            return False
        
        message = {
            "message_type": message_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    def _on_open(self, ws):
        """WebSocket open handler"""
        self.connected = True
        logger.info("WebSocket connected")
        
        # Update session state
        if "ws_connected" in st.session_state:
            st.session_state.ws_connected = True
    
    def _on_message(self, ws, message):
        """WebSocket message handler"""
        try:
            data = json.loads(message)
            message_type = data.get("message_type")
            payload = data.get("payload", {})
            
            # Handle message
            if message_type in self.message_handlers:
                self.message_handlers[message_type](payload)
            else:
                # Add to queue for processing
                self.message_queue.append(data)
                
                # Update session state for Streamlit
                if "ws_messages" in st.session_state:
                    st.session_state.ws_messages.append(data)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error handler"""
        logger.error(f"WebSocket error: {error}")
        self.connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket close handler"""
        self.connected = False
        logger.info("WebSocket disconnected")
        
        # Update session state
        if "ws_connected" in st.session_state:
            st.session_state.ws_connected = False


class APIClient:
    """API client for REST endpoints"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    async def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login and get access token"""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                token = data.get("access_token")
                user_info = data.get("user", {})
                
                # Update session headers
                self.session.headers.update({'Authorization': f'Bearer {token}'})
                
                return {
                    "access_token": token,
                    "user": user_info,
                    "token_type": data.get("token_type", "bearer")
                }
            else:
                logger.error(f"Login failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return None
    
    async def create_session(self, persona: str = "Default", 
                           streaming_mode: str = "text") -> Optional[str]:
        """Create a new streaming session"""
        try:
            payload = {
                "persona": persona,
                "streaming_mode": streaming_mode,
                "metadata": {"frontend": "streamlit"}
            }
            
            response = self.session.post(
                f"{self.base_url}/api/session/create",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("session_id")
            else:
                logger.error(f"Session creation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return None
    
    async def close_session(self, session_id: str) -> bool:
        """Close a streaming session"""
        try:
            payload = {"session_id": session_id}
            
            response = self.session.post(
                f"{self.base_url}/api/session/close",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("closed", False)
            else:
                logger.error(f"Session close failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Session close error: {e}")
            return False
    
    async def list_sessions(self) -> List[str]:
        """List active sessions"""
        try:
            response = self.session.get(f"{self.base_url}/api/session/list")
            
            if response.status_code == 200:
                data = response.json()
                return data.get("active_sessions", [])
            else:
                logger.error(f"Session list failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Session list error: {e}")
            return []
    
    async def send_chat_message(self, message: str, persona: str = "Default", 
                              session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Send chat message to API"""
        try:
            payload = {
                "message": message,
                "persona": persona,
                "session_id": session_id or str(uuid.uuid4())
            }
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Chat API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Chat API error: {e}")
            return None
    
    def stream_chat_message(self, message: str, persona: str = "Default", 
                           session_id: Optional[str] = None):
        """Stream chat message from API"""
        if not SSE_AVAILABLE:
            # Fallback to regular chat
            return self.send_chat_message(message, persona, session_id)
        
        try:
            payload = {
                "message": message,
                "persona": persona,
                "session_id": session_id or str(uuid.uuid4())
            }
            
            response = self.session.post(
                f"{self.base_url}/api/chat/stream",
                json=payload,
                stream=True
            )
            
            if response.status_code == 200:
                client = sseclient.SSEClient(response)
                for event in client.events():
                    try:
                        data = json.loads(event.data)
                        yield data
                    except json.JSONDecodeError:
                        continue
            else:
                logger.error(f"Stream API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Stream API error: {e}")
    
    async def get_personas(self) -> List[Dict[str, Any]]:
        """Get available personas"""
        try:
            response = self.session.get(f"{self.base_url}/api/personas")
            
            if response.status_code == 200:
                return response.json().get("personas", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Personas API error: {e}")
            return []


def initialize_enhanced_session_state():
    """Initialize enhanced session state for real-time features"""
    
    # Basic initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_persona" not in st.session_state:
        st.session_state.current_persona = get_persona_by_name("Default")
    
    # API configuration
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    
    if "api_token" not in st.session_state:
        st.session_state.api_token = None
    
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(st.session_state.api_base_url)
    
    # WebSocket configuration
    if "ws_client" not in st.session_state:
        ws_url = st.session_state.api_base_url.replace('http', 'ws')
        st.session_state.ws_client = WebSocketClient(ws_url, st.session_state.session_id)
    
    if "ws_connected" not in st.session_state:
        st.session_state.ws_connected = False
    
    if "ws_messages" not in st.session_state:
        st.session_state.ws_messages = []
    
    # Streaming state
    if "streaming_response" not in st.session_state:
        st.session_state.streaming_response = ""
    
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False
    
    # UI state
    if "show_analytics" not in st.session_state:
        st.session_state.show_analytics = False
    
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
      # Authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    
    # Backend session management
    if "backend_session_id" not in st.session_state:
        st.session_state.backend_session_id = None


def render_login_form():
    """Render login form"""
    st.title("🚀 LangGraph 101 - Enhanced Chat")
    
    with st.container():
        st.subheader("🔐 Authentication Required")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Login", use_container_width=True)
                  if submit:
                    if username and password:
                        with st.spinner("Authenticating..."):
                            auth_result = asyncio.run(st.session_state.api_client.login(username, password))
                            
                            if auth_result:
                                st.session_state.api_token = auth_result["access_token"]
                                st.session_state.user_info = auth_result["user"]
                                st.session_state.authenticated = True
                                
                                # Create a backend session for the authenticated user
                                session_id = asyncio.run(st.session_state.api_client.create_session(
                                    persona=st.session_state.current_persona.name,
                                    streaming_mode="event_stream"
                                ))
                                
                                if session_id:
                                    st.session_state.backend_session_id = session_id
                                    st.success(f"Login successful! Welcome, {auth_result['user']['username']}")
                                    st.rerun()
                                else:
                                    st.warning("Authentication successful but session creation failed")
                                    st.rerun()
                            else:
                                st.error("Invalid credentials")
                    else:
                        st.error("Please enter both username and password")
        
        # Demo credentials
        st.info("**Demo Credentials:**\n- Username: `admin`, Password: `admin123`\n- Username: `user`, Password: `user123`")


def render_chat_interface():
    """Render enhanced chat interface"""
    
    # Header with persona selection and controls
    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
    
    with col1:
        st.title("💬 Enhanced Chat")
    
    with col2:
        # Persona selection
        personas = get_all_personas()
        persona_names = [p.name for p in personas]
        current_index = persona_names.index(st.session_state.current_persona.name)
        
        selected_persona = st.selectbox(
            "Persona",
            persona_names,
            index=current_index,
            key="persona_selector"
        )
        
        if selected_persona != st.session_state.current_persona.name:
            st.session_state.current_persona = get_persona_by_name(selected_persona)
    
    with col3:
        # Chat mode selection
        chat_mode = st.selectbox(
            "Mode",
            ["websocket", "stream", "standard"],
            index=["websocket", "stream", "standard"].index(st.session_state.chat_mode),
            key="chat_mode_selector"
        )
        st.session_state.chat_mode = chat_mode
    
    with col4:
        # Connection status
        if st.session_state.chat_mode == "websocket":
            if st.session_state.ws_connected:
                st.success("🟢 Connected")
            else:
                st.error("🔴 Disconnected")
                if st.button("Connect", key="ws_connect"):
                    st.session_state.ws_client.connect()
    
    # Chat messages container
    messages_container = st.container()
    
    with messages_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    # AI message with persona avatar
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        st.image(
                            get_persona_avatar(message.get("persona", "Default")),
                            width=40
                        )
                    with col2:
                        if message.get("streaming", False):
                            # Typewriter effect for streaming
                            render_typewriter_text(message["content"], f"message_{i}")
                        else:
                            st.write(message["content"])
        
        # Streaming response display
        if st.session_state.is_streaming:
            with st.chat_message("assistant"):
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.image(
                        get_persona_avatar(st.session_state.current_persona.name),
                        width=40
                    )
                with col2:
                    streaming_placeholder = st.empty()
                    render_typewriter_text(
                        st.session_state.streaming_response,
                        "streaming_response",
                        placeholder=streaming_placeholder
                    )
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process message based on chat mode
        if st.session_state.chat_mode == "websocket":
            process_websocket_message(prompt)
        elif st.session_state.chat_mode == "stream":
            process_streaming_message(prompt)
        else:
            process_standard_message(prompt)
        
        st.rerun()


def process_websocket_message(message: str):
    """Process message via WebSocket"""
    if not st.session_state.ws_connected:
        st.error("WebSocket not connected. Please connect first.")
        return
    
    if not st.session_state.backend_session_id:
        st.error("No backend session available. Please start a new session.")
        return
    
    # Send message via WebSocket
    success = st.session_state.ws_client.send_message("chat", {
        "message": message,
        "persona": st.session_state.current_persona.name,
        "session_id": st.session_state.backend_session_id
    })
    
    if not success:
        st.error("Failed to send message via WebSocket")


def process_streaming_message(message: str):
    """Process message via streaming API"""
    if not st.session_state.backend_session_id:
        st.error("No backend session available. Please start a new session.")
        return
    
    st.session_state.is_streaming = True
    st.session_state.streaming_response = ""
    
    try:
        full_response = ""
        response_container = st.empty()
        
        for chunk in st.session_state.api_client.stream_chat_message(
            message=message,
            persona=st.session_state.current_persona.name,
            session_id=st.session_state.backend_session_id
        ):
            if chunk.get("chunk_type") == "text":
                content = chunk.get("content", "")
                full_response += content
                st.session_state.streaming_response = full_response
                
                # Update display in real-time
                response_container.write(full_response)
                time.sleep(0.01)  # Small delay for typewriter effect
            elif chunk.get("chunk_type") == "final":
                break
            elif chunk.get("chunk_type") == "error":
                st.error(f"Stream error: {chunk.get('content', 'Unknown error')}")
                return
        
        # Add final message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "persona": st.session_state.current_persona.name,
            "timestamp": datetime.now().isoformat(),
            "streaming": True,
            "session_id": st.session_state.backend_session_id
        })
        
    except Exception as e:
        st.error(f"Streaming error: {e}")
        logger.error(f"Streaming error: {traceback.format_exc()}")
    finally:
        st.session_state.is_streaming = False
        st.session_state.streaming_response = ""


def process_standard_message(message: str):
    """Process message via standard API"""
    if not st.session_state.backend_session_id:
        st.error("No backend session available. Please start a new session.")
        return
    
    with st.spinner("Processing..."):
        response = asyncio.run(st.session_state.api_client.send_chat_message(
            message=message,
            persona=st.session_state.current_persona.name,
            session_id=st.session_state.backend_session_id
        ))
        
        if response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("response", ""),
                "persona": st.session_state.current_persona.name,
                "timestamp": datetime.now().isoformat(),
                "session_id": st.session_state.backend_session_id
            })
        else:
            st.error("Failed to get response from API")


def render_typewriter_text(text: str, key: str, placeholder=None):
    """Render text with typewriter effect"""
    if placeholder is None:
        placeholder = st.empty()
    
    # Simple typewriter effect (can be enhanced with JavaScript)
    words = text.split()
    displayed_text = ""
    
    for word in words:
        displayed_text += word + " "
        placeholder.write(displayed_text)
        time.sleep(0.05)  # Adjust speed as needed


def get_persona_avatar(persona_name: str) -> str:
    """Get persona avatar path"""
    avatar_dir = os.path.join(os.path.dirname(__file__), "images")
    avatar_file = f"{persona_name.lower().replace(' ', '_')}.png"
    avatar_path = os.path.join(avatar_dir, avatar_file)
    
    if os.path.exists(avatar_path):
        return avatar_path
    else:
        return os.path.join(avatar_dir, "default.png")


def render_sidebar():
    """Render enhanced sidebar"""
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Connection status
        st.subheader("🔗 Connection")
        if st.session_state.chat_mode == "websocket":
            status = "Connected" if st.session_state.ws_connected else "Disconnected"
            color = "green" if st.session_state.ws_connected else "red"
            st.markdown(f"Status: :{color}[{status}]")
        
        # Settings
        st.subheader("🎨 Settings")
        
        # Dark mode toggle
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            apply_theme()
        
        # Auto-scroll
        auto_scroll = st.toggle("Auto Scroll", value=True)
        
        # Voice settings
        st.subheader("🔊 Voice")
        voice_enabled = st.toggle("Voice Responses", value=False)
        if voice_enabled:
            voice_speed = st.slider("Speed", 0.5, 2.0, 1.0)
            voice_pitch = st.slider("Pitch", 0.5, 2.0, 1.0)
          # Session management
        st.subheader("📊 Session")
        
        # Display user info
        if st.session_state.user_info:
            st.write(f"**User:** {st.session_state.user_info['username']}")
            st.write(f"**Roles:** {', '.join(st.session_state.user_info.get('roles', []))}")
        
        # Session details
        st.write(f"**Frontend ID:** `{st.session_state.session_id[:8]}...`")
        if st.session_state.backend_session_id:
            st.write(f"**Backend ID:** `{st.session_state.backend_session_id[:8]}...`")
        else:
            st.warning("No backend session")
        
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        # Session actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Session", use_container_width=True):
                start_new_session()
                st.rerun()
        
        with col2:
            if st.button("Export Chat", use_container_width=True):
                export_chat_history()
        
        # Active sessions list
        if st.button("List Active Sessions", use_container_width=True):
            try:
                active_sessions = asyncio.run(st.session_state.api_client.list_sessions())
                if active_sessions:
                    st.write("**Active Sessions:**")
                    for session in active_sessions[:5]:  # Show first 5
                        st.write(f"- `{session[:8]}...`")
                    if len(active_sessions) > 5:
                        st.write(f"... and {len(active_sessions) - 5} more")
                else:
                    st.write("No active sessions")
            except Exception as e:
                st.error(f"Failed to list sessions: {e}")
        
        # Analytics toggle
        st.subheader("📈 Analytics")
        if st.button("Show Analytics", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.show_analytics
        
        # Advanced settings
        with st.expander("🔧 Advanced"):
            st.text_input("API Base URL", value=st.session_state.api_base_url)
            st.number_input("Request Timeout", min_value=5, max_value=300, value=30)
            st.number_input("Max Messages", min_value=10, max_value=1000, value=100)


def apply_theme():
    """Apply dark/light theme"""
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stChatMessage {
            background-color: #1e1e1e;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .stChatMessage {
            background-color: #f0f0f0;
        }
        </style>
        """, unsafe_allow_html=True)


def start_new_session():
    """Start a new chat session"""
    # Clear frontend session
    old_session_id = st.session_state.session_id
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.streaming_response = ""
    st.session_state.is_streaming = False
    
    # Close old backend session if exists
    if st.session_state.backend_session_id:
        try:
            asyncio.run(st.session_state.api_client.close_session(st.session_state.backend_session_id))
        except Exception as e:
            logger.warning(f"Failed to close old session: {e}")
    
    # Create new backend session
    try:
        new_session_id = asyncio.run(st.session_state.api_client.create_session(
            persona=st.session_state.current_persona.name,
            streaming_mode="event_stream"
        ))
        
        if new_session_id:
            st.session_state.backend_session_id = new_session_id
            st.success("New session started!")
        else:
            st.error("Failed to create new backend session")
            
    except Exception as e:
        st.error(f"Session creation error: {e}")
    
    # Reconnect WebSocket with new session
    if st.session_state.chat_mode == "websocket":
        st.session_state.ws_client.disconnect()
        ws_url = st.session_state.api_base_url.replace('http', 'ws')
        st.session_state.ws_client = WebSocketClient(ws_url, st.session_state.backend_session_id or st.session_state.session_id)


def export_chat_history():
    """Export chat history"""
    try:
        export_data = {
            "session_id": st.session_state.session_id,
            "persona": st.session_state.current_persona.name,
            "messages": st.session_state.messages,
            "exported_at": datetime.now().isoformat()
        }
        
        st.download_button(
            label="Download Chat History",
            data=json.dumps(export_data, indent=2),
            file_name=f"chat_history_{st.session_state.session_id[:8]}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Export error: {e}")


def render_analytics_tab():
    """Render analytics dashboard"""
    st.title("📊 Analytics Dashboard")
    
    # Message statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    
    with col2:
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("User Messages", user_messages)
    
    with col3:
        ai_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.metric("AI Messages", ai_messages)
    
    # Message timeline
    if st.session_state.messages:
        df = pd.DataFrame(st.session_state.messages)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            
            # Messages by hour
            hourly_counts = df.groupby("hour").size().reset_index(name="count")
            
            fig = px.bar(
                hourly_counts,
                x="hour",
                y="count",
                title="Messages by Hour",
                labels={"hour": "Hour of Day", "count": "Message Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Message length distribution
        df["message_length"] = df["content"].str.len()
        
        fig = px.histogram(
            df,
            x="message_length",
            color="role",
            title="Message Length Distribution",
            labels={"message_length": "Message Length (characters)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent messages
        st.subheader("Recent Messages")
        recent_messages = df.tail(10)[["role", "content", "timestamp"]]
        st.dataframe(recent_messages, use_container_width=True)


def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="LangGraph 101 - Enhanced Chat",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_enhanced_session_state()
    
    # Apply theme
    apply_theme()
    
    # Authentication check
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        render_login_form()
        return
    
    # Main app layout
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            render_chat_interface()
        
        with col2:
            render_sidebar()
    
    with tab2:
        if st.session_state.show_analytics:
            render_analytics_tab()
        else:
            st.info("Enable analytics in the sidebar to view this tab.")


if __name__ == "__main__":
    main()
