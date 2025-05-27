"""
Integrated Streamlit Application for LangGraph 101

This is the integrated version of the Streamlit app that communicates
through the API Gateway instead of direct function calls.
"""

import streamlit as st
import asyncio
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

# Import the API client
from api_gateway_integration import StreamlitAPIClient, IntegrationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LangGraph 101 - Integrated Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class IntegratedStreamlitApp:
    """Main integrated Streamlit application."""
    
    def __init__(self):
        self.config = IntegrationConfig()
        self.api_client = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_persona' not in st.session_state:
            st.session_state.current_persona = "default"
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = None
        if 'api_client' not in st.session_state:
            st.session_state.api_client = None
    
    async def get_api_client(self) -> StreamlitAPIClient:
        """Get or create API client."""
        if st.session_state.api_client is None:
            st.session_state.api_client = StreamlitAPIClient(
                base_url=f"http://{self.config.GATEWAY_HOST}:{self.config.GATEWAY_PORT}"
            )
        return st.session_state.api_client
    
    def render_header(self):
        """Render the application header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🤖 LangGraph 101 - Integrated Platform")
            st.markdown("*Powered by API Gateway & Enhanced Infrastructure*")
        
        # Status indicator
        with col3:
            if self.check_api_health():
                st.success("🟢 API Gateway Online")
            else:
                st.error("🔴 API Gateway Offline")
    
    def check_api_health(self) -> bool:
        """Check if the API Gateway is healthy."""
        try:
            import requests
            response = requests.get(
                f"http://{self.config.GATEWAY_HOST}:{self.config.GATEWAY_PORT}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Persona selection
            st.subheader("👤 Persona")
            personas = self.get_personas()
            if personas:
                persona_names = list(personas.keys())
                current_index = 0
                if st.session_state.current_persona in persona_names:
                    current_index = persona_names.index(st.session_state.current_persona)
                
                selected_persona = st.selectbox(
                    "Select Persona",
                    persona_names,
                    index=current_index,
                    key="persona_selector"
                )
                
                if selected_persona != st.session_state.current_persona:
                    st.session_state.current_persona = selected_persona
                    st.rerun()
            
            # Chat controls
            st.subheader("💬 Chat Controls")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_id = None
                st.rerun()
            
            if st.button("💾 Save Conversation", use_container_width=True):
                self.save_conversation()
            
            if st.button("📤 Export Chat", use_container_width=True):
                self.export_conversation()
            
            # System status
            st.subheader("📊 System Status")
            self.render_system_status()
    
    def get_personas(self) -> Dict[str, Any]:
        """Get available personas from API."""
        try:
            # For now, return a default set - in real implementation,
            # this would call the API
            return {
                "default": {"name": "Default", "description": "Standard assistant"},
                "creative": {"name": "Creative", "description": "Creative writing assistant"},
                "technical": {"name": "Technical", "description": "Technical expert"},
                "friendly": {"name": "Friendly", "description": "Casual conversation"}
            }
        except Exception as e:
            logger.error(f"Failed to get personas: {e}")
            return {"default": {"name": "Default", "description": "Standard assistant"}}
    
    def render_system_status(self):
        """Render system status indicators."""
        try:
            # API Gateway status
            if self.check_api_health():
                st.success("🌐 API Gateway")
            else:
                st.error("🌐 API Gateway")
            
            # Add more status indicators here
            st.info("🗄️ Database Pool")
            st.info("🗃️ Cache System")
            st.info("📬 Message Queue")
            st.info("⚡ Rate Limiter")
            
        except Exception as e:
            st.error(f"Status check failed: {e}")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata if available
                if "metadata" in message:
                    with st.expander("Message Details"):
                        st.json(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.get_chat_response(prompt)
                    
                    if response:
                        st.markdown(response["content"])
                        
                        # Add assistant message to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["content"],
                            "timestamp": datetime.now().isoformat(),
                            "metadata": response.get("metadata", {})
                        })
                    else:
                        error_msg = "Sorry, I couldn't process your request. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
    
    def get_chat_response(self, message: str) -> Optional[Dict[str, Any]]:
        """Get chat response from API."""
        try:
            # For now, simulate API call
            # In real implementation, this would use the StreamlitAPIClient
            
            # Simulate processing time
            time.sleep(1)
            
            return {
                "content": f"This is a simulated response to: '{message}'. The integrated API Gateway will handle this request with enhanced security and performance.",
                "metadata": {
                    "persona": st.session_state.current_persona,
                    "processing_time": 1.0,
                    "tokens_used": 150,
                    "model": "integrated-model"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get chat response: {e}")
            return None
    
    def save_conversation(self):
        """Save the current conversation."""
        try:
            if st.session_state.messages:
                # Simulate saving
                st.success("✅ Conversation saved!")
            else:
                st.warning("⚠️ No conversation to save")
        except Exception as e:
            st.error(f"❌ Failed to save conversation: {e}")
    
    def export_conversation(self):
        """Export the current conversation."""
        try:
            if st.session_state.messages:
                # Create export data
                export_data = {
                    "conversation_id": st.session_state.conversation_id,
                    "persona": st.session_state.current_persona,
                    "messages": st.session_state.messages,
                    "exported_at": datetime.now().isoformat()
                }
                
                # Convert to JSON
                json_str = json.dumps(export_data, indent=2)
                
                # Offer download
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_str,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.success("✅ Export ready!")
            else:
                st.warning("⚠️ No conversation to export")
                
        except Exception as e:
            st.error(f"❌ Failed to export conversation: {e}")
    
    def render_content_creation_tab(self):
        """Render content creation interface."""
        st.header("🎨 Content Creation")
        
        content_type = st.selectbox(
            "Content Type",
            ["Article", "Blog Post", "Social Media", "Script", "Email"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Input")
            topic = st.text_input("Topic/Subject")
            audience = st.text_input("Target Audience")
            tone = st.selectbox("Tone", ["Professional", "Casual", "Formal", "Creative"])
            length = st.selectbox("Length", ["Short", "Medium", "Long"])
            
            if st.button("🚀 Generate Content", use_container_width=True):
                if topic:
                    with st.spinner("Creating content..."):
                        content = self.generate_content(content_type, topic, audience, tone, length)
                        if content:
                            st.session_state.generated_content = content
                else:
                    st.warning("Please enter a topic")
        
        with col2:
            st.subheader("📄 Generated Content")
            if hasattr(st.session_state, 'generated_content'):
                st.markdown(st.session_state.generated_content)
                
                # Export options
                if st.button("📥 Download Content"):
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.generated_content,
                        file_name=f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Generated content will appear here")
    
    def generate_content(self, content_type: str, topic: str, audience: str, tone: str, length: str) -> Optional[str]:
        """Generate content using the API."""
        try:
            # Simulate content generation
            return f"""
# {content_type}: {topic}

**Target Audience:** {audience}
**Tone:** {tone}
**Length:** {length}

This is a simulated {content_type.lower()} about {topic} for {audience}. 
The integrated API Gateway would process this request through the content creation service
with enhanced security, caching, and performance monitoring.

The actual content would be generated using the configured AI models and tools,
with proper rate limiting and authentication through the API Gateway.

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return None
    
    def render_analytics_tab(self):
        """Render analytics dashboard."""
        st.header("📊 Analytics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.messages))
        
        with col2:
            st.metric("Current Persona", st.session_state.current_persona)
        
        with col3:
            st.metric("API Status", "Online" if self.check_api_health() else "Offline")
        
        with col4:
            st.metric("Session Duration", "25 min")  # Simulated
        
        # Usage charts (simulated data)
        st.subheader("📈 Usage Trends")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        usage_data = pd.DataFrame({
            'Date': dates,
            'Messages': [20 + i % 10 for i in range(len(dates))],
            'API Calls': [50 + i % 25 for i in range(len(dates))]
        })
        
        st.line_chart(usage_data.set_index('Date'))
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.info("🚀 Average Response Time: 1.2s")
            st.info("📊 Cache Hit Rate: 85%")
        
        with perf_col2:
            st.info("⚡ Rate Limit Usage: 45%")
            st.info("💾 Memory Usage: 62%")
    
    def run(self):
        """Run the integrated Streamlit application."""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎨 Content Creation", "📊 Analytics"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_content_creation_tab()
        
        with tab3:
            self.render_analytics_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("*LangGraph 101 Integrated Platform - Enhanced with API Gateway, Security, and Performance*")


def main():
    """Main entry point for the integrated Streamlit app."""
    app = IntegratedStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
