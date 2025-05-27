"""
Streamlit web interface for LangGraph 101 project.

This module provides a web interface for interacting with the LangGraph agent.
"""
from datetime import datetime, timedelta  # Added timedelta
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import uuid
import os
import base64
import pandas as pd
import tempfile
import json
import time
import requests  # Added for ConnectionError
import logging  # Added for logging
import plotly.express as px  # Added plotly.express

# Import local modules
from config import load_config
from agent import create_agent, invoke_agent
from tools import get_tools
from personas import get_persona_by_name, get_all_personas, Persona
from history import ConversationHistory
from memory_manager import get_memory_manager, MemoryItem
from database import Database
from export import export_conversation, get_export_formats
from email_sender import email_conversation
from voice import get_voice_manager
from voice_input import get_voice_input_manager
# Import agentic capabilities
from agents import get_agent_types, create_agent as create_autonomous_agent
from agents import AgentStatus, TaskConfig, Task # Ensure Task is imported if used by create_task
from agent_commands import process_agent_command, is_agent_command

# Import error handling, analytics and performance optimization
from analytics_dashboard import render_analytics_dashboard
from content_dashboard import render_content_creation_dashboard # Added this import
from error_handling import ErrorHandler, graceful_degradation
from system_initialization import initialize_all_systems, check_system_status  # Changed get_system_status to check_system_status
from error_integration import GlobalErrorHandler

# Import robust configuration and health checking
from config_robust import load_config_robust, ConfigError as RobustConfigError
from app_health import get_health_summary, start_health_monitoring
from langchain_robust import suppress_langchain_warnings
from health_server import start_health_server

# Import chat interface
from streamlit_components.chat import render_chat_interface # Changed this line

# Import authentication and monitoring systems
from auth_middleware import AuthenticationManager, require_authentication, require_role
from monitoring_dashboard import MonitoringDashboard
from production_features import (
    RateLimiter, InputValidator, rate_limit_check, 
    display_rate_limit_info, SecurityManager
)

# Initialize logger
logger = logging.getLogger(__name__)
# Ensure logger is configured (basic example, can be more sophisticated)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# SECURITY NOTE: For production, add authentication to restrict access to the dashboard.
# Streamlit does not natively support authentication, but you can use solutions like streamlit-authenticator or reverse proxy authentication.
# See: https://github.com/mkhorasani/Streamlit-Authenticator

# Initialize all systems for error handling, analytics, and performance optimization
# This is only done once at module level, and system_initialization
# module prevents multiple initializations
if __name__ != "__main__":  # Only initialize here if imported as a module
    try:
        with suppress_langchain_warnings():
            initialize_all_systems(force=False)  # Don't force re-initialization
            # Start health monitoring in the background
            start_health_monitoring()
            logger.info("Health monitoring started from module init.")
            # Start primary health server unconditionally
            start_health_server(port=8502, health_callback=get_health_summary)
            logger.info("Primary health server started unconditionally from module init on port 8502.")
        logger.info("System initialization complete (module level).")
    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")
        # Continue with the app even if initialization fails


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Initialize authentication manager first
    if "auth_manager" not in st.session_state:
        st.session_state.auth_manager = AuthenticationManager()
      # Initialize monitoring dashboard
    if "monitoring_dashboard" not in st.session_state:
        st.session_state.monitoring_dashboard = MonitoringDashboard()
    
    # Initialize security manager
    if "security_manager" not in st.session_state:
        st.session_state.security_manager = SecurityManager()
    
    # Initialize rate limiter
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    
    # Ensure persona is loaded first as other initializations might depend on it
    if "current_persona" not in st.session_state:
        try:
            # Use robust configuration loading with fallback
            config = load_config_robust()
            current_persona_config_value = config.get("current_persona")

            loaded_persona = None
            if isinstance(current_persona_config_value, Persona):
                # If config already stores the Persona object
                loaded_persona = current_persona_config_value
            elif isinstance(current_persona_config_value, str):
                # If config stores the persona name as a string
                loaded_persona = get_persona_by_name(current_persona_config_value)
            else:
                # Log if the type is unexpected, but still try to load Default later
                if current_persona_config_value is not None:
                    st.warning(f"Unexpected type for 'current_persona' in config: {type(current_persona_config_value)}. Attempting to load Default persona.")
        except (RobustConfigError, Exception) as e:
            st.error(f"Configuration error: {e}")
            st.info("Please check your .env file and ensure all required API keys are set.")
            # Use graceful degradation instead of stopping
            loaded_persona = None

        st.session_state.current_persona = loaded_persona

        # Fallback if persona not found or not loaded correctly
        if not st.session_state.current_persona:
            st.session_state.current_persona = get_persona_by_name("Default")
            if not st.session_state.current_persona:
                st.error("CRITICAL: Default persona could not be loaded. Please check persona definitions.")
                # Application might be unstable if Default persona is missing.

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Original block for "conversation_id" also initialized history.
    # Now, just initialize conversation_id. History is initialized later with proper context.
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())  # Generate a unique ID

    # Original "current_persona" block was here, moved to the top.

    # Initialize ConversationHistory after conversation_id and current_persona are set.
    # This replaces and consolidates previous scattered/conflicting history initializations.
    if "history" not in st.session_state:
        current_persona_obj = st.session_state.get("current_persona")  # Use .get for safer access
        if current_persona_obj and hasattr(current_persona_obj, 'name'):  # Check if persona object exists and has a name attribute
            persona_name_for_history = current_persona_obj.name

            st.session_state.history = ConversationHistory(
                conversation_id=st.session_state.conversation_id,
                persona_name=persona_name_for_history
            )
        else:
            # This case implies a critical failure in loading any persona, including Default,
            # or the persona object is malformed.
            st.error("CRITICAL: Conversation history cannot be initialized because no valid persona is loaded. The application may not function correctly.")
            # To prevent further NoneType errors down the line if other parts expect history to exist,
            # we might set it to None or a dummy/default history object if appropriate.
            # For now, st.session_state.history will remain unset or could be set to a non-functional placeholder.
            # Consider: st.session_state.history = None # or a more robust fallback

    if "agent" not in st.session_state:
        # Create agent with current persona
        create_new_agent()

    # The original `if "history" not in st.session_state: st.session_state.history = ConversationHistory(max_history=20)`
    # block is now removed as history is reliably initialized above in the correct order.

    if "memory_manager" not in st.session_state:
        # Get conversation ID from history. History is now guaranteed to be initialized correctly.
        conversation_id_for_memory = st.session_state.history.conversation_id
        # Initialize memory manager with database persistence
        st.session_state.memory_manager = get_memory_manager(
            conversation_id=conversation_id_for_memory,
            max_items=50,
            extraction_enabled=True
        )

    # Analytics dashboard state
    if "show_email_form" not in st.session_state:
        st.session_state.show_email_form = False

    if "voice_manager" not in st.session_state:
        # Initialize voice manager with cache directory
        cache_dir = os.path.join(os.path.dirname(__file__), "audio")
        st.session_state.voice_manager = get_voice_manager(cache_dir=cache_dir)

    if "voice_enabled" not in st.session_state:
        # Default voice setting to enabled
        st.session_state.voice_enabled = True

    if "auto_play_voice" not in st.session_state:
        # Whether to automatically play voice when receiving responses
        st.session_state.auto_play_voice = True

    if "show_memories" not in st.session_state:
        st.session_state.show_memories = False

    if "export_format" not in st.session_state:
        st.session_state.export_format = "html"

    # Initialize autonomous agents (ensure this part is preserved from original file structure)
    if "autonomous_agents" not in st.session_state:
        st.session_state.autonomous_agents = {}

        # Create default instances of each agent type
        agent_types = get_agent_types()
        for agent_type in agent_types:
            agent_id = str(uuid.uuid4()) # uuid is used here
            st.session_state.autonomous_agents[agent_id] = {
                "id": agent_id,
                "type": agent_type,
                "name": f"{agent_type.title()} Agent",
                "instance": create_autonomous_agent(agent_type, agent_id=agent_id),
                "tasks": {},
                "status": AgentStatus.IDLE  # Added default status
            }

    if "selected_agent_id" not in st.session_state:
        # Default to the first agent
        if st.session_state.autonomous_agents:
            st.session_state.selected_agent_id = next(iter(st.session_state.autonomous_agents))
        else:
            st.session_state.selected_agent_id = None

    if "task_results" not in st.session_state:
        # Store the results of completed tasks
        st.session_state.task_results = {}


def change_persona(persona_name: str):
    """Change the current persona.

    Args:
        persona_name: The name of the persona to change to.
    """
    # Get the new persona
    new_persona = get_persona_by_name(persona_name)
    # Update session state
    st.session_state.current_persona = new_persona
    # Create a new agent with the updated persona
    create_new_agent()
    # Add a system message indicating the persona change
    st.session_state.messages.append({
        "role": "system",
        "content": f"Persona alterada para: {new_persona.name}"
    })
    # Show a visible notification (testable fallback)
    try:
        st.toast(f"Persona switched to: {new_persona.name}", icon="🎭")
        st.session_state['last_persona_toast'] = f"Persona switched to: {new_persona.name}"
    except Exception:
        # Fallback for test mode or headless
        st.session_state['last_persona_toast'] = f"Persona switched to: {new_persona.name}"


def create_new_agent():
    """Create a new agent with the current persona."""
    # Load configuration
    config = load_config()

    # Update config with current persona
    config["current_persona"] = st.session_state.current_persona
    config["system_prompt"] = st.session_state.current_persona.get_system_prompt()

    # Get tools
    tools = get_tools()

    # Create agent
    st.session_state.agent = create_agent(config, tools)


def get_persona_image_path(persona_name: str) -> str:
    """Get the path to a persona's image file.

    Args:
        persona_name: The name of the persona.

    Returns:
        The path to the persona's image, or a default if not found.
    """
    # Build the path to the expected image file
    image_dir = os.path.join(os.path.dirname(__file__), "images")

    # For names with spaces, replace with underscores and lowercase
    filename = persona_name.lower().replace(" ", "_") + ".png"
    image_path = os.path.join(image_dir, filename)

    # Check if the file exists, otherwise return a placeholder
    if os.path.exists(image_path):
        return image_path
    else:
        return os.path.join(image_dir, "default.png")


def get_download_link(content: str, filename: str, link_text: str) -> str:
    """Generate a download link for a file.

    Args:
        content: Content to include in the file.
        filename: Name of the file.
        link_text: Text to display for the link.

    Returns:
        HTML for a download link.
    """
    # Encode the content
    b64 = base64.b64encode(content.encode()).decode()

    # Create the download link
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def format_memory_for_display(memory: MemoryItem) -> Dict[str, Any]:
    """Format a memory item for display in the UI.

    Args:
        memory: The memory item to format.

    Returns:
        A dictionary with formatted memory information.
    """
    # Get emoji for category
    category_emoji = {
        "personal": "👤",
        "preference": "❤️",
        "factual": "📝",
        "manual": "✍️",
        "general": "💡"
    }.get(memory.category, "💡")

    # Try to parse timestamp
    try:
        dt = datetime.fromisoformat(memory.timestamp)
        formatted_time = dt.strftime("%Y-%m-%d %H:%M")
    except:
        formatted_time = memory.timestamp

    return {
        "emoji": category_emoji,
        "content": memory.content,
        "timestamp": formatted_time,
        "importance": memory.importance,
        "category": memory.category,
        "access_count": memory.access_count
    }


def create_new_autonomous_agent(agent_type: str):
    """Create a new instance of an autonomous agent.

    Args:
        agent_type: Type of agent to create.

    Returns:
        Agent ID of the newly created agent.
    """
    agent_id = str(uuid.uuid4())
    try:
        agent_instance = create_autonomous_agent(agent_type, agent_id=agent_id)
        st.session_state.autonomous_agents[agent_id] = {
            "id": agent_id,
            "type": agent_type,
            "name": f"{agent_type.title()} Agent",
            "instance": agent_instance,
            "tasks": {}
        }
        st.success(f"Agent '{agent_type}' criado com sucesso!")
        return agent_id
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None


def delete_autonomous_agent(agent_id: str):
    """Delete an autonomous agent.

    Args:
        agent_id: ID of the agent to delete.
    """
    try:
        if agent_id in st.session_state.autonomous_agents:
            # Clean up any task results
            agent_tasks = st.session_state.autonomous_agents[agent_id]["tasks"]
            for task_id in agent_tasks:
                if task_id in st.session_state.task_results:
                    del st.session_state.task_results[task_id]

            # Remove the agent
            del st.session_state.autonomous_agents[agent_id]

            # If the selected agent was deleted, select another one
            if st.session_state.selected_agent_id == agent_id:
                if st.session_state.autonomous_agents:
                    st.session_state.selected_agent_id = next(iter(st.session_state.autonomous_agents))
                else:
                    st.session_state.selected_agent_id = None
            st.success("Agent deleted successfully.")
        else:
            st.warning("Agent not found.")
    except Exception as e:
        st.error(f"Erro ao deletar agente: {str(e)}")


def create_task(agent_id: str, task_type: str, parameters: Dict[str, Any]):
    """Create a new task for an autonomous agent.

    Args:
        agent_id: ID of the agent to execute the task.
        task_type: Type of task to execute.
        parameters: Parameters for the task.

    Returns:
        Task ID of the newly created task, or None if creation failed.
    """
    if agent_id not in st.session_state.autonomous_agents:
        st.error(f"Agent {agent_id} not found")
        return None

    agent_data = st.session_state.autonomous_agents[agent_id]
    agent_instance = agent_data["instance"]

    # Create task configuration
    task_config = TaskConfig(task_type=task_type, parameters=parameters)

    # Check if the agent can handle this task
    if not agent_instance.can_handle_task(task_config):
        st.error(f"Agent {agent_data['name']} cannot handle task type {task_type}")
        return None

    # Create and register the task
    task_id = agent_instance.add_task(task_config)

    # Add to local tasks dict for tracking
    agent_data["tasks"][task_id] = {
        "task_id": task_id,
        "task_type": task_type,
        "parameters": parameters,
        "status": AgentStatus.PENDING,
        "created_at": datetime.now().isoformat()
    }

    return task_id


def execute_task(agent_id: str, task_id: str):
    """Execute a task on an autonomous agent.

    Args:
        agent_id: ID of the agent that owns the task.
        task_id: ID of the task to execute.

    Returns:
        True if execution was successful, False otherwise.
    """
    if agent_id not in st.session_state.autonomous_agents:
        st.error(f"Agent {agent_id} not found")
        return False

    agent_data = st.session_state.autonomous_agents[agent_id]
    agent_instance = agent_data["instance"]

    if task_id not in agent_data["tasks"]:
        st.error(f"Task {task_id} not found for agent {agent_data['name']}")
        return False

    try:
        # Update task status
        agent_data["tasks"][task_id]["status"] = AgentStatus.RUNNING
        agent_data["tasks"][task_id]["started_at"] = datetime.now().isoformat()

        # Execute the task
        result = agent_instance.execute_task_by_id(task_id)

        # Store the result
        st.session_state.task_results[task_id] = result

        # Update task status
        agent_data["tasks"][task_id]["status"] = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
        agent_data["tasks"][task_id]["completed_at"] = datetime.now().isoformat()

        return result.success
    except Exception as e:
        # Update task status
        agent_data["tasks"][task_id]["status"] = AgentStatus.FAILED
        agent_data["tasks"][task_id]["error"] = str(e)
        agent_data["tasks"][task_id]["completed_at"] = datetime.now().isoformat()

        st.error(f"Error executing task: {str(e)}")
        return False


def cancel_task(agent_id: str, task_id: str):
    """Cancel a pending or running task.

    Args:
        agent_id: ID of the agent that owns the task.
        task_id: ID of the task to cancel.

    Returns:
        True if cancellation was successful, False otherwise.
    """
    if agent_id not in st.session_state.autonomous_agents:
        st.error(f"Agent {agent_id} not found")
        return False

    agent_data = st.session_state.autonomous_agents[agent_id]

    if task_id not in agent_data["tasks"]:
        st.error(f"Task {task_id} not found for agent {agent_data['name']}")
        return False

    # Cancel the task
    agent_data["tasks"][task_id]["status"] = AgentStatus.CANCELLED
    agent_data["tasks"][task_id]["completed_at"] = datetime.now().isoformat()

    return True


def delete_task(agent_id: str, task_id: str):
    """Delete a task from an agent.

    Args:
        agent_id: ID of the agent that owns the task.
        task_id: ID of the task to delete.

    Returns:
        True if deletion was successful, False otherwise.
    """
    if agent_id not in st.session_state.autonomous_agents:
        st.error(f"Agent {agent_id} not found")
        return False

    agent_data = st.session_state.autonomous_agents[agent_id]

    if task_id not in agent_data["tasks"]:
        st.error(f"Task {task_id} not found for agent {agent_data['name']}")
        return False

    # Remove the task
    del agent_data["tasks"][task_id]    # Remove from results if present
    if task_id in st.session_state.task_results:
        del st.session_state.task_results[task_id]

    return True


def display_login_interface():
    """Display the login interface."""
    st.set_page_config(page_title="LangGraph 101 - Login", layout="centered")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🤖 LangGraph 101")
        st.subheader("AI Agent Platform")
        st.markdown("---")
          # Login form
        with st.form("login_form"):
            st.markdown("### 🔐 Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                # Validate inputs
                username_validation = InputValidator.validate_input(username, 'username')
                password_validation = InputValidator.validate_input(password, 'general')
                
                if not username_validation['is_valid']:
                    st.error(f"Invalid username: {', '.join(username_validation['errors'])}")
                elif not password_validation['is_valid']:
                    st.error(f"Invalid password: {', '.join(password_validation['errors'])}")
                else:
                    # Check for account lockout
                    if st.session_state.security_manager.is_account_locked(username):
                        remaining = st.session_state.security_manager.get_lockout_remaining(username)
                        st.error(f"🔒 Account locked due to multiple failed attempts. Try again in {remaining//60}m {remaining%60}s")
                    elif st.session_state.auth_manager.authenticate_user(username, password):
                        st.success("✅ Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                        if st.session_state.security_manager.is_account_locked(username):
                            st.warning("⚠️ Multiple failed attempts detected. Account may be locked after more failures.")
        
        st.markdown("---")
        
        # Demo credentials info
        with st.expander("🔍 Demo Credentials", expanded=False):
            st.info("""
            **Demo Accounts:**
            - **Admin**: username=`admin`, password=`admin123`
            - **User**: username=`demo`, password=`demo123`
            
            Admin accounts have access to all features including system monitoring.
            """)
        
        # Security info
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; color: #666;'>
        <small>🔒 Secure authentication with JWT tokens and role-based access control</small>
        </div>
        """, unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar with persona selection, clear conversation, and help."""
    import streamlit as st
    st.sidebar.title("Don Corleone AI")
    # Persona selection
    personas = get_all_personas()
    persona_names = [p.name for p in personas]
    current_persona = st.session_state.get("current_persona")
    if current_persona:
        default_idx = persona_names.index(current_persona.name) if current_persona.name in persona_names else 0
    else:
        default_idx = 0
    selected = st.sidebar.selectbox("Escolha a persona", persona_names, index=default_idx)
    if current_persona is None or selected != current_persona.name:
        change_persona(selected)
    # Clear conversation
    if st.sidebar.button("Limpar Conversa"):
        st.session_state.messages = []
        st.session_state.history.clear()
        st.success("Conversa limpa.")
        st.rerun()
    # Quick help
    st.sidebar.markdown("""
    ### ℹ️ Ajuda Rápida
    - Use o menu para mudar de persona.
    - Clique em 'Limpar Conversa' para reiniciar.
    - Veja dicas e exemplos no painel principal.
    - [Documentação](WEB_INTERFACE.md)
    """)
    # Feedback link
    st.sidebar.markdown("[Deixe seu feedback na aba Analytics ➡️](#analytics)")


def display_onboarding():
    """Display onboarding and help for new users."""
    with st.sidebar.expander("🧑‍💻 Quick Start & Help", expanded=True):
        st.markdown("""
        **Welcome to LangGraph 101!**

        - **Step 1:** Select or create an agent (e.g., Don Corleone, Sherlock Holmes).
        - **Step 2:** Enter your topic or task (e.g., "5-min video on password security").
        - **Step 3:** Use the dashboard tabs to view analytics, system status, and error tracking.
        - **Tips:**
            - Hover over buttons for tooltips.
            - Use the Analytics tab for AI-powered insights.
            - Check the System Status tab for health info.
        - **Need help?** See the [README](README.md) or [WEB_INTERFACE.md].
        """)
        st.info("For best results, keep your API keys up to date in the .env file.")


def display_dashboard():
    """Render the main dashboard content."""
    # from content_dashboard import render_content_creation_dashboard # Import moved to top
    render_content_creation_dashboard()


def display_analytics_dashboard():
    """Display the analytics dashboard for monitoring system performance and usage."""
    # Create tabs for different analytics views
    analytics_tab, system_tab, error_tab, email_tab = st.tabs(["Analytics", "System Status", "Error Tracking", "Notifications"])

    # Analytics dashboard
    with analytics_tab:
        from analytics_dashboard import render_analytics_dashboard, render_error_anomaly_detection
        # Add error handling for missing or invalid data
        try:
            render_analytics_dashboard()
        except Exception as e:
            st.error(f"Failed to load analytics dashboard: {e}")

        try:
            render_error_anomaly_detection()
        except Exception as e:
            st.error(f"Failed to load error anomaly detection: {e}")

    # System status
    with system_tab:
        st.header("System Status")

        # Get current system status
        try:
            status = check_system_status()  # Changed get_system_status to check_system_status

            # Display overall health
            health = status.get('overall_health', 'unknown')
            if health == 'ok':
                st.success("✅ All systems operational")
            elif health == 'warning':
                st.warning("⚠️ System has warnings - check health checks")
            elif health == 'critical':
                st.error("🚨 System has critical issues - immediate attention required")
            else:
                st.info("❓ System health status unknown")

            # Display health checks
            st.subheader("Health Checks")
            health_checks = status.get('health_checks', {})

            for check_name, check_info in health_checks.items():
                check_status = check_info.get('status', 'unknown')
                if check_status == 'ok':
                    st.success(f"✅ {check_name}: {check_info.get('message', '')}")
                elif check_status == 'warning':
                    st.warning(f"⚠️ {check_name}: {check_info.get('message', '')}")
                elif check_status == 'critical':
                    st.error(f"🚨 {check_name}: {check_info.get('message', '')}")
                else:
                    st.info(f"❓ {check_name}: {check_info.get('message', '')}")

            # Display system uptime and resources
            st.subheader("System Resources")

            # Check if psutil info is available
            if 'cpu_percent' in status['system_info']:
                col1, col2, col3 = st.columns(3)

                with col1:
                    mem_total = status["system_info"]["memory_total_gb"]
                    mem_avail = status["system_info"]["memory_available_gb"]
                    mem_used = mem_total - mem_avail
                    mem_percent = status["system_info"]["memory_percent"] / 100

                    st.metric("Memory Usage", f"{mem_used:.1f} GB / {mem_total:.1f} GB")
                    st.progress(min(mem_percent, 1.0))

                with col2:
                    disk_total = status["system_info"]["disk_usage"]["total_gb"]
                    disk_free = status["system_info"]["disk_usage"]["free_gb"]
                    disk_used = disk_total - disk_free
                    disk_percent = status["system_info"]["disk_usage"]["percent"] / 100

                    st.metric("Disk Usage", f"{disk_used:.1f} GB / {disk_total:.1f} GB")
                    st.progress(min(disk_percent, 1.0))

                with col3:
                    st.metric("CPU Usage", f"{status['system_info']['cpu_percent']}%")
                    st.progress(min(status['system_info']['cpu_percent'] / 100, 1.0))
                    st.metric("CPU Cores", status["system_info"]["cpu_count"])

            # System info
            st.subheader("System Information")
            st.write(f"**Platform:** {status['system_info']['platform']}")
            st.write(f"**Python:** {status['system_info']['python_version']}")
            st.write(f"**Hostname:** {status['system_info']['hostname']}")
            st.write(f"**Timestamp:** {status['system_info']['timestamp']}")

            # Display system components status
            st.subheader("System Components")

            systems = status["systems"]
            for system_name, system_info in systems.items():
                component_status = system_info.get("status", "unknown")
                if component_status == "active":
                    icon = "✅"
                elif component_status == "unconfigured":
                    icon = "⚠️"
                else:
                    icon = "❌"

                st.write(f"{icon} **{system_name.replace('_', ' ').title()}**: {system_info['status']}")

                # Add details for each component
                if system_name == "error_handling":
                    st.write(f"  • Total error logs: {system_info.get('error_logs_total', 0)}")
                    st.write(f"  • Recent errors (24h): {system_info.get('error_logs_recent', 0)}")
                elif system_name == "analytics":
                    st.write(f"  • Analytics files: {system_info.get('data_files', 0)}")
                elif system_name == "content_creation":
                    st.write(f"  • Output files: {system_info.get('output_files', 0)}")
                elif system_name == "resilient_storage":
                    st.write(f"  • Last backup: {system_info.get('last_backup', 'Never')}")
                elif system_name == "email_notification":
                    configured = system_info.get('configured', False)
                    st.write(f"  • Configuration: {'Configured' if configured else 'Not configured'}")

            # System maintenance options
            st.subheader("Maintenance")

            col1, col2 = st.columns(2)

            with col1:
                # Option to force a backup
                if st.button("Create System Backup"):
                    from resilient_storage import get_storage

                    with st.spinner("Creating backup..."):
                        storage = get_storage()
                        success = storage.create_backup()

                        if success:
                            st.success("Backup created successfully!")
                        else:
                            st.error("Failed to create backup")

            with col2:
                # Option to send test notification
                if st.button("Send Test Notification"):
                    from error_notification import get_notifier

                    with st.spinner("Sending test notification..."):
                        notifier = get_notifier()
                        success = notifier.send_test_notification()

                        if success:
                            st.success("Test notification sent successfully!")
                        else:
                            st.error("Failed to send test notification. Check email configuration.")

        except Exception as e:
            st.error(f"Error retrieving system status: {str(e)}")

    # Error Tracking
    with error_tab:
        st.header("Error Tracking")

        from analytics_dashboard import load_analytics_data, ERROR_TRACKING_FILE
        from error_handling import ErrorCategory

        try:
            # Load error data
            error_data = load_analytics_data(ERROR_TRACKING_FILE)

            if not error_data:
                st.info("No error data available")
            else:
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(error_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)

                # Summary metrics
                st.subheader("Error Summary")
                total_errors = len(df)

                # Recent errors (last 24 hours)
                recent_df = df[df['timestamp'] > (datetime.now() - timedelta(days=1))]
                recent_errors = len(recent_df)

                # Critical errors
                critical_cats = [
                    ErrorCategory.AUTHENTICATION_ERROR.value,
                    ErrorCategory.SERVER_API_ERROR.value,
                    ErrorCategory.TIMEOUT_ERROR.value,
                    ErrorCategory.MEMORY_ERROR.value
                ]
                critical_df = df[df['category'].isin(critical_cats)]
                critical_errors = len(critical_df)

                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Errors", total_errors)
                with col2:
                    st.metric("Recent Errors (24h)", recent_errors)
                with col3:
                    st.metric("Critical Errors", critical_errors)

                # Error distribution by category
                st.subheader("Error Distribution")
                category_counts = df['category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']

                fig = px.bar(category_counts, x='Category', y='Count',
                             title='Errors by Category',
                             color='Category')
                st.plotly_chart(fig, use_container_width=True)

                # Error timeline
                st.subheader("Error Timeline")
                df_timeline = df.copy()
                df_timeline['date'] = df_timeline['timestamp'].dt.date
                timeline_data = df_timeline.groupby('date').size().reset_index()
                timeline_data.columns = ['Date', 'Count']

                fig = px.line(timeline_data, x='Date', y='Count',
                              title='Errors Over Time')
                st.plotly_chart(fig, use_container_width=True)

                # Recent errors table
                st.subheader("Recent Errors")
                st.dataframe(
                    df[['timestamp', 'category', 'message', 'source']].head(10),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error loading error tracking data: {str(e)}")

    # Email Notifications
    with email_tab:
        st.header("Email Notifications")

        from analytics_dashboard import load_analytics_data, NOTIFICATION_HISTORY_FILE
        from error_notification import get_notifier

        # Email configuration form
        st.subheader("Email Configuration")

        notifier = get_notifier()
        email_configured = bool(notifier.email_config and notifier.email_config.get('recipients'))

        if email_configured:
            st.success("✅ Email notifications are configured")
            st.write(f"**SMTP Server:** {notifier.email_config.get('smtp_server', '')}")
            st.write(f"**Sender:** {notifier.email_config.get('sender', '')}")
            st.write(f"**Recipients:** {', '.join(notifier.email_config.get('recipients', []))}")

            # Option to update configuration
            if st.button("Update Configuration"):
                st.session_state.show_email_form = True
        else:
            st.warning("⚠️ Email notifications are not configured")
            st.session_state.show_email_form = True

        # Show configuration form if needed
        if st.session_state.get('show_email_form', False):
            with st.form("email_config_form"):
                st.write("Configure Email Notifications")

                smtp_server = st.text_input("SMTP Server", value=notifier.email_config.get('smtp_server', ''))
                smtp_port = st.number_input("SMTP Port", value=notifier.email_config.get('smtp_port', 587), min_value=1, max_value=65535)
                use_tls = st.checkbox("Use TLS", value=notifier.email_config.get('use_tls', True))

                smtp_username = st.text_input("SMTP Username", value=notifier.email_config.get('username', ''))
                smtp_password = st.text_input("SMTP Password", type="password", value="")

                sender = st.text_input("Sender Email", value=notifier.email_config.get('sender', ''))
                recipients_str = st.text_input("Recipients (comma-separated)",
                                              value=",".join(notifier.email_config.get('recipients', [])))

                check_interval = st.number_input(
                    "Check Interval (seconds)",
                    value=notifier.check_interval,
                    min_value=60,
                    max_value=86400
                )

                submitted = st.form_submit_button("Save Configuration")

                if submitted:
                    try:
                        # Parse recipients
                        recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

                        # Use existing password if not provided new one
                        if not smtp_password and notifier.email_config.get('password'):
                            smtp_password = notifier.email_config.get('password')

                        # Update configuration
                        from error_notification import setup_error_monitoring

                        setup_error_monitoring(
                            smtp_server=smtp_server,
                            smtp_port=int(smtp_port),
                            username=smtp_username,
                            password=smtp_password,
                            sender=sender,
                            recipients=recipients,
                            check_interval_seconds=int(check_interval),
                            start=True,
                            use_env_config=False
                        )

                        st.success("Email configuration saved")
                        st.session_state.show_email_form = False
                    except Exception as e:
                        st.error(f"Failed to save configuration: {str(e)}")

        # Notification history
        st.subheader("Notification History")

        try:
            # Load notification history
            history_data = load_analytics_data(NOTIFICATION_HISTORY_FILE)

            if not history_data:
                st.info("No notification history available")
            else:
                # Convert to DataFrame
                df = pd.DataFrame(history_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)

                # Display history
                st.dataframe(
                    df[['timestamp', 'subject', 'recipients', 'host']],
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error loading notification history: {str(e)}")


def display_agents_tab():
    """Render the autonomous agents management tab."""
    st.header("🤖 Autonomous Agents Management")
    st.write("Manage and interact with autonomous agents.")

    st.subheader("Create New Agent")
    agent_types = get_agent_types()
    if not agent_types:
        st.warning("No agent types available. Please check agent configuration.")
    else:
        selected_agent_type = st.selectbox("Select Agent Type:", list(agent_types.keys()), key="agent_type_selectbox")
        if st.button("Create New Agent", key="create_agent_button"):
            if selected_agent_type:
                create_new_autonomous_agent(selected_agent_type)
                st.rerun()

    st.subheader("Existing Agents")
    if not st.session_state.autonomous_agents:
        st.info("No autonomous agents created yet.")
    else:
        agents_copy = dict(st.session_state.autonomous_agents) # Iterate over a copy for safe deletion
        for agent_id, agent_info in agents_copy.items():
            with st.expander(f"Agent ID: {agent_id} (Type: {agent_info['type']}, Status: {agent_info['status'].value})"):
                st.write(f"Agent Details: {agent_info.get('agent', 'N/A')}") # Display agent object or placeholder

                # Placeholder for task management for this agent
                # task_description = st.text_input("New Task Description:", key=f"task_desc_{agent_id}")
                # if st.button("Create Task", key=f"create_task_button_{agent_id}"):
                #     if task_description:
                #         # Simplified task creation - actual implementation might vary
                #         task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}
                #         # Assuming a simple task structure for now
                #         st.session_state.autonomous_agents[agent_id]['tasks'][task_id] = {
                #             "description": task_description, "status": "PENDING"
                #         }
                #         st.success(f"Task '{task_description}' created for agent {agent_id}.")
                #         st.rerun()
                #     else:
                #         st.warning("Task description cannot be empty.")

                # Display tasks for the agent
                # if agent_info['tasks']:
                #     st.write("Tasks:")
                #     for task_id, task_details in agent_info['tasks'].items():
                #         st.write(f"- {task_details['description']} (Status: {task_details['status']})")
                # else:
                #     st.write("No tasks for this agent.")

                if st.button("Delete Agent", key=f"delete_agent_button_{agent_id}"):
                    delete_autonomous_agent(agent_id) # This function is already in streamlit_app.py
                    st.rerun()

    # Ensure all agents have a default status if 'status' is missing
    for agent_id, agent_info in st.session_state.autonomous_agents.items():
        if 'status' not in agent_info:            agent_info['status'] = AgentStatus.IDLE


@require_authentication
def display_monitoring_dashboard():
    """Display advanced performance monitoring dashboard."""
    st.header("📊 Performance Monitoring Dashboard")
    
    # Role-based access - only admins can see detailed metrics
    current_user = st.session_state.auth_manager.get_current_user()
    if current_user and current_user.get('role') == 'admin':
        st.session_state.monitoring_dashboard.render_dashboard()
    else:
        st.warning("🔒 Admin access required for detailed performance monitoring")
        # Show limited metrics for regular users
        st.info("Contact your administrator for full monitoring access.")


def display_health_dashboard():
    """Display system health monitoring dashboard."""
    st.header("🏥 System Health Monitor")
    
    # Get current health status
    health_summary = get_health_summary()
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Overall status
        if health_summary["overall_status"] == "healthy":
            st.success("🟢 System Healthy")
        elif health_summary["overall_status"] == "warning":
            st.warning("🟡 System Warning")
        else:
            st.error("🔴 System Critical")
    
    with col2:
        # Python version
        python_info = health_summary.get("python", {})
        if python_info.get("status") == "healthy":
            st.metric("Python Version", python_info.get("version", "Unknown"), 
                     delta="✅ Compatible")
        else:
            st.metric("Python Version", python_info.get("version", "Unknown"), 
                     delta="⚠️ Issues")
    
    with col3:
        # Memory usage
        memory_info = health_summary.get("memory", {})
        if memory_info.get("status") == "healthy":
            usage_pct = memory_info.get("usage_percent", 0)
            st.metric("Memory Usage", f"{usage_pct:.1f}%", 
                     delta="✅ Normal" if usage_pct < 80 else "⚠️ High")
        else:
            st.metric("Memory Usage", "Unknown", delta="⚠️ Check Failed")
    
    with col4:
        # Disk space
        disk_info = health_summary.get("disk", {})
        if disk_info.get("status") == "healthy":
            free_gb = disk_info.get("free_space_gb", 0)
            st.metric("Free Disk Space", f"{free_gb:.1f} GB", 
                     delta="✅ Sufficient" if free_gb > 1 else "⚠️ Low")
        else:
            st.metric("Free Disk Space", "Unknown", delta="⚠️ Check Failed")
    
    # Detailed health information
    st.subheader("📊 Detailed Health Report")
    
    # Configuration status
    with st.expander("🔧 Configuration Status", expanded=False):
        config_health = health_summary.get("configuration", {})
        if config_health.get("status") == "healthy":
            st.success("✅ Configuration loaded successfully")
            st.json(config_health.get("details", {}))
        else:
            st.error("❌ Configuration issues detected")
            st.error(config_health.get("error", "Unknown configuration error"))
    
    # Package dependencies
    with st.expander("📦 Package Dependencies", expanded=False):
        packages_health = health_summary.get("packages", {})
        if packages_health.get("status") == "healthy":
            st.success("✅ All required packages available")
            missing = packages_health.get("missing", [])
            if missing:
                st.warning(f"Missing optional packages: {', '.join(missing)}")
        else:
            st.error("❌ Missing required packages")
            missing = packages_health.get("missing", [])
            if missing:
                st.error(f"Missing: {', '.join(missing)}")
    
    # Network connectivity
    with st.expander("🌐 Network Connectivity", expanded=False):
        network_health = health_summary.get("network", {})
        if network_health.get("status") == "healthy":
            st.success("✅ Network connectivity OK")
            api_status = network_health.get("api_endpoints", {})
            for endpoint, status in api_status.items():
                if status:
                    st.info(f"✅ {endpoint}: Reachable")
                else:
                    st.warning(f"⚠️ {endpoint}: Unreachable")
        else:
            st.error("❌ Network connectivity issues")
            st.error(network_health.get("error", "Unknown network error"))
    
    # Health history chart
    st.subheader("📈 Health Trends")
    
    # Show recent health check results if available
    health_history = health_summary.get("history", [])
    if health_history:
        # Convert to DataFrame for plotting
        df = pd.DataFrame(health_history)
        if not df.empty:
            # Create time series chart
            fig = px.line(df, x='timestamp', y='overall_score', 
                         title='System Health Score Over Time',
                         labels={'overall_score': 'Health Score (0-100)', 
                                'timestamp': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No health history available yet. Health monitoring will collect data over time.")
    
    # Refresh button
    if st.button("🔄 Refresh Health Status"):
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", 
                              key="health_auto_refresh")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()


@require_authentication
def display_enhanced_health_dashboard():
    """Display enhanced system health monitoring dashboard with provider status."""
    st.header("🏥 Enhanced System Health Monitor")
    
    # Get current health status
    health_summary = get_health_summary()
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Overall status
        if health_summary["overall_status"] == "healthy":
            st.success("🟢 System Healthy")
        elif health_summary["overall_status"] == "warning":
            st.warning("🟡 System Warning")
        else:
            st.error("🔴 System Critical")
    
    with col2:
        # Python version
        python_info = health_summary.get("python", {})
        if python_info.get("status") == "healthy":
            st.metric("Python Version", python_info.get("version", "Unknown"), 
                     delta="✅ Compatible")
        else:
            st.metric("Python Version", python_info.get("version", "Unknown"), 
                     delta="⚠️ Issues")
    
    with col3:
        # Memory usage
        memory_info = health_summary.get("memory", {})
        if memory_info.get("status") == "healthy":
            usage_pct = memory_info.get("usage_percent", 0)
            st.metric("Memory Usage", f"{usage_pct:.1f}%", 
                     delta="✅ Normal" if usage_pct < 80 else "⚠️ High")
        else:
            st.metric("Memory Usage", "Unknown", delta="⚠️ Check Failed")
    
    with col4:
        # Disk space
        disk_info = health_summary.get("disk", {})
        if disk_info.get("status") == "healthy":
            free_gb = disk_info.get("free_space_gb", 0)
            st.metric("Free Disk Space", f"{free_gb:.1f} GB", 
                     delta="✅ Sufficient" if free_gb > 1 else "⚠️ Low")
        else:
            st.metric("Free Disk Space", "Unknown", delta="⚠️ Check Failed")
    
    # Provider Health Status Section
    st.subheader("🤖 AI Provider Health Status")
    
    try:
        # Check Google provider health
        from ai_providers.google_provider import GoogleProvider
        from config_robust import load_config_robust
        
        config = load_config_robust()
        google_api_key = config.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
        
        if google_api_key:
            provider_cols = st.columns(3)
            
            # Test different Google models
            models_to_test = ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash']
            
            for i, model_id in enumerate(models_to_test):
                with provider_cols[i % 3]:
                    with st.container():
                        st.markdown(f"**Google {model_id}**")
                        
                        # Create a placeholder for health status
                        status_placeholder = st.empty()
                        
                        try:
                            # Initialize provider
                            provider = GoogleProvider(model_id, google_api_key)
                            
                            # Get real-time metrics
                            metrics = provider.get_real_time_metrics()
                            
                            if 'message' not in metrics:
                                # Show metrics
                                st.metric("Requests/min", f"{metrics.get('requests_per_minute', 0):.1f}")
                                st.metric("Avg Response", f"{metrics.get('avg_response_time', 0):.0f}ms")
                                st.metric("Error Rate", f"{metrics.get('error_rate', 0):.1%}")
                                status_placeholder.success("🟢 Healthy")
                            else:
                                status_placeholder.info("📊 Monitoring Unavailable")
                                
                        except Exception as e:
                            status_placeholder.error(f"🔴 Error: {str(e)[:50]}...")
        else:
            st.warning("⚠️ Google API key not configured for provider health checks")
    
    except ImportError as e:
        st.warning("⚠️ Provider health monitoring unavailable - missing dependencies")
    except Exception as e:
        st.error(f"❌ Error in provider health monitoring: {e}")
    
    # Real-time Analytics Integration
    st.subheader("📊 Real-time System Metrics")
    
    try:
        from analytics.real_time_analytics import RealTimeAnalytics
        real_time_analytics = RealTimeAnalytics()
        current_metrics = real_time_analytics.get_current_metrics()
        
        # Display real-time metrics
        rt_col1, rt_col2, rt_col3, rt_col4 = st.columns(4)
        
        with rt_col1:
            st.metric("Active Users", getattr(current_metrics, 'active_users', 0))
        
        with rt_col2:
            st.metric("API Calls/min", f"{getattr(current_metrics, 'requests_per_minute', 0):.1f}")
        
        with rt_col3:
            st.metric("Avg Response", f"{getattr(current_metrics, 'avg_response_time', 0):.0f}ms")
        
        with rt_col4:
            st.metric("Error Rate", f"{getattr(current_metrics, 'error_rate', 0):.1%}")
    
    except ImportError:
        st.info("📊 Real-time analytics not available")
    except Exception as e:
        st.warning(f"⚠️ Real-time metrics error: {e}")
    
    # Continue with existing health dashboard content
    display_health_dashboard_content(health_summary)

def display_health_dashboard_content(health_summary):
    """Display the detailed health dashboard content."""
    # Detailed health information
    st.subheader("📊 Detailed Health Report")
    
    # Configuration status
    with st.expander("🔧 Configuration Status", expanded=False):
        config_health = health_summary.get("configuration", {})
        if config_health.get("status") == "healthy":
            st.success("✅ Configuration loaded successfully")
            st.json(config_health.get("details", {}))
        else:
            st.error("❌ Configuration issues detected")
            st.error(config_health.get("error", "Unknown configuration error"))
    
    # Package dependencies
    with st.expander("📦 Package Dependencies", expanded=False):
        packages_health = health_summary.get("packages", {})
        if packages_health.get("status") == "healthy":
            st.success("✅ All required packages available")
            missing = packages_health.get("missing", [])
            if missing:
                st.warning(f"Missing optional packages: {', '.join(missing)}")
        else:
            st.error("❌ Missing required packages")
            missing = packages_health.get("missing", [])
            if missing:
                st.error(f"Missing: {', '.join(missing)}")
    
    # Network connectivity
    with st.expander("🌐 Network Connectivity", expanded=False):
        network_health = health_summary.get("network", {})
        if network_health.get("status") == "healthy":
            st.success("✅ Network connectivity OK")
            api_status = network_health.get("api_endpoints", {})
            for endpoint, status in api_status.items():
                if status:
                    st.info(f"✅ {endpoint}: Reachable")
                else:
                    st.warning(f"⚠️ {endpoint}: Unreachable")
        else:
            st.error("❌ Network connectivity issues")
            st.error(network_health.get("error", "Unknown network error"))
    
    # Health history chart
    st.subheader("📈 Health Trends")
    
    # Show recent health check results if available
    health_history = health_summary.get("history", [])
    if health_history:
        # Convert to DataFrame for plotting
        df = pd.DataFrame(health_history)
        if not df.empty:
            # Create time series chart
            fig = px.line(df, x='timestamp', y='overall_score', 
                         title='System Health Score Over Time',
                         labels={'overall_score': 'Health Score (0-100)', 
                                'timestamp': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No health history available yet. Health monitoring will collect data over time.")
    
    # Refresh button
    if st.button("🔄 Refresh Health Status"):
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", 
                              key="health_auto_refresh")
    if auto_refresh:
        time.sleep(30)
        st.rerun()


@require_authentication
def display_advanced_analytics_dashboard():
    """Display the advanced analytics dashboard with real-time capabilities."""
    try:
        # Import the advanced analytics dashboard
        from analytics.dashboard_components import AdvancedAnalyticsDashboard
        from analytics.analytics_logger import AnalyticsLogger
        from analytics.real_time_analytics import RealTimeAnalytics
        from analytics.user_behavior_analyzer import UserBehaviorAnalyzer
        from analytics.performance_tracker import PerformanceTracker
        from analytics.custom_reports_generator import CustomReportsGenerator
        
        # Initialize analytics components
        analytics_logger = AnalyticsLogger()
        real_time_analytics = RealTimeAnalytics()
        user_behavior_analyzer = UserBehaviorAnalyzer()
        performance_tracker = PerformanceTracker()
        custom_reports_generator = CustomReportsGenerator()
        
        # Create and render the advanced dashboard
        advanced_dashboard = AdvancedAnalyticsDashboard(
            analytics_logger=analytics_logger,
            real_time_analytics=real_time_analytics,
            user_behavior_analyzer=user_behavior_analyzer,
            performance_tracker=performance_tracker,
            custom_reports_generator=custom_reports_generator
        )
        
        advanced_dashboard.render_main_dashboard()
        
    except ImportError as e:
        st.warning("⚠️ Advanced analytics components not available. Falling back to basic analytics.")
        logger.warning(f"Advanced analytics import failed: {e}")
        # Fallback to basic analytics dashboard
        display_analytics_dashboard()
    except Exception as e:
        st.error(f"❌ Error loading advanced analytics dashboard: {e}")
        logger.error(f"Advanced analytics dashboard error: {e}")
        # Fallback to basic analytics dashboard
        display_analytics_dashboard()


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="LangGraph 101 - AI Agent Platform", layout="wide")

    # Initialize session state (including auth manager)
    initialize_session_state()
    
    # Check authentication status
    if not st.session_state.auth_manager.is_authenticated():
        display_login_interface()
        return

    # Show startup banner and deployment readiness
    if "startup_check_completed" not in st.session_state:
        st.info("🚀 **LangGraph 101** - Performing startup checks...")
        
        # Run deployment readiness check
        try:
            from deployment_readiness import run_comprehensive_deployment_check
            
            with st.spinner("Checking system readiness..."):
                deployment_results = run_comprehensive_deployment_check()
                
            summary = deployment_results['summary']
            
            if summary['overall_status'] == 'healthy':
                st.success("✅ **System Ready** - All checks passed!")
            elif summary['overall_status'] == 'warning':
                st.warning(f"⚠️ **System Ready with Warnings** - {summary['checks_warning']} warning(s)")
                with st.expander("View Warnings", expanded=False):
                    for warning in summary['warnings']:
                        st.warning(warning)
            else:
                st.error(f"❌ **System Issues Detected** - {summary['checks_failed']} critical issue(s)")
                with st.expander("View Critical Issues", expanded=True):
                    for issue in summary['critical_issues']:
                        st.error(issue)
                    
                    # Show auto-installation option for missing packages
                    pkg_results = deployment_results['detailed_results'].get('package_dependencies', {})
                    missing_required = pkg_results.get('missing_required', [])
                    if missing_required:
                        st.info("**Auto-Installation Available**")
                        if st.button("📦 Install Missing Packages", key="auto_install_packages"):
                            from deployment_readiness import install_missing_packages
                            with st.spinner("Installing packages..."):
                                install_result = install_missing_packages(missing_required)
                            
                            if install_result['status'] == 'success':
                                st.success("✅ Packages installed successfully! Please refresh the page.")
                                st.balloons()
                            else:
                                st.error(f"❌ Installation failed: {install_result.get('failed', [])}")
            
            st.session_state.startup_check_completed = True
            st.session_state.deployment_results = deployment_results
            
        except ImportError:
            st.warning("⚠️ Deployment readiness check not available - continuing with basic startup")
            st.session_state.startup_check_completed = True
        except Exception as e:
            st.error(f"❌ Startup check failed: {str(e)} - continuing anyway")
            st.session_state.startup_check_completed = True

    # Initialize system (error handling, analytics, etc.)
    # This should be called only once at the start of the application
    if "system_initialized_main_app" not in st.session_state:
        with suppress_langchain_warnings():
            init_result = initialize_all_systems(use_env_vars=True)
            
        if init_result["status"] == "error":
            st.error(f"Critical system initialization failed: {init_result.get('error', 'Unknown error')}")
            # Potentially stop the app or show a maintenance page
            st.session_state["system_initialized_main_app"] = "error"
            return 
        elif init_result["status"] == "already_initialized" and st.session_state.get("system_init_error_reported"):
            st.warning("System reported 'already initialized', but previous attempt had an error: " + str(st.session_state.get("system_init_error_details", "Unknown error")))
        
        st.session_state["system_initialized_main_app"] = True
        st.session_state["system_init_error_reported"] = False # Reset error flag
        logger.info("Main application system initialization complete.")
          # Start health monitoring after successful initialization
        try:
            start_health_monitoring()
            logger.info("Health monitoring started successfully.")
        except Exception as e:
            logger.warning(f"Health monitoring failed to start: {str(e)}")
            
    elif st.session_state["system_initialized_main_app"] == "error":
        st.error("System previously failed to initialize. Please check logs or contact support.")
        return    # Initialize session state
    initialize_session_state()

    # Display onboarding and help in the sidebar
    display_onboarding()

    # Sidebar title and user info
    st.sidebar.title("LangGraph 101 AI")
    
    # User profile section
    current_user = st.session_state.auth_manager.get_current_user()
    if current_user:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Profile")
        st.sidebar.write(f"**Username:** {current_user['username']}")
        st.sidebar.write(f"**Role:** {current_user['role'].title()}")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.auth_manager.logout()
            st.rerun()
    
    # Display rate limit information
    display_rate_limit_info()
    
    # Display sidebar content (persona selector, etc.)
    display_sidebar()
    
    # Display main header with key elements for testing
    st.header("🚀 LangGraph 101 - AI Agent Platform")
    st.caption("Comprehensive AI agent system with analytics, monitoring, and content creation")
    
    # Create tabs for different sections - include monitoring for admins
    if current_user and current_user.get('role') == 'admin':
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💬 Chat", "📊 Analytics", "🤖 Agents", "📋 Dashboard", "🏥 System Health", "⚡ Monitoring"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Chat", "📊 Analytics", "🤖 Agents", "📋 Dashboard", "🏥 System Health"
        ])

    # Chat interface in the first tab
    with tab1:
        render_chat_interface()    # Analytics dashboard in the second tab
    with tab2:
        display_advanced_analytics_dashboard()

    # Agents tab in the third tab
    with tab3:
        display_agents_tab()

    # Dashboard in the fourth tab
    with tab4:
        display_dashboard()

    # System Health monitoring in the fifth tab
    with tab5:
        display_enhanced_health_dashboard()
    
    # Performance monitoring tab for admins only
    if current_user and current_user.get('role') == 'admin':
        with tab6:
            display_monitoring_dashboard()

    # st.title("Test: Streamlit App Reached Main")
    # st.write("If you see this, Streamlit is running and the main function was reached.")

if __name__ == "__main__":
    main()
