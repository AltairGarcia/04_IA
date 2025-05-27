"""
Sidebar component for the Streamlit app.

This module provides the sidebar functionality for the Streamlit interface.
"""
import streamlit as st
from typing import Dict, Any, Tuple
import logging

from personas import get_all_personas, get_persona_by_name

# Initialize logger
logger = logging.getLogger(__name__)

def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar with configuration options.

    Returns:
        Dictionary with selected options
    """
    with st.sidebar:
        st.title("Don Corleone AI")

        # Persona selection
        st.subheader("Persona")
        personas = get_all_personas()
        persona_names = [p.name for p in personas]

        current_persona_name = None
        if "current_persona" in st.session_state and st.session_state.current_persona:
            current_persona_name = st.session_state.current_persona.name

        selected_persona = st.selectbox(
            "Choose a persona:",
            persona_names,
            index=persona_names.index(current_persona_name) if current_persona_name in persona_names else 0
        )

        if (not current_persona_name or selected_persona != current_persona_name):
            # User selected a different persona
            st.session_state.current_persona = get_persona_by_name(selected_persona)
            # Signal that we need to create a new agent
            st.session_state.need_new_agent = True

        # Voice settings
        st.subheader("Voice Settings")
        voice_enabled = st.checkbox("Enable voice", value=st.session_state.get("voice_enabled", True))
        if "voice_enabled" not in st.session_state or voice_enabled != st.session_state.voice_enabled:
            st.session_state.voice_enabled = voice_enabled

        if voice_enabled:
            auto_play = st.checkbox("Auto-play voice", value=st.session_state.get("auto_play_voice", True))
            if "auto_play_voice" not in st.session_state or auto_play != st.session_state.auto_play_voice:
                st.session_state.auto_play_voice = auto_play

        # Memory settings
        st.subheader("Memory Settings")
        show_memories = st.checkbox("Show memories", value=st.session_state.get("show_memories", False))
        if "show_memories" not in st.session_state or show_memories != st.session_state.show_memories:
            st.session_state.show_memories = show_memories

        # Export settings
        st.subheader("Export Settings")
        export_formats = get_export_formats()
        selected_format = st.selectbox(
            "Export format:",
            list(export_formats.keys()),
            index=list(export_formats.keys()).index(st.session_state.get("export_format", "html"))
        )
        if "export_format" not in st.session_state or selected_format != st.session_state.export_format:
            st.session_state.export_format = selected_format

        # Export button
        if st.button("Export Conversation"):
            try:
                export_path = export_conversation(
                    st.session_state.history,
                    format_name=st.session_state.export_format
                )
                st.success(f"Conversation exported to: {export_path}")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

        # Email form toggle
        if st.button("Email Conversation"):
            st.session_state.show_email_form = True

        # Show system info at the bottom
        st.sidebar.markdown("---")
        with st.sidebar.expander("System Info"):
            st.write(f"Conversation ID: {st.session_state.conversation_id}")
            st.write(f"Active persona: {st.session_state.current_persona.name}")

            # Show cache stats if available
            if "cache_stats" in st.session_state:
                st.write("Cache Stats:")
                stats = st.session_state.cache_stats
                st.write(f"- Memory cache: {stats.get('memory_size', 0)} items")
                st.write(f"- Hits: {stats.get('hits', 0)}")
                st.write(f"- Misses: {stats.get('misses', 0)}")

    # Return selected options for use elsewhere
    return {
        "persona": st.session_state.current_persona,
        "voice_enabled": st.session_state.voice_enabled,
        "auto_play_voice": getattr(st.session_state, "auto_play_voice", True),
        "show_memories": st.session_state.show_memories,
        "export_format": st.session_state.export_format
    }

# Import at the end to avoid circular imports
from export import export_conversation, get_export_formats
from email_sender import email_conversation
