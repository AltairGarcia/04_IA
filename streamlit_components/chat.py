"""
Chat interface component for the Streamlit app.

This module provides the chat interface functionality for the Streamlit interface.
"""
import streamlit as st
from typing import Dict, Any, List, Optional
import logging
import time
import os
import base64

from agent import invoke_agent

# Initialize logger
logger = logging.getLogger(__name__)

def render_chat_interface() -> None:
    """Render the chat interface component."""
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add voice playback button for AI messages when voice is enabled
    if st.session_state.voice_enabled:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant" and "audio_path" in message:
                audio_path = message["audio_path"]
                if os.path.exists(audio_path):
                    # Use a unique key for each audio player to avoid conflicts
                    with st.container():
                        st.audio(audio_path, format="audio/mp3")

    # Chat input
    if user_input := st.chat_input("What can I help you with today?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Check if this is an agent command
        if is_agent_command(user_input):
            # Handle agent command
            with st.chat_message("assistant"):
                command_response = process_agent_command(user_input, st.session_state.current_persona)
                st.markdown(command_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": command_response})

        else:
            # Generate AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")

                try:
                    # Get response from agent
                    agent_response = invoke_agent(
                        st.session_state.agent,
                        user_input,
                        st.session_state.history.get_history()  # Convert ConversationHistory to list of messages
                    )

                    # Display the response
                    message_placeholder.markdown(agent_response)

                    # Add to conversation history
                    st.session_state.history.add_user_message(user_input)
                    st.session_state.history.add_ai_message(agent_response)

                    # Generate voice for the response if enabled
                    audio_path = None
                    if st.session_state.voice_enabled:
                        voice_manager = st.session_state.voice_manager
                        audio_path = voice_manager.text_to_speech(  # Changed from generate_speech
                            agent_response,
                            persona_name=st.session_state.current_persona.name
                        )
                        # Add audio path to message for playback
                        if audio_path:
                            st.session_state.messages[-1]["audio_path"] = audio_path

                            # Play the voice automatically if configured
                            if st.session_state.auto_play_voice and audio_path:
                                # st.audio doesn't work with auto-play, so we'll just display the audio element
                                st.audio(audio_path, format="audio/mp3")

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": agent_response,
                        "audio_path": audio_path
                    })

                    # Extract memories from the conversation
                    if st.session_state.memory_manager:
                        st.session_state.memory_manager.extract_memories(
                            user_input,
                            agent_response
                        )

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.markdown(f"❌ {error_msg}")
                    logger.error(error_msg, exc_info=True)

                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}"
                    })

def render_email_form() -> None:
    """Render the email form if requested."""
    if st.session_state.get("show_email_form", False):
        st.subheader("Email Conversation")
        with st.form("email_form"):
            email = st.text_input("Email address:")
            subject = st.text_input("Subject:", value="Conversation with Don Corleone AI")
            format_name = st.selectbox("Format:", list(get_export_formats().keys()))

            submitted = st.form_submit_button("Send Email")

            if submitted and email:
                try:
                    # Get the email_sender module
                    from email_sender import email_conversation

                    # Send the email
                    email_conversation(
                        st.session_state.history,
                        email,
                        subject=subject,
                        format_name=format_name
                    )
                    st.success(f"Email sent to {email}")
                    # Hide the form after submission
                    st.session_state.show_email_form = False
                except Exception as e:
                    st.error(f"Failed to send email: {str(e)}")

            if st.form_submit_button("Cancel"):
                st.session_state.show_email_form = False

def render_memories() -> None:
    """Render the memory panel if enabled."""
    if st.session_state.get("show_memories", False):
        with st.expander("Memories", expanded=True):
            # Get memories from memory manager
            memories = st.session_state.memory_manager.get_all_memories()

            if not memories:
                st.write("No memories extracted yet.")
            else:
                # Display memories in a table
                memory_data = []
                for memory in memories:
                    memory_data.append({
                        "Type": memory.memory_type,
                        "Content": memory.content,
                        "Timestamp": memory.timestamp.strftime("%Y-%m-%d %H:%M")
                    })

                st.dataframe(memory_data)

# Import at the end to avoid circular imports
from agent_commands import process_agent_command, is_agent_command
from export import get_export_formats
