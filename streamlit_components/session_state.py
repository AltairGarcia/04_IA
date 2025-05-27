"""
Session state manager for Streamlit.

This module provides a structured approach to managing Streamlit's session state.
"""
import streamlit as st
import uuid
import logging
from typing import Any, Dict, Optional, List, Callable

# Initialize logger
logger = logging.getLogger(__name__)

class SessionStateManager:
    """Manager for Streamlit session state with structured initialization."""

    @staticmethod
    def initialize_if_absent(key: str, initializer: Callable[[], Any]) -> Any:
        """Initialize a session state variable if it doesn't exist.

        Args:
            key: The session state key
            initializer: Function to initialize the value

        Returns:
            The value (either existing or newly initialized)
        """
        if key not in st.session_state:
            try:
                value = initializer()
                st.session_state[key] = value
                logger.debug(f"Initialized session state key: {key}")
            except Exception as e:
                logger.error(f"Error initializing session state key {key}: {str(e)}")
                # Provide a default value or re-raise the exception
                raise

        return st.session_state[key]

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a value from session state with a default fallback.

        Args:
            key: The session state key
            default: Default value if not found

        Returns:
            The value or default if not found
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set a value in session state.

        Args:
            key: The session state key
            value: Value to set
        """
        st.session_state[key] = value

    @staticmethod
    def get_or_create_id(key: str = "conversation_id") -> str:
        """Get an ID from session state or create a new one.

        Args:
            key: The session state key for the ID

        Returns:
            The ID (either existing or newly generated)
        """
        if key not in st.session_state:
            st.session_state[key] = str(uuid.uuid4())
        return st.session_state[key]

    @staticmethod
    def clear(keys: Optional[List[str]] = None) -> None:
        """Clear specific keys or all keys from session state.

        Args:
            keys: List of keys to clear, or None to clear all
        """
        if keys is None:
            # Clear all keys except protected ones
            protected_keys = ["_is_running", "_component_instance_counter"]
            for key in list(st.session_state.keys()):
                if key not in protected_keys:
                    del st.session_state[key]
        else:
            # Clear only specified keys
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]
