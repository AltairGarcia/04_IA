"""
Streamlit UI components for LangGraph 101 project.

This module provides UI functions specifically for Streamlit interfaces.
"""

import streamlit as st
from typing import Optional, Dict, Any
import os
import base64
from ui_base import get_i18n

def display_tooltip(label: str, tooltip: str) -> None:
    """Display a label with a tooltip icon.

    Args:
        label: The label text to display.
        tooltip: The tooltip text to show on hover.
    """
    st.markdown(f"{label} ℹ️", help=tooltip)


def display_user_avatar() -> None:
    """Display a user avatar image in Streamlit."""
    # Path to the default user avatar
    avatar_path = os.path.join(os.path.dirname(__file__), "ui_assets", "default_user.png")

    # Use a default image if the avatar file doesn't exist
    if not os.path.exists(avatar_path):
        st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", width=40)
    else:
        st.image(avatar_path, width=40)


def display_bot_avatar(persona_name: str = "Default") -> None:
    """Display a bot avatar image based on persona.

    Args:
        persona_name: The name of the current persona.
    """
    # Path to the persona avatar
    avatar_path = os.path.join(
        os.path.dirname(__file__),
        "ui_assets",
        f"{persona_name.lower().replace(' ', '_')}.png"
    )

    # Use a default image if the persona avatar doesn't exist
    if not os.path.exists(avatar_path):
        st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=identicon", width=40)
    else:
        st.image(avatar_path, width=40)


def add_vertical_space(n_spaces: int = 1) -> None:
    """Add vertical space to the Streamlit UI.

    Args:
        n_spaces: Number of vertical spaces to add.
    """
    for _ in range(n_spaces):
        st.write("")


def local_css(file_name: str) -> None:
    """Load and apply a local CSS file to Streamlit.

    Args:
        file_name: Path to the CSS file.
    """
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url: str) -> None:
    """Load and apply a remote CSS file to Streamlit.

    Args:
        url: URL of the CSS file.
    """
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def show_loading(message: str = "Processing...") -> None:
    """Display a loading spinner with a message.

    Args:
        message: The message to show during loading.
    """
    with st.spinner(message):
        # Placeholder for spinner with message
        pass


def show_success(message: str) -> None:
    """Show a success message in Streamlit.

    Args:
        message: The success message to display.
    """
    st.success(message)


def show_error(message: str) -> None:
    """Show an error message in Streamlit.

    Args:
        message: The error message to display.
    """
    st.error(message)


def show_info(message: str) -> None:
    """Show an info message in Streamlit.

    Args:
        message: The info message to display.
    """
    st.info(message)


def show_warning(message: str) -> None:
    """Show a warning message in Streamlit.

    Args:
        message: The warning message to display.
    """
    st.warning(message)


def get_download_link(file_path: str, link_text: str = "Download file") -> str:
    """Generate a download link for a file.

    Args:
        file_path: Path to the file to download.
        link_text: The text to show for the download link.

    Returns:
        HTML for the download link.
    """
    if not os.path.exists(file_path):
        return "File not found"

    with open(file_path, "rb") as file:
        file_contents = file.read()

    base64_encoded = base64.b64encode(file_contents).decode()
    file_name = os.path.basename(file_path)
    file_type = "application/octet-stream"

    href = f'<a href="data:{file_type};base64,{base64_encoded}" download="{file_name}">{link_text}</a>'
    return href


def create_tab_ui(tabs: Dict[str, Any]) -> None:
    """Create a tabbed interface in Streamlit.

    Args:
        tabs: Dictionary with tab names as keys and tab content functions as values.
    """
    tab_names = list(tabs.keys())
    selected_tabs = st.tabs(tab_names)

    for i, tab in enumerate(selected_tabs):
        with tab:
            tabs[tab_names[i]]()


def create_responsive_columns(num_columns: int = 2) -> list:
    """Create responsive columns for different screen sizes.

    Args:
        num_columns: Number of columns to create.

    Returns:
        List of column objects.
    """
    return st.columns(num_columns)


def create_card(title: str, content: str, icon: str = "📝") -> None:
    """Create a card-like container with styling.

    Args:
        title: Card title.
        content: Card content.
        icon: Icon to display with the title.
    """
    with st.container():
        st.markdown(
            f"""
            <div style="border:1px solid #ddd; border-radius:8px; padding:15px; margin-bottom:15px;">
                <h3 style="margin-top:0">{icon} {title}</h3>
                <p>{content}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def show_persona_selection(personas: list, current_persona: str) -> Optional[str]:
    """Show a persona selection widget.

    Args:
        personas: List of available personas.
        current_persona: Currently selected persona.

    Returns:
        Selected persona name or None if unchanged.
    """
    selected = st.selectbox("Choose a persona:", personas, index=personas.index(current_persona))

    if selected != current_persona:
        return selected

    return None


def create_action_button(label: str, key: str, help_text: str = "") -> bool:
    """Create a styled action button.

    Args:
        label: Button label.
        key: Unique key for the button.
        help_text: Help text to show on hover.

    Returns:
        True if the button was clicked, False otherwise.
    """
    return st.button(label, key=key, help=help_text)


def display_audio_player(audio_file_path: str) -> None:
    """Display an audio player for voice responses.

    Args:
        audio_file_path: Path to the audio file.
    """
    if os.path.exists(audio_file_path):
        st.audio(audio_file_path)
    else:
        st.warning("Audio file not found")
