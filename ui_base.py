"""
UI module for LangGraph 101 project.

This module handles UI elements like colorized output and progress indicators
for both terminal and Streamlit interfaces.
"""

import sys
import time
import json
import os
from typing import Optional, Dict, Any
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Create a base directory for UI-related files
UI_BASE_DIR = os.path.join(os.path.dirname(__file__), "ui_assets")
os.makedirs(UI_BASE_DIR, exist_ok=True)

# Create locales directory
LOCALES_DIR = os.path.join(UI_BASE_DIR, "locales")
os.makedirs(LOCALES_DIR, exist_ok=True)

# I18n class for internationalization
class I18n:
    """Internationalization support for the UI."""

    def __init__(self, locale: str = "en"):
        """Initialize the I18n instance.

        Args:
            locale: The locale to use (e.g., "en", "pt_BR").
        """
        self.locale = locale
        self.strings: Dict[str, str] = {}
        self.load_strings()

    def load_strings(self) -> None:
        """Load the strings for the current locale."""
        try:
            locale_file = os.path.join(LOCALES_DIR, f"{self.locale}.json")

            if not os.path.exists(locale_file):
                # Create default locale file if it doesn't exist
                if self.locale == "en":
                    self._create_default_english_locale()
                elif self.locale == "pt_BR":
                    self._create_default_portuguese_locale()
                else:
                    logger.warning(f"No locale file found for {self.locale}, falling back to English")
                    self.locale = "en"
                    self._create_default_english_locale()

            # Load the locale file
            with open(os.path.join(LOCALES_DIR, f"{self.locale}.json"), "r", encoding="utf-8") as f:
                self.strings = json.load(f)

            logger.info(f"Loaded {len(self.strings)} strings for locale {self.locale}")
        except Exception as e:
            logger.error(f"Error loading strings for locale {self.locale}: {str(e)}")
            self.strings = {}

    def _create_default_english_locale(self) -> None:
        """Create the default English locale file."""
        default_strings = {
            "welcome_title": "=== Don Corleone AI ===",
            "welcome_message": "I'll make you an offer you can't refuse...",
            "exit_command": "Type 'exit' to end the chat.",
            "help_command": "Type 'help' to see available commands.",
            "available_commands": "--- Available Commands ---",
            "exit_desc": "exit       - End the conversation",
            "help_desc": "help       - Show this help message",
            "clear_desc": "clear     - Clear conversation history",
            "quick_suggestions": "Quick suggestions:",
            "error_prefix": "Error: ",
            "thinking": "Thinking",
            "user_prompt": "You"
        }

        try:
            with open(os.path.join(LOCALES_DIR, "en.json"), "w", encoding="utf-8") as f:
                json.dump(default_strings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error creating default English locale file: {str(e)}")

    def _create_default_portuguese_locale(self) -> None:
        """Create the default Portuguese (Brazil) locale file."""
        default_strings = {
            "welcome_title": "=== Don Corleone AI ===",
            "welcome_message": "Farei uma oferta que você não poderá recusar...",
            "exit_command": "Digite 'sair' para encerrar o chat.",
            "help_command": "Digite 'ajuda' para ver comandos disponíveis.",
            "available_commands": "--- Comandos Disponíveis ---",
            "exit_desc": "sair       - Encerra a conversa",
            "help_desc": "ajuda      - Mostra esta mensagem de ajuda",
            "clear_desc": "limpar    - Limpa o histórico da conversa",
            "quick_suggestions": "Sugestões rápidas:",
            "error_prefix": "Erro: ",
            "thinking": "Pensando",
            "user_prompt": "Você"
        }

        try:
            with open(os.path.join(LOCALES_DIR, "pt_BR.json"), "w", encoding="utf-8") as f:
                json.dump(default_strings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error creating default Portuguese locale file: {str(e)}")

    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get a string for the given key.

        Args:
            key: The key to look up
            default: The default value if the key is not found

        Returns:
            The localized string
        """
        return self.strings.get(key, default if default else key)


# Lazy-loaded singleton instance
_i18n = None

def get_i18n(locale: str = "en") -> I18n:
    """Get the I18n instance.

    Args:
        locale: The locale to use

    Returns:
        The I18n instance
    """
    global _i18n
    if _i18n is None or _i18n.locale != locale:
        _i18n = I18n(locale)
    return _i18n


# Create separate modules for terminal and Streamlit UIs
from ui_terminal import *
from ui_streamlit import *
