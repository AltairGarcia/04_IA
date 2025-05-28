"""
Configuration management module for LangGraph 101 application.

This module provides configuration loading with robust error handling
and backward compatibility. Now uses the unified configuration system.
"""
import os
import logging
from typing import Dict, Any, Optional

# Import from unified configuration system
from core.config import UnifiedConfig, get_config, ConfigError
from personas import get_default_persona, get_persona_by_name, get_all_personas, Persona

# Configure logging
logger = logging.getLogger(__name__)

# ConfigError is now imported from core.config
# class ConfigError(Exception):
#     """Exception raised for configuration errors."""
#     pass


def load_config() -> Dict[str, Any]:
    """
    Load configuration from unified configuration system.

    Returns:
        Dict containing configuration values.

    Raises:
        ConfigError: If required configuration values are missing.
    """
    try:
        # Use the unified configuration system
        unified_config = get_config()
        
        # Model configuration with defaults
        model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")
        temperature = float(os.getenv("TEMPERATURE", "0.7"))

        # Persona configuration
        persona_name = os.getenv("PERSONA", "Don Corleone")
        try:
            current_persona = get_persona_by_name(persona_name)
        except Exception as e:
            logger.warning(f"Failed to load persona '{persona_name}': {e}. Using default persona.")
            current_persona = get_default_persona()

        # Application settings
        save_history = os.getenv("SAVE_HISTORY", "false").lower() == "true"
        max_history = int(os.getenv("MAX_HISTORY", "10"))
        
        # Build configuration dictionary using unified config
        config = {
            # Core API keys from unified config
            "api_key": unified_config.api.gemini_api_key or unified_config.api.google_api_key,
            "tavily_api_key": unified_config.api.tavily_api_key,
            "model_name": model_name,
            "temperature": temperature,
            "current_persona": current_persona,
            "save_history": save_history,
            "max_history": max_history,
            "system_prompt": get_system_prompt(current_persona),
            
            # Optional API keys from unified config
            "elevenlabs_api_key": unified_config.api.elevenlabs_api_key,
            "dalle_api_key": unified_config.api.openai_api_key,  # Correctly mapped to OpenAI
            "stabilityai_api_key": unified_config.api.stability_api_key,
            "pixabay_api_key": unified_config.api.pixabay_api_key,
            "pexels_api_key": unified_config.api.pexels_api_key,
            "deepgram_api_key": unified_config.api.deepgram_api_key,
            "assemblyai_api_key": unified_config.api.assemblyai_api_key,
            "youtube_data_api_key": unified_config.api.youtube_data_api_key,
            "news_api_key": unified_config.api.news_api_key,
            "openweather_api_key": unified_config.api.openweather_api_key,
            
            # Email and error monitoring (placeholder values for compatibility)
            "smtp_server": None,  # Not implemented in unified config yet
            "smtp_port": 587,
            "smtp_username": None,
            "smtp_password": None,
            "alert_email_from": None,
            "alert_email_to": None,
            "smtp_use_tls": True,
            "error_threshold": 10,
            "error_window_hours": 24,
            
            # Application settings from unified config
            "debug": unified_config.app.debug,
            "log_level": unified_config.logging.level
        }
        
        return config
        
    except Exception as e:
        raise ConfigError(f"Configuration loading failed: {e}")


def get_system_prompt(persona: Optional[Persona] = None) -> str:
    """Get the system prompt for the agent.

    Args:
        persona: Optional persona to use. If None, uses the default or environment-specified persona.

    Returns:
        The system prompt string.
    """
    # Use provided persona or get from environment
    if persona is None:
        # Check if there's a custom prompt in the environment
        custom_prompt = os.getenv("SYSTEM_PROMPT")
        if custom_prompt:
            return custom_prompt

        # Otherwise use the default persona
        persona_name = os.getenv("PERSONA", "Don Corleone")
        try:
            persona = get_persona_by_name(persona_name)
        except Exception as e:
            logger.warning(f"Failed to load persona '{persona_name}': {e}. Using default persona.")
            persona = get_default_persona()

    # Return the persona's system prompt
    return persona.get_system_prompt()


def get_available_personas() -> Dict[str, Dict[str, str]]:
    """Get information about all available personas.

    Returns:
        Dictionary mapping persona names to their information.
    """
    try:
        personas = get_all_personas()
        return {persona.name: persona.get_info() for persona in personas}
    except Exception as e:
        logger.error(f"Failed to load personas: {e}")
        return {}
