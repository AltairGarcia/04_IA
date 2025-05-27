"""
Voice module for LangGraph 101 project.

This module handles text-to-speech functionality for different personas.
"""

import os
import base64
import tempfile
from typing import Dict, Any, Optional
from gtts import gTTS
from datetime import datetime


class VoiceManager:
    """Manages text-to-speech functionality for personas."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize a new VoiceManager.

        Args:
            cache_dir: Optional directory to cache audio files.
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Voice configurations for different personas
        self.voice_configs = {
            "Don Corleone": {
                "language": "it",  # Italian accent for Don Corleone
                "tld": "com",
                "slow": False
            },
            "Sócrates": {
                "language": "pt",  # Portuguese for a Greek philosopher (closest option)
                "tld": "com.br",
                "slow": True
            },
            "Ada Lovelace": {
                "language": "en",  # English for Ada Lovelace
                "tld": "co.uk",    # British accent
                "slow": False
            },
            "Captain Jack Sparrow": {
                "language": "en",  # English for Jack Sparrow
                "tld": "com",      # American accent
                "slow": False
            },
            "default": {
                "language": "en",
                "tld": "com",
                "slow": False
            }
        }

    def get_voice_config(self, persona_name: str) -> Dict[str, Any]:
        """Get voice configuration for a persona.

        Args:
            persona_name: Name of the persona.

        Returns:
            Voice configuration dictionary.
        """
        return self.voice_configs.get(persona_name, self.voice_configs["default"])

    def text_to_speech(self, text: str, persona_name: str) -> str:
        """Convert text to speech audio file.

        Args:
            text: Text to convert to speech.
            persona_name: Name of the persona.

        Returns:
            Path to the generated audio file.
        """
        voice_config = self.get_voice_config(persona_name)

        # Create gTTS object with persona-specific configuration
        tts = gTTS(
            text=text,
            lang=voice_config["language"],
            tld=voice_config["tld"],
            slow=voice_config["slow"]
        )

        # Save to a temporary file or cached file
        if self.cache_dir:
            # Create a filename based on text and persona
            safe_text = text[:20].replace(" ", "_").replace("/", "")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{persona_name.lower().replace(' ', '_')}_{safe_text}_{timestamp}.mp3"
            filepath = os.path.join(self.cache_dir, filename)
            tts.save(filepath)
            return filepath
        else:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                tts.save(temp.name)
                return temp.name

    def get_audio_html(self, audio_path: str) -> str:
        """Get HTML audio tag for the audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            HTML audio tag.
        """
        audio_tag = f'<audio controls autoplay><source src="file://{audio_path}" type="audio/mpeg">Your browser does not support the audio element.</audio>'
        return audio_tag

    def get_audio_data_url(self, audio_path: str) -> str:
        """Convert audio file to a data URL.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Data URL for the audio.
        """
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        audio_b64 = base64.b64encode(audio_bytes).decode()
        data_url = f"data:audio/mp3;base64,{audio_b64}"
        return data_url

    def cleanup_audio_file(self, audio_path: str) -> None:
        """Delete audio file when no longer needed.

        Args:
            audio_path: Path to the audio file.
        """
        if os.path.exists(audio_path) and not self.cache_dir:
            try:
                os.unlink(audio_path)
            except:
                pass  # Ignore errors in cleanup


def get_voice_manager(cache_dir: Optional[str] = None) -> VoiceManager:
    """Get a VoiceManager instance.

    Args:
        cache_dir: Optional directory to cache audio files.

    Returns:
        A VoiceManager instance.
    """
    # Create voice cache directory if specified
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    return VoiceManager(cache_dir=cache_dir)
