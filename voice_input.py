"""
Voice Input module for LangGraph 101 project.

This module handles speech recognition for the Don Corleone AI, allowing
users to interact with the AI through voice commands.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import voice-related dependencies with graceful fallback
VOICE_DEPENDENCIES_AVAILABLE = True
MISSING_DEPENDENCIES = []

try:
    import speech_recognition as sr
except ImportError as e:
    VOICE_DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCIES.append("speech_recognition")
    logger.warning(f"speech_recognition not available: {e}")
    logger.info("Voice input will be disabled. To enable voice features, install: pip install SpeechRecognition")
except Exception as e:
    # Handle Python 3.13 compatibility issues (missing aifc module)
    VOICE_DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCIES.append("speech_recognition (Python 3.13 compatibility)")
    logger.warning(f"speech_recognition has compatibility issues with Python 3.13: {e}")
    logger.info("Voice input disabled due to Python 3.13 compatibility. Using fallback mode.")

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError as e:
    VOICE_DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCIES.append("audio libraries")
    logger.warning(f"Audio libraries not available: {e}")
    logger.info("To enable voice features, install: pip install sounddevice soundfile numpy")

class VoiceInputManager:
    """Class to manage voice input processing."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the voice input manager.

        Args:
            cache_dir: Directory to cache temporary audio files.
        """
        self.voice_available = VOICE_DEPENDENCIES_AVAILABLE
        
        if not self.voice_available:
            logger.warning("Voice input functionality disabled due to missing dependencies: " + 
                         ", ".join(MISSING_DEPENDENCIES))
            self.cache_dir = None
            return
            
        self.recognizer = sr.Recognizer()

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "audio", "input")

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_queue = queue.Queue()

    def record_audio(self, duration: int = 5) -> Optional[str]:
        """Record audio from the microphone and save it to a temporary file.

        Args:
            duration: Duration in seconds to record.

        Returns:
            Path to the recorded audio file, or None if recording failed.
        """
        # Generate a temp file path
        _, temp_path = tempfile.mkstemp(suffix=".wav", dir=self.cache_dir)

        try:
            logger.info(f"Recording audio for {duration} seconds...")

            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels
            )
            sd.wait()  # Wait until recording is finished

            # Save as WAV file
            sf.write(temp_path, recording, self.sample_rate)
            logger.info(f"Audio saved to {temp_path}")

            return temp_path
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None

    def start_recording(self):
        """Start recording audio in a separate thread."""
        if self.recording:
            return

        self.recording = True
        self.audio_data = []

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_thread)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        logger.info("Recording started. Speak now...")

    def stop_recording(self) -> Optional[str]:
        """Stop recording and process the audio.

        Returns:
            Path to the recorded audio file, or None if recording failed.
        """
        if not self.recording:
            return None

        self.recording = False

        # Wait for thread to finish
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)

        # Generate a temp file path
        _, temp_path = tempfile.mkstemp(suffix=".wav", dir=self.cache_dir)

        try:
            # Get all data from queue
            audio_frames = []
            while not self.audio_queue.empty():
                audio_frames.append(self.audio_queue.get_nowait())

            if not audio_frames:
                logger.warning("No audio data recorded")
                return None

            # Combine all frames
            audio_data = np.concatenate(audio_frames)

            # Save as WAV file
            sf.write(temp_path, audio_data, self.sample_rate)
            logger.info(f"Audio saved to {temp_path}")

            return temp_path
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None

    def _record_thread(self):
        """Background thread for continuous recording."""
        try:
            # Open the stream
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback):
                while self.recording:
                    # Keep thread alive while recording
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
            self.recording = False

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio status: {status}")

        # Add audio data to queue
        self.audio_queue.put(indata.copy())

    def speech_to_text(self, audio_path: str) -> Tuple[bool, str]:
        """Convert speech to text using the recognizer.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple containing success flag and either text or error message.
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)

            # Try to recognize the speech
            text = self.recognizer.recognize_google(audio_data)
            return True, text
        except sr.UnknownValueError:
            return False, "Could not understand audio"
        except sr.RequestError as e:
            return False, f"Recognition service error: {e}"
        except Exception as e:
            return False, f"Error: {e}"


# Singleton instance
_voice_input_manager = None

def get_voice_input_manager(cache_dir: Optional[str] = None) -> VoiceInputManager:
    """Get the voice input manager singleton.

    Args:
        cache_dir: Optional directory for audio cache.

    Returns:
        Voice input manager instance.
    """
    global _voice_input_manager

    if _voice_input_manager is None:
        _voice_input_manager = VoiceInputManager(cache_dir)

    return _voice_input_manager
