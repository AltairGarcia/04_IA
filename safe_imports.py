"""
Safe imports module for LangGraph 101 project.
Handles missing dependencies gracefully for Python 3.13 compatibility.
"""

import logging
import warnings
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def safe_import(module_name: str, package_name: str = None, alternative_name: str = None):
    """
    Safely import a module with fallback handling.
    
    Args:
        module_name: Name of the module to import
        package_name: Package name for installation instructions
        alternative_name: Alternative name for the module
    
    Returns:
        Imported module or None if import fails
    """
    try:
        if alternative_name:
            try:
                return __import__(module_name)
            except ImportError:
                return __import__(alternative_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        pkg_name = package_name or module_name
        logger.warning(f"Module '{module_name}' not available: {e}")
        logger.info(f"To install: pip install {pkg_name}")
        return None
    except Exception as e:
        logger.warning(f"Module '{module_name}' failed to load (compatibility issue): {e}")
        return None

# Safe imports for voice functionality
speech_recognition = safe_import("speech_recognition", "SpeechRecognition")
sounddevice = safe_import("sounddevice")
soundfile = safe_import("soundfile") 
numpy_audio = safe_import("numpy")

# Safe imports for AI providers
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logger.warning("Google AI not available. Install with: pip install google-generativeai")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available. Install with: pip install anthropic")

# Safe imports for LangChain
try:
    import langchain
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain")

# Safe imports for other dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not available (optional)")

try:
    import celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.info("Celery not available (optional)")

# Voice functionality status
VOICE_INPUT_AVAILABLE = all([speech_recognition, sounddevice, soundfile, numpy_audio])
if not VOICE_INPUT_AVAILABLE:
    logger.info("Voice input functionality disabled due to missing dependencies or Python 3.13 compatibility issues")

# Export availability flags
__all__ = [
    'VOICE_INPUT_AVAILABLE',
    'GOOGLE_AI_AVAILABLE', 
    'OPENAI_AVAILABLE',
    'ANTHROPIC_AVAILABLE',
    'LANGCHAIN_AVAILABLE',
    'REDIS_AVAILABLE',
    'CELERY_AVAILABLE',
    'safe_import'
]
