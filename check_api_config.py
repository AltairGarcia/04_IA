#!/usr/bin/env python
"""API Configuration Check Script"""

import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_api_configurations():
    # Explicitly load the .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    # Debug log for .env loading
    logger.debug(f"Loading .env file from: {dotenv_path}")
    if not os.path.exists(dotenv_path):
        logger.error(".env file not found at the specified path.")
    else:
        logger.debug(".env file loaded successfully.")

    print('🔍 Testing AI Service Integrations...')

    # Check API keys
    keys_to_check = [
        ('Gemini AI', 'API_KEY', 'GEMINI_API_KEY'),
        ('ElevenLabs TTS', 'ELEVENLABS_API_KEY'),
        ('Pexels Images', 'PEXELS_API_KEY'),
        ('Pixabay Media', 'PIXABAY_API_KEY'),
        ('StabilityAI Images', 'STABILITY_API_KEY'),
        ('DALL-E Images', 'OPENAI_API_KEY'),
        ('AssemblyAI Transcription', 'ASSEMBLYAI_API_KEY'),
        ('Deepgram Transcription', 'DEEPGRAM_API_KEY'),
        ('YouTube Data API', 'YOUTUBE_API_KEY'),
        ('Tavily Search', 'TAVILY_API_KEY')
    ]

    configured_services = []
    missing_services = []

    for service_info in keys_to_check:
        service_name = service_info[0]
        env_vars = service_info[1:] if len(service_info) > 2 else (service_info[1],)

        api_key = None
        found_var = None
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if api_key:
                found_var = env_var
                break

        if api_key:
            configured_services.append(service_name)
            print(f'✅ {service_name}: Configured ({found_var})')
        else:
            missing_services.append(service_name)
            var_list = " or ".join(env_vars)
            print(f'❌ {service_name}: Missing ({var_list})')

    # Debugging API key retrieval
    print(f"STABILITY_API_KEY: {os.getenv('STABILITY_API_KEY', 'Not Found')}")
    print(f"YOUTUBE_API_KEY: {os.getenv('YOUTUBE_API_KEY', 'Not Found')}")

    print(f'\n📊 Summary:')
    print(f'Configured services: {len(configured_services)}/{len(keys_to_check)}')
    print(f'Missing services: {len(missing_services)}')

    if configured_services:
        print(f'\n✅ Available: {" | ".join(configured_services)}')
    if missing_services:
        print(f'\n❌ Missing: {" | ".join(missing_services)}')

    return configured_services, missing_services

if __name__ == "__main__":
    check_api_configurations()
