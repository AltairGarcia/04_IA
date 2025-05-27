#!/usr/bin/env python
"""
Simple API Testing Script for LangGraph 101 Content Creation System
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("🧪 LangGraph 101 - Simple API Testing")
print("=" * 50)

# Test API key availability
def test_api_keys():
    print("🔑 Checking API Keys...")
    
    keys = {
        'ELEVENLABS_API_KEY': 'ElevenLabs TTS',
        'STABILITY_API_KEY': 'Stability AI',
        'OPENAI_API_KEY': 'DALL-E (OpenAI)',
        'PEXELS_API_KEY': 'Pexels',
        'PIXABAY_API_KEY': 'Pixabay',
        'ASSEMBLYAI_API_KEY': 'AssemblyAI',
        'DEEPGRAM_API_KEY': 'Deepgram',
        'YOUTUBE_API_KEY': 'YouTube Data API',
        'GEMINI_API_KEY': 'Google Gemini'
    }
    
    available = 0
    for key, service in keys.items():
        value = os.getenv(key)
        is_available = bool(value and value.strip())
        status = "✅" if is_available else "❌"
        print(f"{status} {service}: {'Available' if is_available else 'Missing'}")
        if is_available:
            available += 1
    
    print(f"\n📊 {available}/{len(keys)} API keys available")
    return available

# Test imports
def test_imports():
    print("\n📦 Testing Imports...")
    
    try:
        from tools import ElevenLabsTTS
        print("✅ ElevenLabsTTS imported successfully")
    except Exception as e:
        print(f"❌ ElevenLabsTTS import failed: {e}")
    
    try:
        from tools import GeminiAPI
        print("✅ GeminiAPI imported successfully")
    except Exception as e:
        print(f"❌ GeminiAPI import failed: {e}")
    
    try:
        from tools import PexelsAPI, PixabayAPI
        print("✅ Stock media APIs imported successfully")
    except Exception as e:
        print(f"❌ Stock media APIs import failed: {e}")
    
    try:
        from tools import StabilityAIAPI, DalleAPI
        print("✅ Image generation APIs imported successfully")
    except Exception as e:
        print(f"❌ Image generation APIs import failed: {e}")

# Simple API test
def test_gemini_basic():
    print("\n📝 Testing Gemini API (Basic)...")
    
    try:
        from tools import GeminiAPI
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("⏭️ Skipped - No API key")
            return
        
        gemini = GeminiAPI(api_key=api_key)
        result = gemini.generate_content("Say 'Hello from Gemini API test!'")
        
        if result and len(result.strip()) > 0:
            print("✅ Gemini API test successful")
            print(f"📝 Response: {result[:100]}...")
        else:
            print("❌ Gemini API test failed - empty response")
            
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")

def test_elevenlabs_basic():
    print("\n🎤 Testing ElevenLabs API (Basic)...")
    
    try:
        from tools import ElevenLabsTTS
        api_key = os.getenv('ELEVENLABS_API_KEY')
        
        if not api_key:
            print("⏭️ Skipped - No API key")
            return
        
        tts = ElevenLabsTTS(api_key=api_key)
        audio_data = tts.text_to_speech("Hello from ElevenLabs test!")
        
        if audio_data and len(audio_data) > 0:
            print("✅ ElevenLabs API test successful")
            print(f"🎵 Audio data size: {len(audio_data)} bytes")
        else:
            print("❌ ElevenLabs API test failed - no audio data")
            
    except Exception as e:
        print(f"❌ ElevenLabs API test failed: {e}")

def main():
    available_keys = test_api_keys()
    test_imports()
    
    if available_keys > 0:
        test_gemini_basic()
        test_elevenlabs_basic()
    else:
        print("\n⚠️ No API keys available for testing")
    
    print("\n🎯 Summary:")
    print("- If APIs are working, proceed with implementing fallbacks")
    print("- If APIs are failing, check API keys and network connectivity")
    print("- Run the full content dashboard to test integrated functionality")

if __name__ == "__main__":
    main()
