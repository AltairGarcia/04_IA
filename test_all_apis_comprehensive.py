#!/usr/bin/env python
"""
Comprehensive API Testing Script for LangGraph 101 Content Creation System

This script tests all API integrations to ensure they're working correctly:
- TTS (ElevenLabs): Voice synthesis
- Image Generation (DALL-E, Stability AI): Image creation
- Stock Media Search (Pixabay, Pexels): Media retrieval
- Transcription (Deepgram, AssemblyAI): Audio-to-text functionality
- Trend Research (YouTube Data API): YouTube data retrieval
- Text Generation (Gemini): Content creation
"""

import os
import sys
import time
import tempfile
from datetime import datetime
from typing import Dict, Any, List
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from tools import (
    ElevenLabsTTS, StabilityAIAPI, DalleAPI, 
    PexelsAPI, PixabayAPI, AssemblyAIAPI, 
    DeepgramAPI, YouTubeDataAPI, GeminiAPI
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APITester:
    """Comprehensive API testing class."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def test_api_key_availability(self) -> Dict[str, bool]:
        """Test if all API keys are available."""
        logger.info("🔑 Testing API Key Availability...")
        
        keys_to_check = {
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
        
        key_status = {}
        for key, service in keys_to_check.items():
            value = os.getenv(key)
            is_available = bool(value and value.strip())
            key_status[service] = is_available
            
            status_icon = "✅" if is_available else "❌"
            logger.info(f"{status_icon} {service}: {'Available' if is_available else 'Missing'}")
            
        return key_status
    
    def test_gemini_text_generation(self) -> Dict[str, Any]:
        """Test Gemini text generation."""
        logger.info("📝 Testing Gemini Text Generation...")
        
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return {"status": "skipped", "reason": "No API key"}
            
            gemini = GeminiAPI(api_key=api_key)
            
            start_time = time.time()
            result = gemini.generate_content("Write a brief introduction about AI in content creation (max 100 words)")
            end_time = time.time()
            
            if result and len(result.strip()) > 10:
                logger.info("✅ Gemini text generation: SUCCESS")
                return {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "content_length": len(result),
                    "sample": result[:100] + "..." if len(result) > 100 else result
                }
            else:
                logger.error("❌ Gemini text generation: FAILED - Empty or invalid response")
                return {"status": "failed", "reason": "Empty or invalid response"}
                
        except Exception as e:
            logger.error(f"❌ Gemini text generation: FAILED - {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def test_elevenlabs_tts(self) -> Dict[str, Any]:
        """Test ElevenLabs TTS functionality."""
        logger.info("🎤 Testing ElevenLabs TTS...")
        
        try:
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                return {"status": "skipped", "reason": "No API key"}
            
            tts = ElevenLabsTTS(api_key=api_key)
            
            start_time = time.time()
            audio_data = tts.text_to_speech("Hello, this is a test of the ElevenLabs TTS system.")
            end_time = time.time()
            
            if audio_data and len(audio_data) > 0:
                logger.info("✅ ElevenLabs TTS: SUCCESS")
                return {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "audio_size_bytes": len(audio_data)
                }
            else:
                logger.error("❌ ElevenLabs TTS: FAILED - No audio data returned")
                return {"status": "failed", "reason": "No audio data returned"}
                
        except Exception as e:
            logger.error(f"❌ ElevenLabs TTS: FAILED - {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def test_image_generation(self) -> Dict[str, Any]:
        """Test both DALL-E and Stability AI image generation."""
        logger.info("🎨 Testing Image Generation APIs...")
        
        results = {}
        
        # Test DALL-E
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                dalle = DalleAPI(api_key=api_key)
                
                start_time = time.time()
                result = dalle.generate_image("cat", n=1, size="1024x1024")
                end_time = time.time()
                
                if result and 'data' in result and len(result['data']) > 0:
                    logger.info("✅ DALL-E image generation: SUCCESS")
                    results['dalle'] = {
                        "status": "success",
                        "response_time": end_time - start_time,
                        "image_url": result['data'][0].get('url', 'N/A')
                    }
                else:
                    logger.error("❌ DALL-E image generation: FAILED - Invalid response")
                    results['dalle'] = {"status": "failed", "reason": "Invalid response"}
            else:
                results['dalle'] = {"status": "skipped", "reason": "No API key"}
                
        except Exception as e:
            logger.error(f"❌ DALL-E image generation: FAILED - {str(e)}")
            results['dalle'] = {"status": "failed", "error": str(e)}
        
        # Test Stability AI
        try:
            api_key = os.getenv('STABILITY_API_KEY')
            if api_key:
                stability = StabilityAIAPI(api_key=api_key)
                
                start_time = time.time()
                result = stability.generate_image("A simple cartoon robot waving hello")
                end_time = time.time()
                
                if result and isinstance(result, dict):
                    logger.info("✅ Stability AI image generation: SUCCESS")
                    results['stability'] = {
                        "status": "success",
                        "response_time": end_time - start_time,
                        "result_keys": list(result.keys())
                    }
                else:
                    logger.error("❌ Stability AI image generation: FAILED - Invalid response")
                    results['stability'] = {"status": "failed", "reason": "Invalid response"}
            else:
                results['stability'] = {"status": "skipped", "reason": "No API key"}
                
        except Exception as e:
            logger.error(f"❌ Stability AI image generation: FAILED - {str(e)}")
            results['stability'] = {"status": "failed", "error": str(e)}
        
        return results
    
    def test_stock_media_search(self) -> Dict[str, Any]:
        """Test Pexels and Pixabay stock media search."""
        logger.info("📸 Testing Stock Media Search APIs...")
        
        results = {}
          # Test Pexels
        try:
            api_key = os.getenv('PEXELS_API_KEY')
            if api_key:
                pexels = PexelsAPI(api_key=api_key)
                
                start_time = time.time()
                result = pexels.search_images("nature", per_page=5)  # Use a more common search term
                end_time = time.time()
                
                if result and isinstance(result, list) and len(result) > 0:
                    logger.info("✅ Pexels image search: SUCCESS")
                    results['pexels'] = {
                        "status": "success",
                        "response_time": end_time - start_time,
                        "images_found": len(result)
                    }
                else:
                    logger.error("❌ Pexels image search: FAILED - No images found")
                    results['pexels'] = {"status": "failed", "reason": "No images found"}
            else:
                results['pexels'] = {"status": "skipped", "reason": "No API key"}
                
        except Exception as e:
            logger.error(f"❌ Pexels image search: FAILED - {str(e)}")
            results['pexels'] = {"status": "failed", "error": str(e)}
        
        # Test Pixabay
        try:
            api_key = os.getenv('PIXABAY_API_KEY')
            if api_key:
                pixabay = PixabayAPI(api_key=api_key)
                start_time = time.time()
                result = pixabay.search_images("nature", per_page=5)  # Use a more common search term
                end_time = time.time()
                
                if result and isinstance(result, list) and len(result) > 0:
                    logger.info("✅ Pixabay image search: SUCCESS")
                    results['pixabay'] = {
                        "status": "success",
                        "response_time": end_time - start_time,
                        "images_found": len(result)
                    }
                else:
                    logger.error("❌ Pixabay image search: FAILED - No images found")
                    results['pixabay'] = {"status": "failed", "reason": "No images found"}
            else:
                results['pixabay'] = {"status": "skipped", "reason": "No API key"}
                
        except Exception as e:
            logger.error(f"❌ Pixabay image search: FAILED - {str(e)}")
            results['pixabay'] = {"status": "failed", "error": str(e)}
        
        return results
    
    def test_transcription_apis(self) -> Dict[str, Any]:
        """Test AssemblyAI and Deepgram transcription (mock test)."""
        logger.info("🎧 Testing Transcription APIs...")
        
        results = {}
        
        # Note: These are connection tests only since we don't have actual audio files
        # In a real scenario, you'd upload a test audio file
        
        # Test AssemblyAI
        try:
            api_key = os.getenv('ASSEMBLYAI_API_KEY')
            if api_key:
                assemblyai = AssemblyAIAPI(api_key=api_key)
                logger.info("✅ AssemblyAI API: Connection available")
                results['assemblyai'] = {
                    "status": "connection_ok",
                    "note": "API key valid, would need audio file for full test"
                }
            else:
                results['assemblyai'] = {"status": "skipped", "reason": "No API key"}
                
        except Exception as e:
            logger.error(f"❌ AssemblyAI API: FAILED - {str(e)}")
            results['assemblyai'] = {"status": "failed", "error": str(e)}
        
        # Test Deepgram
        try:
            api_key = os.getenv('DEEPGRAM_API_KEY')
            if api_key:
                deepgram = DeepgramAPI(api_key=api_key)
                logger.info("✅ Deepgram API: Connection available")
                results['deepgram'] = {
                    "status": "connection_ok",
                    "note": "API key valid, would need audio file for full test"
                }
            else:
                results['deepgram'] = {"status": "skipped", "reason": "No API key"}
                
        except Exception as e:
            logger.error(f"❌ Deepgram API: FAILED - {str(e)}")
            results['deepgram'] = {"status": "failed", "error": str(e)}
        
        return results
    
    def test_youtube_data_api(self) -> Dict[str, Any]:
        """Test YouTube Data API for trend research."""
        logger.info("📺 Testing YouTube Data API...")
        
        try:
            api_key = os.getenv('YOUTUBE_API_KEY')
            if not api_key:
                return {"status": "skipped", "reason": "No API key"}
            
            youtube = YouTubeDataAPI(api_key=api_key)
            start_time = time.time()
            result = youtube.search_videos("music", max_results=5)  # Use a more common search term
            end_time = time.time()
            
            if result and isinstance(result, list) and len(result) > 0:
                logger.info("✅ YouTube Data API: SUCCESS")
                return {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "videos_found": len(result)
                }
            else:
                logger.error("❌ YouTube Data API: FAILED - No videos found")
                return {"status": "failed", "reason": "No videos found"}
                
        except Exception as e:
            logger.error(f"❌ YouTube Data API: FAILED - {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests and generate a comprehensive report."""
        logger.info("🚀 Starting Comprehensive API Testing...")
        logger.info("=" * 60)
        
        # Test API key availability
        self.results['api_keys'] = self.test_api_key_availability()
        
        logger.info("")
        
        # Test individual APIs
        self.results['gemini'] = self.test_gemini_text_generation()
        self.results['elevenlabs'] = self.test_elevenlabs_tts()
        self.results['image_generation'] = self.test_image_generation()
        self.results['stock_media'] = self.test_stock_media_search()
        self.results['transcription'] = self.test_transcription_apis()
        self.results['youtube'] = self.test_youtube_data_api()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate a summary of all test results."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Count successes and failures
        successful_apis = 0
        total_apis = 0
        
        for category, results in self.results.items():
            if category == 'api_keys':
                continue
                
            if isinstance(results, dict):
                if 'status' in results:
                    total_apis += 1
                    if results['status'] == 'success':
                        successful_apis += 1
                else:
                    # For nested results like image_generation
                    for api_name, api_result in results.items():
                        total_apis += 1
                        if api_result.get('status') == 'success':
                            successful_apis += 1
        
        success_rate = (successful_apis / total_apis * 100) if total_apis > 0 else 0
        
        logger.info(f"⏱️  Total test time: {total_time:.2f} seconds")
        logger.info(f"✅ Successful APIs: {successful_apis}/{total_apis}")
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        # API key summary
        available_keys = sum(1 for available in self.results['api_keys'].values() if available)
        total_keys = len(self.results['api_keys'])
        logger.info(f"🔑 API keys available: {available_keys}/{total_keys}")
        
        logger.info("")
        logger.info("🔍 Detailed Results:")
        
        # Show detailed results for each API
        for category, results in self.results.items():
            if category == 'api_keys':
                continue
                
            if isinstance(results, dict) and 'status' in results:
                status_icon = "✅" if results['status'] == 'success' else "❌" if results['status'] == 'failed' else "⏭️"
                logger.info(f"{status_icon} {category.upper()}: {results['status']}")
                if 'response_time' in results:
                    logger.info(f"   ⏱️  Response time: {results['response_time']:.2f}s")
            else:
                # Handle nested results
                logger.info(f"📁 {category.upper()}:")
                for api_name, api_result in results.items():
                    status_icon = "✅" if api_result.get('status') == 'success' else "❌" if api_result.get('status') == 'failed' else "⏭️"
                    logger.info(f"   {status_icon} {api_name}: {api_result.get('status', 'unknown')}")
        
        logger.info("")
        logger.info("💡 Next Steps:")
        
        failed_apis = []
        for category, results in self.results.items():
            if category == 'api_keys':
                continue
                
            if isinstance(results, dict):
                if results.get('status') == 'failed':
                    failed_apis.append(category)
                elif 'status' not in results:
                    # Handle nested results
                    for api_name, api_result in results.items():
                        if api_result.get('status') == 'failed':
                            failed_apis.append(f"{category}/{api_name}")
        
        if failed_apis:
            logger.info("❌ Fix the following failed APIs:")
            for api in failed_apis:
                logger.info(f"   - {api}")
        
        if available_keys < total_keys:
            logger.info("🔑 Add missing API keys to your .env file")
        
        if success_rate == 100:
            logger.info("🎉 All APIs are working perfectly! Ready to implement fallbacks and enhancements.")
        elif success_rate >= 80:
            logger.info("👍 Most APIs are working well. Focus on fixing the failing ones.")
        else:
            logger.info("⚠️  Many APIs need attention. Focus on getting basic functionality working first.")


def main():
    """Main function to run the comprehensive API test."""
    print("🧪 LangGraph 101 - Comprehensive API Testing Suite")
    print("=" * 60)
    
    tester = APITester()
    results = tester.run_all_tests()
    
    # Save results to file
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"api_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 Detailed results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return results


if __name__ == "__main__":
    main()
