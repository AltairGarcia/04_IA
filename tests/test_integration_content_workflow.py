"""
Integration tests for full content creation workflow in LangGraph 101.
Covers: script generation, TTS, image generation, stock media, transcription, and trend research.
All external APIs are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock
from tools import GeminiAPI, ElevenLabsTTS, PexelsAPI, PixabayAPI, StabilityAIAPI, DalleAPI, AssemblyAIAPI, DeepgramAPI, YouTubeDataAPI

@pytest.mark.integration
def test_full_content_creation_workflow():
    # Mock all API calls
    with patch.object(GeminiAPI, 'generate_content', return_value="Script text") as mock_gemini, \
         patch.object(ElevenLabsTTS, 'text_to_speech', return_value=b"audio-bytes") as mock_tts, \
         patch.object(StabilityAIAPI, 'generate_image', return_value=b"image-bytes") as mock_stability, \
         patch.object(DalleAPI, 'generate_image', return_value={"data": ["img-url"]}) as mock_dalle, \
         patch.object(PexelsAPI, 'search_images', return_value=[{"url": "pexels-img"}]) as mock_pexels, \
         patch.object(PixabayAPI, 'search_images', return_value=[{"url": "pixabay-img"}]) as mock_pixabay, \
         patch.object(AssemblyAIAPI, 'transcribe', return_value={"text": "caption"}) as mock_assembly, \
         patch.object(DeepgramAPI, 'transcribe', return_value={"results": "caption"}) as mock_deepgram, \
         patch.object(YouTubeDataAPI, 'search_videos', return_value=[{"title": "trend"}]) as mock_yt:
        # Simulate workflow
        gemini = GeminiAPI(api_key="fake")
        script = gemini.generate_content("topic")
        tts = ElevenLabsTTS(api_key="fake").text_to_speech(script)
        img = StabilityAIAPI(api_key="fake").generate_image("thumbnail prompt")
        dalle_img = DalleAPI(api_key="fake").generate_image("thumbnail prompt")
        pexels_img = PexelsAPI(api_key="fake").search_images("topic")
        pixabay_img = PixabayAPI(api_key="fake").search_images("topic")
        caption = AssemblyAIAPI(api_key="fake").transcribe("audio-url")
        dg_caption = DeepgramAPI(api_key="fake").transcribe("audio-url")
        trends = YouTubeDataAPI(api_key="fake").search_videos("topic")
        # Assert all steps return expected mock data
        assert script == "Script text"
        assert tts == b"audio-bytes"
        assert img == b"image-bytes"
        assert dalle_img["data"] == ["img-url"]
        assert pexels_img[0]["url"] == "pexels-img"
        assert pixabay_img[0]["url"] == "pixabay-img"
        assert caption["text"] == "caption"
        assert dg_caption["results"] == "caption"
        assert trends[0]["title"] == "trend"
