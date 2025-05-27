"""
Unit tests for content creation API wrappers in tools.py
"""
import pytest
from unittest.mock import patch, MagicMock
from tools import GeminiAPI, YouTubeDataAPI, ElevenLabsTTS, PexelsAPI, PixabayAPI, StabilityAIAPI, DalleAPI, AssemblyAIAPI, DeepgramAPI

class TestGeminiAPI:
    @patch('tools.requests.post')
    def test_generate_content_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Hello Gemini!"}]}}]
        }
        api = GeminiAPI(api_key="fake-key")
        result = api.generate_content("Say hi")
        assert result == "Hello Gemini!"

    @patch('tools.requests.post')
    def test_generate_content_error(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        api = GeminiAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.generate_content("fail")

class TestYouTubeDataAPI:
    @patch('tools.requests.get')
    def test_search_videos_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"items": ["video1", "video2"]}
        api = YouTubeDataAPI(api_key="fake-key")
        result = api.search_videos("test")
        assert result == ["video1", "video2"]

    @patch('tools.requests.get')
    def test_search_videos_error(self, mock_get):
        mock_get.return_value.status_code = 403
        mock_get.return_value.text = "Forbidden"
        api = YouTubeDataAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.search_videos("fail")

class TestElevenLabsTTS:
    @patch('tools.requests.post')
    def test_text_to_speech_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"audio-bytes"
        api = ElevenLabsTTS(api_key="fake-key")
        result = api.text_to_speech("Hello")
        assert result == b"audio-bytes"

    @patch('tools.requests.post')
    def test_text_to_speech_error(self, mock_post):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Server Error"
        api = ElevenLabsTTS(api_key="fake-key")
        with pytest.raises(Exception):
            api.text_to_speech("fail")

class TestPexelsAPI:
    @patch('tools.requests.get')
    def test_search_images_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"photos": [1,2]}
        api = PexelsAPI(api_key="fake-key")
        result = api.search_images("cat")
        assert result == [1,2]

    @patch('tools.requests.get')
    def test_search_images_error(self, mock_get):
        mock_get.return_value.status_code = 401
        mock_get.return_value.text = "Unauthorized"
        api = PexelsAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.search_images("fail")

class TestPixabayAPI:
    @patch('tools.requests.get')
    def test_search_images_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"hits": [1,2]}
        api = PixabayAPI(api_key="fake-key")
        result = api.search_images("cat")
        assert result == [1,2]

    @patch('tools.requests.get')
    def test_search_images_error(self, mock_get):
        mock_get.return_value.status_code = 401
        mock_get.return_value.text = "Unauthorized"
        api = PixabayAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.search_images("fail")

class TestStabilityAIAPI:
    @patch('tools.requests.post')
    def test_generate_image_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"image-bytes"
        api = StabilityAIAPI(api_key="fake-key")
        result = api.generate_image("cat")
        assert result == b"image-bytes"

    @patch('tools.requests.post')
    def test_generate_image_error(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        api = StabilityAIAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.generate_image("fail")

class TestDalleAPI:
    @patch('tools.requests.post')
    def test_generate_image_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"data": ["img1"]}
        api = DalleAPI(api_key="fake-key")
        result = api.generate_image("cat")
        assert "data" in result

    @patch('tools.requests.post')
    def test_generate_image_error(self, mock_post):
        mock_post.return_value.status_code = 401
        mock_post.return_value.text = "Unauthorized"
        api = DalleAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.generate_image("fail")

class TestAssemblyAIAPI:
    @patch('tools.requests.post')
    def test_transcribe_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"text": "transcribed"}
        api = AssemblyAIAPI(api_key="fake-key")
        result = api.transcribe("http://audio.url")
        assert result["text"] == "transcribed"

    @patch('tools.requests.post')
    def test_transcribe_error(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        api = AssemblyAIAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.transcribe("fail")

class TestDeepgramAPI:
    @patch('tools.requests.post')
    def test_transcribe_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"results": "ok"}
        api = DeepgramAPI(api_key="fake-key")
        result = api.transcribe("http://audio.url")
        assert "results" in result or isinstance(result, dict)

    @patch('tools.requests.post')
    def test_transcribe_error(self, mock_post):
        mock_post.return_value.status_code = 403
        mock_post.return_value.text = "Forbidden"
        api = DeepgramAPI(api_key="fake-key")
        with pytest.raises(Exception):
            api.transcribe("fail")
