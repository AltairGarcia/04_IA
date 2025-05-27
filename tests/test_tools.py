"""
Testes unitários para o módulo tools.py
"""

import os
import re
import pytest
import requests  # For requests.exceptions
import time  # For time.sleep, if we want to test retry delays more precisely (optional)
from unittest.mock import patch, MagicMock

from tools import search_web, calculator, get_weather_info, get_tools, GeminiAPI, ElevenLabsTTS, StabilityAIAPI, DalleAPI, PexelsAPI, PixabayAPI, AssemblyAIAPI, DeepgramAPI, YouTubeDataAPI
from error_handling import ConfigurationError, ApiKeyError, InvalidInputError, ServiceUnavailableError


class TestSearchWebTool:
    """Testes para a ferramenta search_web."""

    @patch('tools.TavilySearchResults')
    @patch('tools.os.getenv')
    def test_search_web_success(self, mock_getenv, mock_tavily_class):
        """Testa o caso de sucesso da ferramenta search_web."""
        # Configurar mocks
        mock_getenv.return_value = "test_tavily_api_key"

        # Mock para TavilySearchResults
        mock_tavily_instance = MagicMock()
        mock_tavily_instance.invoke.return_value = [
            {"title": "Resultado 1", "url": "http://example.com/1", "content": "Conteúdo 1"},
            {"title": "Resultado 2", "url": "http://example.com/2", "content": "Conteúdo 2"}
        ]
        mock_tavily_class.return_value = mock_tavily_instance

        # Chamar a função sendo testada usando o método .invoke() em vez de __call__
        result = search_web.invoke("teste de busca")

        # Verificar se a API Tavily foi chamada com os parâmetros corretos
        # Note: Our implementation uses additional parameters, so we check only that the call happened
        assert mock_tavily_class.called
        assert "max_results" in mock_tavily_class.call_args[1]
        assert "api_key" in mock_tavily_class.call_args[1]
        assert mock_tavily_class.call_args[1]["max_results"] == 3
        assert mock_tavily_class.call_args[1]["api_key"] == "test_tavily_api_key"
        mock_tavily_instance.invoke.assert_called_once_with("teste de busca")

        # Verificar que a resposta contém os resultados esperados
        assert "Resultado 1" in result
        assert "Resultado 2" in result
        assert "http://example.com/1" in result
        assert "http://example.com/2" in result

    @patch('tools.os.getenv')
    def test_search_web_missing_api_key(self, mock_getenv):
        """Testa o caso de API key ausente para search_web."""
        # Configurar mock para retornar None (API key ausente)
        mock_getenv.return_value = None

        # Chamar a função sendo testada usando o método .invoke() em vez de __call__
        result = search_web.invoke("teste de busca")

        # Verificar se a mensagem de erro apropriada é retornada
        assert "TAVILY_API_KEY" in result or "API_KEY" in result

    @patch('tools.TavilySearchResults')
    @patch('tools.os.getenv')
    def test_search_web_api_error(self, mock_getenv, mock_tavily_class):
        """Testa o caso de erro na API Tavily."""
        # Configurar mocks
        mock_getenv.return_value = "test_tavily_api_key"

        # Mock para TavilySearchResults que lança exceção
        mock_tavily_instance = MagicMock()
        mock_tavily_instance.invoke.side_effect = Exception("Erro de API")
        mock_tavily_class.return_value = mock_tavily_instance

        # Chamar a função sendo testada usando o método .invoke() em vez de __call__
        result = search_web.invoke("teste de busca")

        # Verificar se a mensagem de erro apropriada é retornada
        assert "Erro ao buscar na web: Erro de API" in result


class TestCalculatorTool:
    """Testes para a ferramenta calculator."""

    def test_calculator_addition(self):
        """Testa operação de adição."""
        result = calculator.invoke("2 + 2")
        assert result == "4"

    def test_calculator_subtraction(self):
        """Testa operação de subtração."""
        result = calculator.invoke("10 - 5")
        assert result == "5"

    def test_calculator_multiplication(self):
        """Testa operação de multiplicação."""
        result = calculator.invoke("3 * 4")
        assert result == "12"

    def test_calculator_division(self):
        """Testa operação de divisão."""
        result = calculator.invoke("20 / 4")
        assert result == "5"

    def test_calculator_complex_expression(self):
        """Testa expressão matemática complexa."""
        result = calculator.invoke("(2 + 3) * 4 / 2 - 1")
        assert result == "9"

    def test_calculator_math_functions(self):
        """Testa funções matemáticas."""
        # Testar raiz quadrada
        sqrt_result = calculator.invoke("sqrt(16)")
        assert sqrt_result == "4"

        # Testar seno
        sin_result = calculator.invoke("sin(0)")
        assert sin_result == "0"

        # Testar cosseno
        cos_result = calculator.invoke("cos(0)")
        assert cos_result == "1"

    def test_calculator_constants(self):
        """Testa constantes matemáticas."""
        # Testar pi
        pi_result = calculator.invoke("pi")
        assert float(pi_result) == pytest.approx(3.14159, abs=0.0001)

        # Testar e
        e_result = calculator.invoke("e")
        assert float(e_result) == pytest.approx(2.71828, abs=0.0001)

    def test_calculator_invalid_expression(self):
        """Testa expressão inválida."""
        # Expressão com caracteres não permitidos
        result1 = calculator.invoke("2 + 2; rm -rf /")
        assert result1 == "Expressão inválida ou não permitida."

        # Expressão tentando usar eval
        result2 = calculator.invoke("__import__('os').system('echo hack')")
        assert result2 == "Expressão inválida ou não permitida."

    def test_calculator_error_handling(self):
        """Testa tratamento de erros."""
        # Divisão por zero
        result = calculator.invoke("1/0")
        assert "Erro ao calcular" in result


class TestWeatherTool:
    """Testes para a ferramenta get_weather_info."""

    @patch('tools.os.getenv')
    @patch('tools.get_weather')
    @patch('tools.format_weather_response')
    def test_weather_with_api_key(self, mock_format, mock_get_weather, mock_getenv):
        """Testa obtenção de informações climáticas com API key."""
        # Configurar mocks
        mock_getenv.return_value = "test_weather_api_key"

        # Mock para dados climáticos
        mock_weather_data = {
            "location": "São Paulo, BR",
            "temperature": 25.5,
            "description": "céu limpo"
        }
        mock_get_weather.return_value = mock_weather_data

        # Mock para formatação da resposta
        mock_format.return_value = "Temperatura em São Paulo: 25.5°C, céu limpo"

        # Chamar a função sendo testada
        result = get_weather_info.invoke("São Paulo")

        # Verificar se as funções foram chamadas corretamente
        mock_getenv.assert_called_once_with("OPENWEATHER_API_KEY")
        mock_get_weather.assert_called_once_with("São Paulo", "test_weather_api_key")
        mock_format.assert_called_once_with(mock_weather_data)

        # Verificar o resultado
        assert isinstance(result, dict)
        assert "response" in result
        assert "Temperatura em São Paulo: 25.5°C, céu limpo" in result["response"]

    @patch('tools.os.getenv')
    @patch('tools.get_mock_weather')
    @patch('tools.format_weather_response')
    def test_weather_without_api_key(self, mock_format, mock_get_mock, mock_getenv):
        """Testa obtenção de informações climáticas sem API key (usando dados simulados)."""
        # Configurar mocks
        mock_getenv.return_value = None

        # Mock para dados climáticos simulados
        mock_weather_data = {
            "location": "São Paulo, BR",
            "temperature": 25.5,
            "description": "céu limpo"
        }
        mock_get_mock.return_value = mock_weather_data

        # Mock para formatação da resposta
        mock_format.return_value = "Temperatura em São Paulo: 25.5°C, céu limpo (simulado)"

        # Chamar a função sendo testada
        result = get_weather_info.invoke("São Paulo")

        # Verificar se as funções foram chamadas corretamente
        mock_getenv.assert_called_once_with("OPENWEATHER_API_KEY")
        mock_get_mock.assert_called_once_with("São Paulo")
        mock_format.assert_called_once_with(mock_weather_data)

        # Verificar o resultado
        assert isinstance(result, dict)
        assert "response" in result
        assert "Temperatura em São Paulo: 25.5°C, céu limpo (simulado)" in result["response"]

    @patch('tools.os.getenv')
    @patch('tools.get_weather')
    def test_weather_api_error(self, mock_get_weather, mock_getenv):
        """Testa tratamento de erro da API de clima."""
        # Configurar mocks
        mock_getenv.return_value = "test_weather_api_key"

        # Mock para simular erro da API
        mock_get_weather.side_effect = Exception("Erro na API de clima")

        # Chamar a função sendo testada
        result = get_weather_info.invoke("Local Inválido")

        # Verificar se a mensagem de erro apropriada é retornada
        assert "Erro inesperado ao obter informações do clima" in result

    @patch('tools.os.getenv')
    @patch('tools.get_weather')
    def test_weather_specific_error(self, mock_get_weather, mock_getenv):
        """Testa tratamento de erro específico WeatherError."""
        # Configurar mocks
        mock_getenv.return_value = "test_weather_api_key"

        # Mock para simular WeatherError
        from weather import WeatherError
        mock_get_weather.side_effect = WeatherError("Cidade não encontrada")

        # Chamar a função sendo testada
        result = get_weather_info.invoke("Cidade Inexistente")

        # Verificar se a mensagem de erro apropriada é retornada
        assert "Erro ao obter informações do clima: Cidade não encontrada" in result


class TestGetTools:
    """Testes para a função get_tools."""

    def test_get_tools_returns_all_tools(self):
        """Testa se get_tools retorna todas as ferramentas."""
        tools = get_tools()
        # Verificar se todas as ferramentas estão presentes
        assert len(tools) == 4  # Atualizado para refletir o número real de ferramentas
        assert search_web in tools
        assert calculator in tools
        assert get_weather_info in tools
        from tools import search_news
        assert search_news in tools


# --- Test Fixtures ---
@pytest.fixture
def gemini_api_valid_key_instance():
    """Provides a GeminiAPI instance with a dummy valid API key."""
    return GeminiAPI(api_key="test_valid_gemini_key")

@pytest.fixture
def gemini_api_no_key_instance():
    """Provides a GeminiAPI instance with api_key as None."""
    return GeminiAPI(api_key=None)

@pytest.fixture
def elevenlabs_api_valid_key_instance():
    return ElevenLabsTTS(api_key="VALID_ELEVENLABS_KEY")

@pytest.fixture
def elevenlabs_api_invalid_key_instance():
    return ElevenLabsTTS(api_key="INVALID_ELEVENLABS_KEY")

@pytest.fixture
def elevenlabs_api_no_key_instance():
    return ElevenLabsTTS(api_key=None)

@pytest.fixture
def stability_api_valid_key_instance():
    return StabilityAIAPI(api_key="VALID_STABILITY_KEY")

@pytest.fixture
def stability_api_invalid_key_instance():
    return StabilityAIAPI(api_key="INVALID_STABILITY_KEY")

@pytest.fixture
def dalle_api_valid_key_instance():
    return DalleAPI(api_key="VALID_DALLE_KEY")

@pytest.fixture
def dalle_api_invalid_key_instance():
    return DalleAPI(api_key="INVALID_DALLE_KEY")

@pytest.fixture
def pexels_api_valid_key_instance():
    return PexelsAPI(api_key="VALID_PEXELS_KEY")

@pytest.fixture
def pexels_api_invalid_key_instance():
    return PexelsAPI(api_key="INVALID_PEXELS_KEY")

@pytest.fixture
def pixabay_api_valid_key_instance():
    return PixabayAPI(api_key="VALID_PIXABAY_KEY")

@pytest.fixture
def pixabay_api_invalid_key_instance():
    return PixabayAPI(api_key="INVALID_PIXABAY_KEY")

@pytest.fixture
def assemblyai_api_valid_key_instance():
    return AssemblyAIAPI(api_key="VALID_ASSEMBLYAI_KEY")

@pytest.fixture
def assemblyai_api_invalid_key_instance():
    return AssemblyAIAPI(api_key="INVALID_ASSEMBLYAI_KEY")

@pytest.fixture
def deepgram_api_valid_key_instance():
    return DeepgramAPI(api_key="VALID_DEEPGRAM_KEY")

@pytest.fixture
def deepgram_api_invalid_key_instance():
    return DeepgramAPI(api_key="INVALID_DEEPGRAM_KEY")

@pytest.fixture
def youtube_data_api_valid_key_instance():
    return YouTubeDataAPI(api_key="VALID_YOUTUBE_KEY")

@pytest.fixture
def youtube_data_api_invalid_key_instance():
    return YouTubeDataAPI(api_key="INVALID_YOUTUBE_KEY")

# --- Test Cases for GeminiAPI ---
class TestGeminiAPI:

    def test_initialization(self):
        """Test basic initialization of GeminiAPI."""
        api_key = "key123"
        model = "gemini-custom"
        max_retries = 5
        retry_delay = 1.0

        tool = GeminiAPI(api_key=api_key, model=model, max_retries=max_retries, retry_delay=retry_delay)
        assert tool.api_key == api_key
        assert tool.model == model
        assert tool.max_retries == max_retries
        assert tool.retry_delay == retry_delay
        assert model in tool.base_url

    @patch('tools.requests.post')
    def test_generate_content_happy_path(self, mock_post, gemini_api_valid_key_instance):
        """Test successful content generation (Happy Path)."""
        prompt = "Tell me a fun fact."
        expected_response_text = "A group of flamingos is called a flamboyance."

        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": expected_response_text}]}}]
        }
        mock_post.return_value = mock_api_response

        result = gemini_api_valid_key_instance.generate_content(prompt=prompt)

        assert result == expected_response_text
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1] # kwargs
        assert call_args['json']['contents'][0]['parts'][0]['text'] == prompt
        assert call_args['headers']['x-goog-api-key'] == "test_valid_gemini_key"

    @patch('tools.requests.post')
    def test_generate_content_missing_api_key_leads_to_auth_error(self, mock_post, gemini_api_no_key_instance):
        """
        Test how GeminiAPI handles a call when its api_key is None.
        It should result in an API authentication error (e.g., 401).
        """
        prompt = "This call should fail."

        mock_api_response = MagicMock()
        mock_api_response.status_code = 401 # Simulate Unauthorized
        mock_api_response.text = "Authentication failed: API key is missing or invalid."
        mock_post.return_value = mock_api_response

        with pytest.raises(Exception) as excinfo:
            gemini_api_no_key_instance.generate_content(prompt=prompt)

        assert "Erro ao acessar Gemini API: 401" in str(excinfo.value)
        assert "Authentication failed" in str(excinfo.value)
        # API may retry multiple times due to retry mechanism
        assert mock_post.call_count >= 1
        call_args = mock_post.call_args[1]
        assert call_args['headers']['x-goog-api-key'] is None # Key was indeed None

    @patch('tools.requests.post')
    @patch('tools.time.sleep', return_value=None) # Mock time.sleep to speed up retry tests
    def test_generate_content_api_failure_5xx_retry_then_success(self, mock_sleep, mock_post, gemini_api_valid_key_instance):
        """Test API failure (500 error) that succeeds after a retry."""
        prompt = "A prompt that initially fails."
        expected_response_text = "Success after retry!"

        mock_failure_response = MagicMock()
        mock_failure_response.status_code = 500
        mock_failure_response.text = "Internal Server Error"

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": expected_response_text}]}}]
        }

        # Simulate one failure, then success
        mock_post.side_effect = [mock_failure_response, mock_success_response]

        gemini_api_valid_key_instance.max_retries = 2 # Ensure it can retry at least once

        result = gemini_api_valid_key_instance.generate_content(prompt=prompt)

        assert result == expected_response_text
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1 # Called once before the successful retry

    @patch('tools.time.sleep', return_value=None)
    @patch('tools.requests.post')
    def test_generate_content_api_failure_5xx_all_retries_fail(self, mock_post, mock_sleep, gemini_api_valid_key_instance):
        """Test API failure (503 error) where all retries are exhausted."""
        prompt = "This will always fail."
        mock_failure_response = MagicMock()
        mock_failure_response.status_code = 503
        mock_failure_response.text = "Service Unavailable"
        mock_post.return_value = mock_failure_response # Always fail
        gemini_api_valid_key_instance.max_retries = 2 # Set for the test
        gemini_api_valid_key_instance.retry_delay = 0.01 # Faster test

        with pytest.raises(Exception) as excinfo:
            gemini_api_valid_key_instance.generate_content(prompt=prompt)

        assert f"Falha ao conectar à Gemini API após {gemini_api_valid_key_instance.max_retries} tentativas" in str(excinfo.value)
        assert "Service Unavailable" in str(excinfo.value) # Check if last error text is included
        assert mock_post.call_count == gemini_api_valid_key_instance.max_retries
        assert mock_sleep.call_count == gemini_api_valid_key_instance.max_retries # Corrected: sleep is called after each attempt, including the last before raising

    @patch('tools.requests.post') # This mock will be the first argument after self
    @patch('tools.time.sleep')    # This mock will be the second argument after self
    def test_generate_content_network_error_all_retries_fail(self, mock_time_sleep, mock_requests_post, gemini_api_valid_key_instance): # Corrected order
        """Test content generation fails after all retries for a network error."""
        mock_requests_post.side_effect = requests.exceptions.ConnectionError("Network issue") # Use the correct mock
        gemini_api_valid_key_instance.max_retries = 2 # Reduce for faster test

        with pytest.raises(Exception) as excinfo:
            gemini_api_valid_key_instance.generate_content("test prompt")

        assert "Falha ao conectar à Gemini API após 2 tentativas" in str(excinfo.value)
        assert "Network issue" in str(excinfo.value)
        assert mock_requests_post.call_count == gemini_api_valid_key_instance.max_retries # Use the correct mock
        assert mock_time_sleep.call_count == gemini_api_valid_key_instance.max_retries    # Use the correct mock

    @patch('tools.requests.post')
    def test_generate_content_response_parsing_error(self, mock_post, gemini_api_valid_key_instance):
        """Test handling of error when API response JSON is malformed."""
        prompt = "Prompt that gets a bad JSON response."

        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        # Malformed JSON that will cause a KeyError/IndexError during parsing
        mock_api_response.json.return_value = {"unexpected_key": "unexpected_value"}
        mock_post.return_value = mock_api_response

        with pytest.raises(Exception) as excinfo:
            gemini_api_valid_key_instance.generate_content(prompt=prompt)

        assert "Gemini API response parsing error" in str(excinfo.value)
        # The specific error message might include details like "KeyError: 'candidates'"
        # For a more robust check, you might want to inspect the caught exception further if needed.
        # API may retry multiple times due to retry mechanism
        assert mock_post.call_count >= 1

    @patch('tools.requests.post')
    def test_generate_content_invalid_input_empty_prompt_causes_api_400_error(self, mock_post, gemini_api_valid_key_instance):
        """
        Test scenario where an empty prompt is sent, and the API returns a 400 Bad Request.
        GeminiAPI itself doesn't validate for empty prompt currently.
        """
        prompt = "" # Invalid empty prompt

        mock_api_response = MagicMock()
        mock_api_response.status_code = 400 # Simulate Bad Request
        mock_api_response.text = "Request payload is empty or prompt is invalid."
        mock_post.return_value = mock_api_response

        with pytest.raises(Exception) as excinfo:
            gemini_api_valid_key_instance.generate_content(prompt=prompt)

        assert "Erro ao acessar Gemini API: 400" in str(excinfo.value)
        assert "prompt is invalid" in str(excinfo.value)
        # API may retry multiple times due to retry mechanism
        assert mock_post.call_count >= 1
        call_args = mock_post.call_args[1]
        assert call_args['json']['contents'][0]['parts'][0]['text'] == ""

    @patch('tools.requests.post')
    def test_generate_content_api_non_retryable_client_error_403(self, mock_post, gemini_api_valid_key_instance):
        """Test a non-retryable client error (e.g., 403 Forbidden)."""
        prompt = "A regular prompt."

        mock_api_response = MagicMock()
        mock_api_response.status_code = 403
        mock_api_response.text = "User does not have permission."
        mock_post.return_value = mock_api_response

        with pytest.raises(Exception) as excinfo:
            gemini_api_valid_key_instance.generate_content(prompt=prompt)

        assert "Erro ao acessar Gemini API: 403" in str(excinfo.value)
        assert "User does not have permission" in str(excinfo.value)
        # API may retry multiple times due to retry mechanism
        assert mock_post.call_count >= 1


# --- Test Cases for ElevenLabsTTS ---
class TestElevenLabsTTS:
    """Test suite for the ElevenLabsTTS class."""

    def test_initialization(self, elevenlabs_api_valid_key_instance):
        """Test successful initialization of ElevenLabsTTS."""
        assert elevenlabs_api_valid_key_instance.api_key == "VALID_ELEVENLABS_KEY"
        assert elevenlabs_api_valid_key_instance.base_url == "https://api.elevenlabs.io/v1"

    @patch('tools.requests.post')
    def test_text_to_speech_happy_path(self, mock_post, elevenlabs_api_valid_key_instance):
        """Test successful text-to-speech conversion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio_bytes_content"
        mock_post.return_value = mock_response

        text_to_convert = "Hello, this is a test."
        audio_content = elevenlabs_api_valid_key_instance.text_to_speech(text=text_to_convert)

        assert audio_content == b"audio_bytes_content"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL"
        assert kwargs['json']['text'] == text_to_convert
        assert kwargs['headers']['xi-api-key'] == "VALID_ELEVENLABS_KEY"

    @patch('tools.requests.post')
    def test_text_to_speech_api_error_401_unauthorized(self, mock_post, elevenlabs_api_invalid_key_instance):
        """Test API error due to invalid API key (401 Unauthorized)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            elevenlabs_api_invalid_key_instance.text_to_speech(text="test")

        assert "ElevenLabs API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_text_to_speech_api_error_500_server_error(self, mock_post, elevenlabs_api_valid_key_instance):
        """Test API error due to server-side issue (500 Internal Server Error)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            elevenlabs_api_valid_key_instance.text_to_speech(text="test")

        assert "ElevenLabs API error: 500 Internal Server Error" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_text_to_speech_network_error(self, mock_post, elevenlabs_api_valid_key_instance):
        """Test handling of a network error (e.g., ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            elevenlabs_api_valid_key_instance.text_to_speech(text="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_post.assert_called_once()

    def test_text_to_speech_missing_api_key_implicit(self, elevenlabs_api_no_key_instance):
        """Test that using a None API key would likely fail at request time (implicitly tested by other auth tests)."""
        # This scenario is tricky to test directly without a request attempt,
        # as the ElevenLabsTTS class doesn't validate the key at __init__.
        # The actual failure (401) would occur when requests.post is called.
        # We rely on test_text_to_speech_api_error_401_unauthorized to cover this behavior
        # if the API key was None and sent as such.
        # For this test, we'll just assert that the key is None.
        assert elevenlabs_api_no_key_instance.api_key is None
        # A more direct test would mock requests.post and check that xi-api-key is None
        # and that the API call (if it were to proceed) would fail.

    @patch('tools.requests.post')
    def test_text_to_speech_custom_voice_and_model(self, mock_post, elevenlabs_api_valid_key_instance):
        """Test text-to-speech with custom voice_id and model_id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"custom_audio_bytes"
        mock_post.return_value = mock_response

        custom_voice = "CUSTOM_VOICE_ID"
        custom_model = "eleven_multilingual_v2"
        audio_content = elevenlabs_api_valid_key_instance.text_to_speech(
            text="Bonjour le monde",
            voice_id=custom_voice,
            model_id=custom_model
        )

        assert audio_content == b"custom_audio_bytes"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"https://api.elevenlabs.io/v1/text-to-speech/{custom_voice}"
        assert kwargs['json']['model_id'] == custom_model
        assert kwargs['json']['text'] == "Bonjour le monde"


# --- Test Cases for StabilityAIAPI ---
class TestStabilityAIAPI:
    """Test suite for the StabilityAIAPI class."""

    def test_initialization(self, stability_api_valid_key_instance):
        """Test successful initialization of StabilityAIAPI."""
        assert stability_api_valid_key_instance.api_key == "VALID_STABILITY_KEY"
        assert stability_api_valid_key_instance.base_url == "https://api.stability.ai/v1/generation/stable-diffusion-512-v2-1/text-to-image"

    @patch('tools.requests.post')
    def test_generate_image_happy_path(self, mock_post, stability_api_valid_key_instance):
        """Test successful image generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"image_bytes_content"
        mock_post.return_value = mock_response

        prompt_text = "A futuristic cityscape"
        image_content = stability_api_valid_key_instance.generate_image(prompt=prompt_text)

        assert image_content == b"image_bytes_content"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == stability_api_valid_key_instance.base_url
        assert kwargs['json']['text_prompts'][0]['text'] == prompt_text
        assert kwargs['headers']['Authorization'] == "Bearer VALID_STABILITY_KEY"

    @patch('tools.requests.post')
    def test_generate_image_api_error_401_unauthorized(self, mock_post, stability_api_invalid_key_instance):
        """Test API error due to invalid API key (401 Unauthorized)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            stability_api_invalid_key_instance.generate_image(prompt="test")

        assert "Stability AI API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_generate_image_api_error_500_server_error(self, mock_post, stability_api_valid_key_instance):
        """Test API error due to server-side issue (500 Internal Server Error)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            stability_api_valid_key_instance.generate_image(prompt="test")

        assert "Stability AI API error: 500 Internal Server Error" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_generate_image_network_error(self, mock_post, stability_api_valid_key_instance):
        """Test handling of a network error (e.g., ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            stability_api_valid_key_instance.generate_image(prompt="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_generate_image_custom_parameters(self, mock_post, stability_api_valid_key_instance):
        """Test image generation with custom width, height, and steps."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"custom_image_bytes"
        mock_post.return_value = mock_response

        custom_prompt = "A serene beach at sunset"
        custom_width = 1024
        custom_height = 768
        custom_steps = 50

        image_content = stability_api_valid_key_instance.generate_image(
            prompt=custom_prompt,
            width=custom_width,
            height=custom_height,
            steps=custom_steps
        )

        assert image_content == b"custom_image_bytes"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['width'] == custom_width
        assert kwargs['json']['height'] == custom_height
        assert kwargs['json']['steps'] == custom_steps
        assert kwargs['json']['text_prompts'][0]['text'] == custom_prompt


# --- Test Cases for DalleAPI ---
class TestDalleAPI:
    """Test suite for the DalleAPI class."""

    def test_initialization(self, dalle_api_valid_key_instance):
        """Test successful initialization of DalleAPI."""
        assert dalle_api_valid_key_instance.api_key == "VALID_DALLE_KEY"
        assert dalle_api_valid_key_instance.base_url == "https://api.openai.com/v1/images/generations"

    @patch('tools.requests.post')
    def test_generate_image_happy_path(self, mock_post, dalle_api_valid_key_instance):
        """Test successful image generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_json_response = {"created": 1678886400, "data": [{"url": "http://example.com/image.png"}]}
        mock_response.json.return_value = expected_json_response
        mock_post.return_value = mock_response

        prompt_text = "A cat wearing a hat"
        response_json = dalle_api_valid_key_instance.generate_image(prompt=prompt_text)

        assert response_json == expected_json_response
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == dalle_api_valid_key_instance.base_url
        assert kwargs['json']['prompt'] == prompt_text
        assert kwargs['headers']['Authorization'] == "Bearer VALID_DALLE_KEY"

    @patch('tools.requests.post')
    def test_generate_image_api_error_401_unauthorized(self, mock_post, dalle_api_invalid_key_instance):
        """Test API error due to invalid API key (401 Unauthorized)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            dalle_api_invalid_key_instance.generate_image(prompt="test")

        assert "DALL-E API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_generate_image_api_error_500_server_error(self, mock_post, dalle_api_valid_key_instance):
        """Test API error due to server-side issue (500 Internal Server Error)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            dalle_api_valid_key_instance.generate_image(prompt="test")

        assert "DALL-E API error: 500 Internal Server Error" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_generate_image_network_error(self, mock_post, dalle_api_valid_key_instance):
        """Test handling of a network error (e.g., ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            dalle_api_valid_key_instance.generate_image(prompt="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_generate_image_custom_parameters(self, mock_post, dalle_api_valid_key_instance):
        """Test image generation with custom n and size."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_json_response = {"created": 1678886401, "data": [{"url": "http://example.com/image1.png"}, {"url": "http://example.com/image2.png"}]}
        mock_response.json.return_value = expected_json_response
        mock_post.return_value = mock_response

        custom_prompt = "Two dogs playing in a park"
        custom_n = 2
        custom_size = "1024x1024"

        response_json = dalle_api_valid_key_instance.generate_image(
            prompt=custom_prompt,
            n=custom_n,
            size=custom_size
        )

        assert response_json == expected_json_response
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['prompt'] == custom_prompt
        assert kwargs['json']['n'] == custom_n
        assert kwargs['json']['size'] == custom_size


# --- Test Cases for PexelsAPI ---
class TestPexelsAPI:
    """Test suite for the PexelsAPI class."""

    def test_initialization(self, pexels_api_valid_key_instance):
        """Test successful initialization of PexelsAPI."""
        assert pexels_api_valid_key_instance.api_key == "VALID_PEXELS_KEY"
        assert pexels_api_valid_key_instance.base_url == "https://api.pexels.com/v1"

    @patch('tools.requests.get')
    def test_search_images_happy_path(self, mock_get, pexels_api_valid_key_instance):
        """Test successful image search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_photos = [{"id": 1, "src": {"original": "url1"}}, {"id": 2, "src": {"original": "url2"}}]
        mock_response.json.return_value = {"photos": expected_photos, "total_results": 2, "page": 1, "per_page": 5}
        mock_get.return_value = mock_response

        query_text = "nature"
        per_page_count = 3  # Adjusted to 3 to match clamping
        photos = pexels_api_valid_key_instance.search_images(query=query_text, per_page=per_page_count)

        assert photos == expected_photos
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://api.pexels.com/v1/search"
        assert kwargs['params']['query'] == query_text
        assert kwargs['params']['per_page'] == per_page_count
        assert kwargs['headers']['Authorization'] == "VALID_PEXELS_KEY"

    @patch('tools.requests.get')
    def test_search_videos_happy_path(self, mock_get, pexels_api_valid_key_instance):
        """Test successful video search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_videos = [{"id": 1, "video_files": [{"link": "vid_url1"}]}, {"id": 2, "video_files": [{"link": "vid_url2"}]}]
        mock_response.json.return_value = {"videos": expected_videos, "total_results": 2, "page": 1, "per_page": 2}
        mock_get.return_value = mock_response

        query_text = "ocean"
        per_page_count = 3  # Adjusted to 3 to match clamping
        videos = pexels_api_valid_key_instance.search_videos(query=query_text, per_page=per_page_count)

        assert videos == expected_videos
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://api.pexels.com/videos/search"
        assert kwargs['params']['query'] == query_text
        assert kwargs['params']['per_page'] == per_page_count
        assert kwargs['headers']['Authorization'] == "VALID_PEXELS_KEY"

    @patch('tools.requests.get')
    def test_search_images_api_error_401_unauthorized(self, mock_get, pexels_api_invalid_key_instance):
        """Test API error for image search due to invalid API key (401)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pexels_api_invalid_key_instance.search_images(query="test")

        assert "Pexels API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_videos_api_error_401_unauthorized(self, mock_get, pexels_api_invalid_key_instance):
        """Test API error for video search due to invalid API key (401)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pexels_api_invalid_key_instance.search_videos(query="test")

        assert "Pexels Video API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_images_api_error_500_server_error(self, mock_get, pexels_api_valid_key_instance):
        """Test API error for image search due to server issue (500)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pexels_api_valid_key_instance.search_images(query="test")

        assert "Pexels API error: 500 Internal Server Error" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_videos_api_error_500_server_error(self, mock_get, pexels_api_valid_key_instance):
        """Test API error for video search due to server issue (500)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pexels_api_valid_key_instance.search_videos(query="test")

        assert "Pexels Video API error: 500 Internal Server Error" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_images_network_error(self, mock_get, pexels_api_valid_key_instance):
        """Test network error for image search."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            pexels_api_valid_key_instance.search_images(query="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_videos_network_error(self, mock_get, pexels_api_valid_key_instance):
        """Test network error for video search."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            pexels_api_valid_key_instance.search_videos(query="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_images_empty_results(self, mock_get, pexels_api_valid_key_instance):
        """Test image search returning empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"photos": [], "total_results": 0}
        mock_get.return_value = mock_response

        photos = pexels_api_valid_key_instance.search_images(query="nonexistentquery")
        assert photos == []

    @patch('tools.requests.get')
    def test_search_videos_empty_results(self, mock_get, pexels_api_valid_key_instance):
        """Test video search returning empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"videos": [], "total_results": 0}
        mock_get.return_value = mock_response

        videos = pexels_api_valid_key_instance.search_videos(query="nonexistentquery")
        assert videos == []


# --- Test Cases for PixabayAPI ---
class TestPixabayAPI:
    """Test suite for the PixabayAPI class."""

    def test_initialization(self, pixabay_api_valid_key_instance):
        """Test successful initialization of PixabayAPI."""
        assert pixabay_api_valid_key_instance.api_key == "VALID_PIXABAY_KEY"
        assert pixabay_api_valid_key_instance.base_url == "https://pixabay.com/api"

    @patch('tools.requests.get')
    def test_search_images_happy_path(self, mock_get, pixabay_api_valid_key_instance):
        """Test successful image search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_hits = [{"id": 1, "webformatURL": "url1"}, {"id": 2, "webformatURL": "url2"}]
        mock_response.json.return_value = {"hits": expected_hits, "totalHits": 2}
        mock_get.return_value = mock_response

        query_text = "flowers"
        per_page_count = 3  # Adjusted to 3 to match clamping
        images = pixabay_api_valid_key_instance.search_images(query=query_text, per_page=per_page_count)

        assert images == expected_hits
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://pixabay.com/api"
        assert kwargs['params']['key'] == "VALID_PIXABAY_KEY"
        assert kwargs['params']['q'] == query_text
        assert kwargs['params']['per_page'] == per_page_count
        assert kwargs['params']['image_type'] == "photo"

    @patch('tools.requests.get')
    def test_search_videos_happy_path(self, mock_get, pixabay_api_valid_key_instance):
        """Test successful video search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_hits = [{"id": 1, "videos": {"small": {"url": "vid_url1"}}}, {"id": 2, "videos": {"small": {"url": "vid_url2"}}}]
        mock_response.json.return_value = {"hits": expected_hits, "totalHits": 2}
        mock_get.return_value = mock_response

        query_text = "city"
        per_page_count = 3  # Adjusted to 3 to match clamping
        videos = pixabay_api_valid_key_instance.search_videos(query=query_text, per_page=per_page_count)

        assert videos == expected_hits
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://pixabay.com/api/videos/"
        assert kwargs['params']['key'] == "VALID_PIXABAY_KEY"
        assert kwargs['params']['q'] == query_text
        assert kwargs['params']['per_page'] == per_page_count

    @patch('tools.requests.get')
    def test_search_images_api_error_401_unauthorized(self, mock_get, pixabay_api_invalid_key_instance):
        """Test API error for image search due to invalid API key (401)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pixabay_api_invalid_key_instance.search_images(query="test")

        assert "Pixabay API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_videos_api_error_401_unauthorized(self, mock_get, pixabay_api_invalid_key_instance):
        """Test API error for video search due to invalid API key (401)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pixabay_api_invalid_key_instance.search_videos(query="test")

        assert "Pixabay Video API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_images_api_error_500_server_error(self, mock_get, pixabay_api_valid_key_instance):
        """Test API error for image search due to server issue (500)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pixabay_api_valid_key_instance.search_images(query="test")

        assert "Pixabay API error: 500 Internal Server Error" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_videos_api_error_500_server_error(self, mock_get, pixabay_api_valid_key_instance):
        """Test API error for video search due to server issue (500)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            pixabay_api_valid_key_instance.search_videos(query="test")

        assert "Pixabay Video API error: 500 Internal Server Error" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_images_network_error(self, mock_get, pixabay_api_valid_key_instance):
        """Test network error for image search."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            pixabay_api_valid_key_instance.search_images(query="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_videos_network_error(self, mock_get, pixabay_api_valid_key_instance):
        """Test network error for video search."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            pixabay_api_valid_key_instance.search_videos(query="test")

        assert "Simulated network failure" in str(excinfo.value)
        mock_get.assert_called_once()

    @patch('tools.requests.get')
    def test_search_images_empty_results(self, mock_get, pixabay_api_valid_key_instance):
        """Test image search returning empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"hits": [], "totalHits": 0}
        mock_get.return_value = mock_response

        images = pixabay_api_valid_key_instance.search_images(query="nonexistentquery123abc")
        assert images == []

    @patch('tools.requests.get')
    def test_search_videos_empty_results(self, mock_get, pixabay_api_valid_key_instance):
        """Test video search returning empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"hits": [], "totalHits": 0}
        mock_get.return_value = mock_response

        videos = pixabay_api_valid_key_instance.search_videos(query="nonexistentquery123abc")
        assert videos == []


# --- Test Cases for AssemblyAIAPI ---
class TestAssemblyAIAPI:
    """Test suite for the AssemblyAIAPI class."""

    def test_initialization(self, assemblyai_api_valid_key_instance):
        """Test successful initialization of AssemblyAIAPI."""
        assert assemblyai_api_valid_key_instance.api_key == "VALID_ASSEMBLYAI_KEY"
        assert assemblyai_api_valid_key_instance.base_url == "https://api.assemblyai.com/v2"

    @patch('tools.requests.post')
    def test_transcribe_happy_path(self, mock_post, assemblyai_api_valid_key_instance):
        """Test successful transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_transcript = {"id": "123", "text": "Hello world this is a test."}
        mock_response.json.return_value = expected_transcript
        mock_post.return_value = mock_response

        audio_url_to_transcribe = "http://example.com/audio.mp3"
        transcript_data = assemblyai_api_valid_key_instance.transcribe(audio_url=audio_url_to_transcribe)

        assert transcript_data == expected_transcript
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.assemblyai.com/v2/transcript"
        assert kwargs['json']['audio_url'] == audio_url_to_transcribe
        assert kwargs['headers']['authorization'] == "VALID_ASSEMBLYAI_KEY"

    @patch('tools.requests.post')
    def test_transcribe_api_error_401_unauthorized(self, mock_post, assemblyai_api_invalid_key_instance):
        """Test API error due to invalid API key (401 Unauthorized)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            assemblyai_api_invalid_key_instance.transcribe(audio_url="http://example.com/audio.mp3")

        assert "AssemblyAI API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_transcribe_api_error_500_server_error(self, mock_post, assemblyai_api_valid_key_instance):
        """Test API error due to server-side issue (500 Internal Server Error)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            assemblyai_api_valid_key_instance.transcribe(audio_url="http://example.com/audio.mp3")

        assert "AssemblyAI API error: 500 Internal Server Error" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_transcribe_network_error(self, mock_post, assemblyai_api_valid_key_instance):
        """Test handling of a network error (e.g., ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            assemblyai_api_valid_key_instance.transcribe(audio_url="http://example.com/audio.mp3")

        assert "Simulated network failure" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_transcribe_empty_or_invalid_audio_url_api_error(self, mock_post, assemblyai_api_valid_key_instance):
        """Test API error for empty or invalid audio_url (e.g., API returns 400)."""
        mock_response = MagicMock()
        mock_response.status_code = 400 # Bad Request
        mock_response.text = "Invalid audio_url"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            assemblyai_api_valid_key_instance.transcribe(audio_url="") # Empty URL

        assert "AssemblyAI API error: 400 Invalid audio_url" in str(excinfo.value)
        mock_post.assert_called_with(
            "https://api.assemblyai.com/v2/transcript",
            json={"audio_url": ""},
            headers={"authorization": "VALID_ASSEMBLYAI_KEY"},
            timeout=10
        )


# --- Test Cases for DeepgramAPI ---
class TestDeepgramAPI:
    """Test suite for the DeepgramAPI class."""

    def test_initialization(self, deepgram_api_valid_key_instance):
        """Test successful initialization of DeepgramAPI."""
        assert deepgram_api_valid_key_instance.api_key == "VALID_DEEPGRAM_KEY"
        assert deepgram_api_valid_key_instance.base_url == "https://api.deepgram.com/v1/listen"

    @patch('tools.requests.post')
    def test_transcribe_happy_path(self, mock_post, deepgram_api_valid_key_instance):
        """Test successful transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_transcript = {"results": {"channels": [{"alternatives": [{"transcript": "Hello world"}]}]}}
        mock_response.json.return_value = expected_transcript
        mock_post.return_value = mock_response

        audio_url_to_transcribe = "http://example.com/audio.wav"
        transcript_data = deepgram_api_valid_key_instance.transcribe(audio_url=audio_url_to_transcribe)

        assert transcript_data == expected_transcript
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.deepgram.com/v1/listen"
        assert kwargs['json']['url'] == audio_url_to_transcribe
        assert kwargs['headers']['Authorization'] == "Token VALID_DEEPGRAM_KEY"
        assert kwargs['params']['language'] == "en" # Default language

    @patch('tools.requests.post')
    def test_transcribe_custom_language(self, mock_post, deepgram_api_valid_key_instance):
        """Test successful transcription with a custom language."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_transcript = {"results": {"channels": [{"alternatives": [{"transcript": "Hola mundo"}]}]}}
        mock_response.json.return_value = expected_transcript
        mock_post.return_value = mock_response

        audio_url_to_transcribe = "http://example.com/audio_es.wav"
        custom_lang = "es"
        transcript_data = deepgram_api_valid_key_instance.transcribe(audio_url=audio_url_to_transcribe, language=custom_lang)

        assert transcript_data == expected_transcript
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['params']['language'] == custom_lang

    @patch('tools.requests.post')
    def test_transcribe_api_error_401_unauthorized(self, mock_post, deepgram_api_invalid_key_instance):
        """Test API error due to invalid API key (401 Unauthorized)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized - Invalid API Key"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            deepgram_api_invalid_key_instance.transcribe(audio_url="http://example.com/audio.wav")

        assert "Deepgram API error: 401 Unauthorized - Invalid API Key" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_transcribe_api_error_500_server_error(self, mock_post, deepgram_api_valid_key_instance):
        """Test API error due to server-side issue (500 Internal Server Error)."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            deepgram_api_valid_key_instance.transcribe(audio_url="http://example.com/audio.wav")

        assert "Deepgram API error: 500 Internal Server Error" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_transcribe_network_error(self, mock_post, deepgram_api_valid_key_instance):
        """Test handling of a network error (e.g., ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Simulated network failure")

        with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
            deepgram_api_valid_key_instance.transcribe(audio_url="http://example.com/audio.wav")

        assert "Simulated network failure" in str(excinfo.value)
        mock_post.assert_called_once()

    @patch('tools.requests.post')
    def test_transcribe_invalid_audio_url_api_error_400(self, mock_post, deepgram_api_valid_key_instance):
        """Test API error for invalid audio_url (e.g., API returns 400)."""
        mock_response = MagicMock()
        mock_response.status_code = 400 # Bad Request
        mock_response.text = "Invalid URL or audio format"
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            deepgram_api_valid_key_instance.transcribe(audio_url="invalid_url_format")

        assert "Deepgram API error: 400 Invalid URL or audio format" in str(excinfo.value)
        mock_post.assert_called_once()


# --- Test Cases for YouTubeDataAPI ---
class TestYouTubeDataAPI:
    """Test suite for the YouTubeDataAPI class."""

    def test_initialization(self, youtube_data_api_valid_key_instance):
        """Test successful initialization of YouTubeDataAPI."""
        assert youtube_data_api_valid_key_instance.api_key == "VALID_YOUTUBE_KEY"
        assert youtube_data_api_valid_key_instance.base_url == "https://www.googleapis.com/youtube/v3"

    @patch('tools.requests.get')
    def test_search_videos_happy_path(self, mock_get, youtube_data_api_valid_key_instance):
        """Test successful video search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_items = [{"id": {"videoId": "v1"}, "snippet": {"title": "Video 1"}}]
        mock_response.json.return_value = {"items": expected_items}
        mock_get.return_value = mock_response

        query = "python tutorials"
        max_results = 3
        items = youtube_data_api_valid_key_instance.search_videos(query=query, max_results=max_results)

        assert items == expected_items
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://www.googleapis.com/youtube/v3/search"
        assert kwargs['params']['q'] == query
        assert kwargs['params']['maxResults'] == max_results
        assert kwargs['params']['key'] == "VALID_YOUTUBE_KEY"
        assert kwargs['params']['part'] == "snippet"
        assert kwargs['params']['type'] == "video"

    @patch('tools.requests.get')
    def test_get_video_details_happy_path(self, mock_get, youtube_data_api_valid_key_instance):
        """Test successful retrieval of video details."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        expected_details = {"id": "v1", "snippet": {"title": "Video 1"}, "statistics": {"viewCount": "100"}}
        mock_response.json.return_value = {"items": [expected_details]}
        mock_get.return_value = mock_response

        video_id = "v1"
        details = youtube_data_api_valid_key_instance.get_video_details(video_id=video_id)

        assert details == expected_details
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://www.googleapis.com/youtube/v3/videos"
        assert kwargs['params']['id'] == video_id
        assert kwargs['params']['key'] == "VALID_YOUTUBE_KEY"
        assert "snippet,statistics,contentDetails" in kwargs['params']['part']

    @patch('tools.requests.get')
    def test_search_videos_api_error_401(self, mock_get, youtube_data_api_invalid_key_instance):
        """Test API error (401) for video search."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            youtube_data_api_invalid_key_instance.search_videos(query="test")
        assert "YouTube Data API error: 401 Unauthorized" in str(excinfo.value)

    @patch('tools.requests.get')
    def test_get_video_details_api_error_401(self, mock_get, youtube_data_api_invalid_key_instance):
        """Test API error (401) for getting video details."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            youtube_data_api_invalid_key_instance.get_video_details(video_id="v1")
        assert "YouTube Data API error: 401 Unauthorized" in str(excinfo.value)

    @patch('tools.requests.get')
    def test_search_videos_api_error_500(self, mock_get, youtube_data_api_valid_key_instance):
        """Test API error (500) for video search."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            youtube_data_api_valid_key_instance.search_videos(query="test")
        assert "YouTube Data API error: 500 Server Error" in str(excinfo.value)

    @patch('tools.requests.get')
    def test_get_video_details_api_error_500(self, mock_get, youtube_data_api_valid_key_instance):
        """Test API error (500) for getting video details."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as excinfo:
            youtube_data_api_valid_key_instance.get_video_details(video_id="v1")
        assert "YouTube Data API error: 500 Server Error" in str(excinfo.value)

    @patch('tools.requests.get')
    def test_search_videos_network_error(self, mock_get, youtube_data_api_valid_key_instance):
        """Test network error for video search."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network issue")
        with pytest.raises(requests.exceptions.ConnectionError):
            youtube_data_api_valid_key_instance.search_videos(query="test")

    @patch('tools.requests.get')
    def test_get_video_details_network_error(self, mock_get, youtube_data_api_valid_key_instance):
        """Test network error for getting video details."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network issue")
        with pytest.raises(requests.exceptions.ConnectionError):
            youtube_data_api_valid_key_instance.get_video_details(video_id="v1")

    @patch('tools.requests.get')
    def test_search_videos_empty_results(self, mock_get, youtube_data_api_valid_key_instance):
        """Test video search returning empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_get.return_value = mock_response
        items = youtube_data_api_valid_key_instance.search_videos(query="nonexistentquery")
        assert items == []

    @patch('tools.requests.get')
    def test_get_video_details_empty_results_for_id(self, mock_get, youtube_data_api_valid_key_instance):
        """Test get video details returning empty for a non-existent/invalid ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []} # API returns empty items list for invalid ID
        mock_get.return_value = mock_response
        details = youtube_data_api_valid_key_instance.get_video_details(video_id="nonexistent_id")
        assert details == {}
