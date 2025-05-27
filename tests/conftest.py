"""
Configurações compartilhadas para os testes unitários.

Este arquivo contém fixtures do pytest que podem ser reutilizadas em vários testes.
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock


@pytest.fixture
def mock_api_keys():
    """Fixture que retorna um dicionário com chaves de API simuladas."""
    return {
        "API_KEY": "fake_gemini_api_key",
        "TAVILY_API_KEY": "fake_tavily_api_key",
        "OPENWEATHER_API_KEY": "fake_weather_api_key"
    }


@pytest.fixture
def temp_file():
    """Fixture que cria um arquivo temporário e o remove após o teste."""
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name

    yield temp_path

    # Limpeza após o teste
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_persona():
    """Fixture que cria uma persona simulada."""
    persona = MagicMock()
    persona.name = "Test Persona"
    persona.description = "A test persona"
    persona.system_prompt = "You are a test persona."
    persona.get_system_prompt.return_value = "You are a test persona."
    persona.get_info.return_value = {
        "name": "Test Persona",
        "description": "A test persona"
    }
    return persona


@pytest.fixture
def mock_selectbox():
    """Fixture para simular streamlit.selectbox."""
    return MagicMock()


@pytest.fixture
def mock_history():
    """Fixture para simular histórico de conversa."""
    history = MagicMock()
    history.messages = []
    history.get_messages.return_value = []
    return history


@pytest.fixture
def mock_get_tools():
    """Fixture para simular get_tools function."""
    return MagicMock(return_value=[MagicMock(), MagicMock()])
