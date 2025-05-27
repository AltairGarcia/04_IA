"""
Testes unitários para o módulo config.py
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from config import load_config, get_system_prompt, get_available_personas, ConfigError


class TestConfig:
    """Testes para as funções no módulo config.py"""

    @patch('config.load_dotenv')
    @patch('config.os.getenv')
    @patch('config.get_persona_by_name')
    def test_load_config_with_required_keys(self, mock_get_persona, mock_getenv, mock_load_dotenv):
        """Testa o carregamento de configuração com todas as chaves obrigatórias."""
        # Configurar os mocks
        def getenv_side_effect(key, default=None):
            env_vars = {
                "API_KEY": "test_api_key",
                "TAVILY_API_KEY": "test_tavily_key",
                "MODEL_NAME": "test-model",
                "TEMPERATURE": "0.5",
                "PERSONA": "Test Persona",
                "SAVE_HISTORY": "true",
                "MAX_HISTORY": "20"
            }
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect

        # Mock para a persona
        mock_persona = MagicMock()
        mock_persona.name = "Test Persona"
        mock_get_persona.return_value = mock_persona

        # Chamar a função sendo testada
        config = load_config()

        # Verificar se load_dotenv foi chamado
        mock_load_dotenv.assert_called_once_with(encoding='utf-16-le')

        # Verificar se a configuração foi carregada corretamente
        assert config["api_key"] == "test_api_key"
        assert config["tavily_api_key"] == "test_tavily_key"
        assert config["model_name"] == "test-model"
        assert config["temperature"] == 0.5
        assert config["current_persona"] == mock_persona
        assert config["save_history"] is True
        assert config["max_history"] == 20

    @patch('config.load_dotenv')
    @patch('config.os.getenv')
    def test_load_config_missing_api_key(self, mock_getenv, mock_load_dotenv):
        """Testa se ConfigError é levantado quando API_KEY está ausente."""
        # Configurar o mock para retornar None para API_KEY
        mock_getenv.side_effect = lambda key, default=None: default if key == "API_KEY" else "value"

        # Verificar se a exceção é levantada
        with pytest.raises(ConfigError, match="API_KEY.*não encontrada"):
            load_config()

    @patch('config.load_dotenv')
    @patch('config.os.getenv')
    def test_load_config_missing_tavily_key(self, mock_getenv, mock_load_dotenv):
        """Testa se ConfigError é levantado quando TAVILY_API_KEY está ausente."""
        # Configurar o mock para retornar valor para API_KEY mas None para TAVILY_API_KEY
        def getenv_side_effect(key, default=None):
            if key == "API_KEY":
                return "test_api_key"
            elif key == "TAVILY_API_KEY":
                return None
            return default

        mock_getenv.side_effect = getenv_side_effect

        # Verificar se a exceção é levantada
        with pytest.raises(ConfigError, match="TAVILY_API_KEY não encontrada"):
            load_config()

    @patch('config.load_dotenv')
    @patch('config.os.getenv')
    @patch('config.get_persona_by_name')
    def test_load_config_with_optional_keys(self, mock_get_persona, mock_getenv, mock_load_dotenv):
        """Testa o carregamento de configuração incluindo chaves opcionais."""
        # Configurar os mocks
        def getenv_side_effect(key, default=None):
            env_vars = {
                "API_KEY": "test_api_key",
                "TAVILY_API_KEY": "test_tavily_key",
                "OPENWEATHER_API_KEY": "test_weather_key",
                "MODEL_NAME": "test-model",
                "TEMPERATURE": "0.5",
                "PERSONA": "Test Persona",
                "SAVE_HISTORY": "true",
                "MAX_HISTORY": "20"
            }
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect

        # Mock para a persona
        mock_persona = MagicMock()
        mock_persona.name = "Test Persona"
        mock_get_persona.return_value = mock_persona

        # Chamar a função sendo testada
        config = load_config()

        # Verificar se a chave opcional foi adicionada
        assert "openweather_api_key" in config
        assert config["openweather_api_key"] == "test_weather_key"

    @patch('config.os.getenv')
    @patch('config.get_persona_by_name')
    def test_get_system_prompt_with_custom_prompt(self, mock_get_persona, mock_getenv):
        """Testa a obtenção do prompt do sistema quando um prompt personalizado está definido."""
        # Configurar mock para retornar um prompt personalizado
        mock_getenv.return_value = "Este é um prompt personalizado"

        # Chamar a função sendo testada
        prompt = get_system_prompt()

        # Verificar se o prompt personalizado foi retornado
        assert prompt == "Este é um prompt personalizado"

        # Verificar que get_persona_by_name não foi chamado
        mock_get_persona.assert_not_called()

    @patch('config.os.getenv')
    @patch('config.get_persona_by_name')
    def test_get_system_prompt_with_persona(self, mock_get_persona, mock_getenv):
        """Testa a obtenção do prompt do sistema de uma persona fornecida."""
        # Configurar mock para os.getenv retornar None para SYSTEM_PROMPT
        mock_getenv.return_value = None

        # Configurar mock para a persona
        mock_persona = MagicMock()
        mock_persona.get_system_prompt.return_value = "Prompt da persona de teste"

        # Fornecendo uma persona diretamente
        prompt = get_system_prompt(persona=mock_persona)

        # Verificar se o prompt da persona foi retornado
        assert prompt == "Prompt da persona de teste"

        # Verificar que get_persona_by_name não foi chamado
        mock_get_persona.assert_not_called()

    @patch('config.os.getenv')
    @patch('config.get_persona_by_name')
    def test_get_system_prompt_from_environment(self, mock_get_persona, mock_getenv):
        """Testa a obtenção do prompt do sistema através da persona especificada no ambiente."""
        # Configurar mocks
        # Primeiro os.getenv retorna None para SYSTEM_PROMPT, depois retorna nome da persona
        mock_getenv.side_effect = lambda key, default=None: None if key == "SYSTEM_PROMPT" else "Persona Ambiente"

        # Mock para a persona
        mock_persona = MagicMock()
        mock_persona.get_system_prompt.return_value = "Prompt da persona do ambiente"
        mock_get_persona.return_value = mock_persona

        # Chamar a função sendo testada
        prompt = get_system_prompt()

        # Verificar se o prompt foi obtido da persona do ambiente
        assert prompt == "Prompt da persona do ambiente"

        # Verificar que get_persona_by_name foi chamado com o nome correto
        mock_get_persona.assert_called_once_with("Persona Ambiente")

    @patch('config.get_all_personas')
    def test_get_available_personas(self, mock_get_all_personas):
        """Testa a obtenção da lista de personas disponíveis."""
        # Configurar mock para retornar uma lista de personas
        persona1 = MagicMock()
        persona1.name = "Persona 1"
        persona1.get_info.return_value = {"description": "Descrição 1"}

        persona2 = MagicMock()
        persona2.name = "Persona 2"
        persona2.get_info.return_value = {"description": "Descrição 2"}

        mock_get_all_personas.return_value = [persona1, persona2]

        # Chamar a função sendo testada
        personas = get_available_personas()

        # Verificar se a função retornou o dicionário esperado
        assert personas == {
            "Persona 1": {"description": "Descrição 1"},
            "Persona 2": {"description": "Descrição 2"}
        }

        # Verificar que get_all_personas foi chamado
        mock_get_all_personas.assert_called_once()
