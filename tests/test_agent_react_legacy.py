"""
Testes unitários para o módulo agent.py
"""

import pytest
from unittest.mock import MagicMock, patch

from agent import create_agent, invoke_agent
from langchain_core.messages import SystemMessage, HumanMessage


class TestAgent:
    """Testes para as funções de gerenciamento do agente."""

    @patch('agent.ChatGoogleGenerativeAI')
    @patch('agent.create_react_agent')
    def test_create_agent(self, mock_create_react_agent, mock_chat_model):
        """Testa a função create_agent."""
        # Configurar os mocks
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance

        mock_agent = MagicMock()
        mock_create_react_agent.return_value = mock_agent

        # Configuração de teste
        test_config = {
            "api_key": "fake_api_key",
            "model_name": "gemini-2.0-flash",
            "temperature": 0.7,
            "system_prompt": "Você é um assistente útil."
        }

        test_tools = [MagicMock(), MagicMock()]

        # Chamar a função sendo testada
        result = create_agent(test_config, test_tools)

        # Verificações
        mock_chat_model.assert_called_once_with(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key="fake_api_key"
        )

        # Verificar se create_react_agent foi chamado com os parâmetros corretos
        mock_create_react_agent.assert_called_once()

        # Verificar os argumentos passados para create_react_agent
        args, kwargs = mock_create_react_agent.call_args
        assert args[0] == mock_model_instance  # modelo
        assert kwargs["tools"] == test_tools  # ferramentas

        # Verificar se o prompt do sistema foi criado corretamente
        system_message = kwargs["prompt"]
        assert isinstance(system_message, SystemMessage)
        assert system_message.content == "Você é um assistente útil."

        # Verificar o resultado
        assert result == mock_agent

    def test_invoke_agent_success(self):
        """Testa a invocação bem-sucedida do agente."""
        # Criar um mock para o agente
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Esta é a resposta do agente."}

        # Chamar a função sendo testada
        result = invoke_agent(mock_agent, "Olá, como você está?")

        # Verificações - nota: atual implementação adiciona a mensagem do usuário ao histórico
        mock_agent.invoke.assert_called_once_with({
            "input": "Olá, como você está?",
            "chat_history": [HumanMessage(content="Olá, como você está?")]
        })

        assert result == "Esta é a resposta do agente."

    def test_invoke_agent_with_history(self):
        """Testa a invocação do agente com histórico de chat."""
        # Criar um mock para o agente
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Esta é a resposta com histórico."}

        # Criar histórico de teste
        test_history = [
            HumanMessage(content="Mensagem anterior do usuário"),
            SystemMessage(content="Resposta anterior do sistema")
        ]

        # Chamar a função sendo testada
        result = invoke_agent(mock_agent, "Pergunta de follow-up", test_history)

        # Verificações - nota: atual implementação adiciona a mensagem atual ao histórico
        expected_history = test_history + [HumanMessage(content="Pergunta de follow-up")]
        mock_agent.invoke.assert_called_once_with({
            "input": "Pergunta de follow-up",
            "chat_history": expected_history
        })

        assert result == "Esta é a resposta com histórico."

    def test_invoke_agent_no_output(self):
        """Testa a invocação do agente quando a resposta não contém a chave 'output'."""
        # Criar um mock para o agente
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"some_other_key": "value"}

        # Chamar a função sendo testada
        result = invoke_agent(mock_agent, "Pergunta sem output")

        # Verificações
        assert result == "Não foi possível gerar uma resposta. Tente novamente."

    def test_invoke_agent_exception(self):
        """Testa o comportamento quando a invocação do agente lança uma exceção."""
        # Criar um mock para o agente que lança exceção
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("Erro de teste")

        # Chamar a função sendo testada
        result = invoke_agent(mock_agent, "Pergunta com erro")

        # Verificações
        assert result.startswith("Erro ao processar sua solicitação: Erro de teste")
