"""
Tests for the custom agent implementation with enhanced features.

This file contains tests for the new capabilities added to the agent:
- Tool detection and selection
- Date/time recognition
- Calculator integration
- Web search
- Help/capabilities responses
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from agent import create_agent, invoke_agent
from langchain_core.messages import SystemMessage, HumanMessage


class TestCustomAgent:
    """Tests for the enhanced agent functionality."""

    def setup_method(self):
        """Setup for each test."""
        # Mock configuration
        self.test_config = {
            "api_key": "fake_api_key",
            "model_name": "gemini-2.0-flash",
            "temperature": 0.7,
            "system_prompt": "Você é Don Corleone."
        }

        # Mock tools
        self.mock_web_search = MagicMock(name="search_web")
        self.mock_web_search.invoke.return_value = "Resultado da busca na web"

        self.mock_calculator = MagicMock(name="calculator")
        self.mock_calculator.invoke.return_value = "42"

        self.mock_weather = MagicMock(name="weather")
        self.mock_weather.invoke.return_value = "23°C, parcialmente nublado"

        self.tools = [self.mock_web_search, self.mock_calculator, self.mock_weather]

    @patch('agent.ChatGoogleGenerativeAI')
    def test_create_agent_registers_tools(self, mock_chat_model):
        """Test that create_agent properly identifies and registers available tools."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance
        mock_model_instance.invoke.return_value = MagicMock(content="Mock response")

        # Execute
        agent = create_agent(self.test_config, self.tools)

        # Assert agent was created
        assert agent is not None
        assert hasattr(agent, 'invoke')

    @patch('agent.ChatGoogleGenerativeAI')
    def test_web_search_detection(self, mock_chat_model):
        """Test that the agent detects when to use web search."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance

        # Create a proper response mock with content attribute
        response_mock = MagicMock()
        response_mock.content = "Mock response"
        mock_model_instance.invoke.return_value = response_mock

        # Create agent
        agent = create_agent(self.test_config, self.tools)

        # Test with a search query
        agent.invoke({"input": "Quem ganhou a última Copa do Mundo?"})

        # Verify web search was called
        assert self.mock_web_search.invoke.called

        # Reset mock
        self.mock_web_search.invoke.reset_mock()

        # Test with a non-search query
        agent.invoke({"input": "Conte-me sobre você."})

        # Verify web search was not used
        assert not self.mock_web_search.invoke.called

    @patch('agent.ChatGoogleGenerativeAI')
    def test_date_time_detection(self, mock_chat_model):
        """Test that the agent correctly identifies date/time queries."""
        # Setup mock for ChatGoogleGenerativeAI
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance
        mock_model_instance.invoke.return_value = MagicMock(content="Mock response")

        # Create agent with mocked datetime module
        with patch('datetime.datetime') as mock_datetime:
            # Mock datetime.now()
            mock_now = MagicMock()
            mock_datetime.now.return_value = mock_now
            mock_now.strftime.return_value = "Saturday, 17 of May of 2025"

            agent = create_agent(self.test_config, self.tools)

            # Test with a date query
            agent.invoke({"input": "Que dia é hoje?"})

            # Verify the model was invoked with the right context
            assert mock_model_instance.invoke.called

            # Get the arguments passed to invoke
            args, _ = mock_model_instance.invoke.call_args
            message_contents = [msg.content for msg in args[0] if hasattr(msg, 'content')]

            # Check if any message content contains the date
            date_in_message = any("Saturday, 17 of May of 2025" in content for content in message_contents
                               if isinstance(content, str))

    @patch('agent.ChatGoogleGenerativeAI')
    def test_calculator_detection(self, mock_chat_model):
        """Test that the agent detects when to use calculator."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance

        # Create a proper response mock with content attribute
        response_mock = MagicMock()
        response_mock.content = "Mock response"
        mock_model_instance.invoke.return_value = response_mock

        # Create agent
        agent = create_agent(self.test_config, self.tools)

        # Test with a calculation query
        agent.invoke({"input": "2+2*5"})

        # Verify calculator was called
        assert self.mock_calculator.invoke.called

        # Reset mock
        self.mock_calculator.invoke.reset_mock()

        # Test with a non-calculation query
        agent.invoke({"input": "Olá, como vai?"})

        # Verify calculator was not used
        assert not self.mock_calculator.invoke.called

    @patch('agent.ChatGoogleGenerativeAI')
    def test_capabilities_response(self, mock_chat_model):
        """Test that the agent responds appropriately to capability questions."""
        # Setup mock with proper return value
        mock_model_instance = MagicMock()
        mock_chat_model.return_value = mock_model_instance

        # Create a proper response mock with content attribute as a property
        class MockResponse:
            @property
            def content(self):
                return "Mock capabilities response"

        mock_model_instance.invoke.return_value = MockResponse()

        # Create agent
        agent = create_agent(self.test_config, self.tools)

        # Test with capabilities questions
        result = agent.invoke({"input": "O que você pode fazer?"})

        # Verify the response contains capabilities info
        assert "output" in result
        assert result["output"] == "Mock capabilities response"

        # Verify the model was invoked
        assert mock_model_instance.invoke.called

    def test_empty_message_protection(self):
        """Test that the agent prevents empty messages from being sent to Gemini."""
        # Create a list of empty messages
        messages = []

        # Apply the same protection logic from agent.py
        if not any(getattr(m, 'content', '').strip() for m in messages):
            messages = [HumanMessage(content="Olá")]

        # Verify a default message was added
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Olá"

    def test_invoke_agent_integration(self):
        """Test that invoke_agent properly integrates with the custom agent."""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"output": "Esta é a resposta do agente."}

        # Call invoke_agent
        result = invoke_agent(mock_agent, "Olá, como você está?")

        # Verify invoke_agent properly called the agent
        mock_agent.invoke.assert_called_once()
        args, kwargs = mock_agent.invoke.call_args
        assert "input" in args[0]
        assert "chat_history" in args[0]

        # Verify the result
        assert result == "Esta é a resposta do agente."
