"""
Simple and focused tests for the agent features.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent import create_agent, invoke_agent


class TestAgentFeatures:
    """Tests for the core agent functionality."""

    def setup_method(self):
        """Setup basic mocks for each test."""
        self.config = {
            "api_key": "fake_api_key",
            "model_name": "gemini-2.0-flash",
            "temperature": 0.7,
            "system_prompt": "Você é Don Corleone."
        }

        self.mock_web_search = MagicMock(name="search_web")
        self.mock_web_search.invoke = MagicMock(return_value="Web search results")

        self.mock_calculator = MagicMock(name="calculator")
        self.mock_calculator.invoke = MagicMock(return_value="12")

        self.tools = [self.mock_web_search, self.mock_calculator]

    @patch('agent.ChatGoogleGenerativeAI')
    def test_agent_creation(self, mock_chat):
        """Test that an agent can be created."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.invoke = MagicMock(return_value=MagicMock(content="Response"))
        mock_chat.return_value = mock_instance

        # Create the agent
        agent = create_agent(self.config, self.tools)

        # Verify agent was created with an invoke method
        assert agent is not None
        assert hasattr(agent, 'invoke')

    @patch('agent.ChatGoogleGenerativeAI')
    def test_invoke_function(self, mock_chat):
        """Test that the invoke_agent function works correctly."""
        # Setup mock agent
        mock_agent = MagicMock()
        mock_agent.invoke = MagicMock(return_value={"output": "Agent response"})

        # Call invoke_agent
        result = invoke_agent(mock_agent, "Hello")

        # Verify result
        assert result == "Agent response"

        # Verify agent.invoke was called
        mock_agent.invoke.assert_called_once()

    @patch('agent.ChatGoogleGenerativeAI')
    def test_agent_empty_input(self, mock_chat):
        """Test that the agent handles empty input properly."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.invoke = MagicMock(return_value=MagicMock(content="Response to empty"))
        mock_chat.return_value = mock_instance

        # Create agent
        agent = create_agent(self.config, self.tools)

        # Call with empty input
        result = agent.invoke({"input": ""})

        # Should get a response asking for input
        assert "output" in result
        assert "Por favor" in result["output"]
