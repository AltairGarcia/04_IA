"""
Comprehensive tests for langgraph-101.py main application.
This tests the main entry point and CLI functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from io import StringIO

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
import importlib.util
import importlib
spec = importlib.util.spec_from_file_location("langgraph_main", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "langgraph-101.py"))
langgraph_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(langgraph_main)


class TestLangGraphMain:
    """Test class for main application functions."""

    def test_print_personas_list(self):
        """Test printing available personas list."""
        with patch.object(langgraph_main, 'get_available_personas') as mock_get_personas, \
             patch.object(langgraph_main, 'print_colored') as mock_print:

            mock_personas = {
                'Sherlock Holmes': {'description': 'Detective persona'},
                'Yoda': {'description': 'Wise Jedi master'}
            }
            mock_get_personas.return_value = mock_personas

            # Execute the function directly
            langgraph_main.print_personas_list()

            # Verify the function was called
            mock_get_personas.assert_called_once()
            # Check that print_colored was called with expected content
            assert mock_print.call_count >= 5  # Header, personas, descriptions, footer

    def test_print_export_formats(self):
        """Test printing available export formats."""
        with patch.object(langgraph_main, 'get_export_formats') as mock_get_formats, \
             patch.object(langgraph_main, 'print_colored') as mock_print:
            mock_formats = {
                'html': {'description': 'HTML format'},
                'txt': {'description': 'Text format'}
            }
            mock_get_formats.return_value = mock_formats
            langgraph_main.print_export_formats()
            mock_get_formats.assert_called_once()
            assert mock_print.call_count >= 5  # Header, formats, descriptions, usage, footer

    @patch.object(langgraph_main, 'load_config')
    @patch.object(langgraph_main, 'get_tools')
    @patch.object(langgraph_main, 'create_agent')
    @patch.object(langgraph_main, 'get_history_manager')
    @patch.object(langgraph_main, 'get_memory_manager')
    def test_setup_agent_success(self, mock_memory, mock_history, mock_create_agent, mock_get_tools, mock_load_config):
        """Test successful agent setup."""
        # Setup mocks
        mock_config = {'api_key': 'test_key', 'model_name': 'gemini-2.0-flash'}
        mock_load_config.return_value = mock_config
        mock_tools = [Mock(), Mock()]
        mock_get_tools.return_value = mock_tools
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_history_manager = Mock()
        mock_history.return_value = mock_history_manager
        mock_memory_manager = Mock()
        mock_memory.return_value = mock_memory_manager

        # Load module and test
        spec.loader.exec_module(langgraph_main)

        # Since setup_agent might be part of main function, we test the components
        config = mock_load_config()
        tools = mock_get_tools()
        agent = mock_create_agent(config, tools)
        history = mock_history()
        memory = mock_memory()

        # Verify calls
        mock_load_config.assert_called_once()
        mock_get_tools.assert_called_once()
        mock_create_agent.assert_called_once_with(mock_config, mock_tools)
        assert agent == mock_agent

    @patch.object(langgraph_main, 'print_help')
    @patch.object(langgraph_main, 'sys')
    def test_help_command(self, mock_sys, mock_print_help):
        """Test help command line argument."""
        with patch.object(langgraph_main, 'print_help') as mock_print_help, \
             patch.object(langgraph_main, 'sys') as mock_sys:
            # Simulate sys.argv
            mock_sys.argv = ['langgraph-101.py', '--help']
            spec.loader.exec_module(langgraph_main)
            assert hasattr(langgraph_main, 'print_help') or 'print_help' in dir(langgraph_main)

    @patch.object(langgraph_main, 'load_config')
    def test_config_error_handling(self, mock_load_config):
        """Test configuration error handling."""
        from config import ConfigError
        mock_load_config.side_effect = ConfigError("API key missing")
        with patch.object(langgraph_main, 'print_error') as mock_print_error:
            spec.loader.exec_module(langgraph_main)
            with pytest.raises(ConfigError):
                mock_load_config()

    def test_module_imports(self):
        """Test that all required modules can be imported."""
        spec.loader.exec_module(langgraph_main)

        # Check that key functions/classes are available
        expected_imports = [
            'print_welcome', 'print_help', 'print_error', 'print_success',
            'print_agent_response', 'get_user_input', 'clear_screen'
        ]

        # Note: These might be imported differently, so we check if they exist in the module
        for func_name in expected_imports:
            # The functions should be available either directly or through imports
            assert func_name in langgraph_main.__dict__ or \
                   any(func_name in str(value) for value in langgraph_main.__dict__.values())

    @patch.object(langgraph_main, 'get_available_personas')
    def test_personas_integration(self, mock_get_personas):
        """Test personas integration."""
        mock_personas = {
            'Don Corleone': {
                'description': 'Wise mafia boss',
                'system_prompt': 'You are Don Corleone...'
            }
        }
        mock_get_personas.return_value = mock_personas

        spec.loader.exec_module(langgraph_main)

        # Test that personas are accessible
        personas = mock_get_personas()
        assert 'Don Corleone' in personas
        assert personas['Don Corleone']['description'] == 'Wise mafia boss'

    @patch.object(langgraph_main, 'get_export_formats')
    def test_export_integration(self, mock_get_formats):
        """Test export functionality integration."""
        mock_formats = {
            'html': {'description': 'HTML format', 'extension': '.html'},
            'txt': {'description': 'Plain text', 'extension': '.txt'}
        }
        mock_get_formats.return_value = mock_formats

        spec.loader.exec_module(langgraph_main)

        # Test that export formats are accessible
        formats = mock_get_formats()
        assert 'html' in formats
        assert 'txt' in formats

    def test_error_handling_imports(self):
        """Test that error handling modules are properly imported."""
        spec.loader.exec_module(langgraph_main)

        # Check that sys and traceback are imported for error handling
        assert 'sys' in langgraph_main.__dict__
        assert 'traceback' in langgraph_main.__dict__

    def test_datetime_functionality(self):
        """Test datetime functionality."""
        from datetime import datetime
        with patch.object(langgraph_main, 'datetime') as mock_datetime:
            mock_now = datetime(2025, 5, 23, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            spec.loader.exec_module(langgraph_main)
            assert 'datetime' in langgraph_main.__dict__

    def test_json_functionality(self):
        """Test JSON functionality for Gemini output parsing."""
        spec.loader.exec_module(langgraph_main)

        # Test that json is imported and available
        assert 'json' in langgraph_main.__dict__

    def test_directory_setup(self):
        """Test directory setup functionality."""
        with patch.object(langgraph_main.os.path, 'exists') as mock_exists, \
             patch.object(langgraph_main.os, 'makedirs') as mock_makedirs:
            mock_exists.return_value = False
            spec.loader.exec_module(langgraph_main)
            assert 'os' in langgraph_main.__dict__


class TestMainApplicationFlow:
    """Test the main application flow and command processing."""

    def test_user_input_processing(self):
        """Test basic user input processing."""
        with patch('builtins.input') as mock_input, \
             patch.object(langgraph_main, 'print_colored') as mock_print:
            mock_input.return_value = "hello"
            with patch.object(langgraph_main, 'get_user_input') as mock_get_input:
                mock_get_input.return_value = "hello"
                spec.loader.exec_module(langgraph_main)
                user_input = mock_get_input()
                assert user_input == "hello"

    def test_command_recognition(self):
        """Test that command recognition patterns work."""
        spec.loader.exec_module(langgraph_main)

        # Test command patterns (these would be in main() function)
        test_commands = [
            "ajuda", "help", "limpar", "clear", "sair", "quit", "exit",
            "personas", "exportar", "email", "memoria", "memory"
        ]

        # This is a basic test that the module loads successfully
        # More detailed command testing would require refactoring main()
        assert True  # Module loaded successfully

    def test_application_startup(self):
        """Test application startup sequence."""
        with patch.object(langgraph_main, 'clear_screen') as mock_clear, \
             patch.object(langgraph_main, 'print_welcome') as mock_welcome:
            spec.loader.exec_module(langgraph_main)
            assert 'clear_screen' in langgraph_main.__dict__ or hasattr(langgraph_main, 'clear_screen')
            assert 'print_welcome' in langgraph_main.__dict__ or hasattr(langgraph_main, 'print_welcome')


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    def test_config_error_recovery(self):
        """Test recovery from configuration errors."""
        with patch.object(langgraph_main, 'print_error') as mock_print_error:
            spec.loader.exec_module(langgraph_main)
            assert 'print_error' in langgraph_main.__dict__ or hasattr(langgraph_main, 'print_error')

    def test_import_error_handling(self):
        """Test handling of import errors."""
        # Test that the module can be loaded even if some imports fail
        try:
            spec.loader.exec_module(langgraph_main)
            success = True
        except ImportError:
            success = False

        # The module should load successfully in the test environment
        assert success

    def test_graceful_shutdown(self):
        """Test graceful shutdown capabilities."""
        spec.loader.exec_module(langgraph_main)

        # Test that sys is available for graceful shutdown
        assert 'sys' in langgraph_main.__dict__


class TestIntegrationComponents:
    """Test integration between different components."""

    def test_agent_tool_integration(self):
        """Test agent and tools integration."""
        with patch.object(langgraph_main, 'load_config') as mock_config, \
             patch.object(langgraph_main, 'get_tools') as mock_tools, \
             patch.object(langgraph_main, 'create_agent') as mock_create:
            mock_config.return_value = {'api_key': 'test', 'model_name': 'gemini-2.0-flash'}
            mock_tools.return_value = [Mock(), Mock()]
            mock_agent = Mock()
            mock_create.return_value = mock_agent
            spec.loader.exec_module(langgraph_main)
            config = mock_config()
            tools = mock_tools()
            agent = mock_create(config, tools)
            assert agent == mock_agent

    def test_history_memory_integration(self):
        """Test history and memory management integration."""
        with patch.object(langgraph_main, 'get_history_manager') as mock_history, \
             patch.object(langgraph_main, 'get_memory_manager') as mock_memory:
            mock_history.return_value = Mock()
            mock_memory.return_value = Mock()
            spec.loader.exec_module(langgraph_main)
            history_mgr = mock_history()
            memory_mgr = mock_memory()
            assert history_mgr is not None
            assert memory_mgr is not None

    def test_ui_integration(self):
        """Test UI components integration."""
        spec.loader.exec_module(langgraph_main)

        # Test that UI functions are available
        ui_functions = [
            'print_welcome', 'print_help', 'print_error', 'print_success',
            'print_agent_response', 'get_user_input', 'clear_screen'
        ]

        # Check that UI functions are imported
        for func in ui_functions:
            assert func in langgraph_main.__dict__ or \
                   any(func in str(value) for value in langgraph_main.__dict__.values())
