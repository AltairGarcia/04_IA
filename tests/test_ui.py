"""
Testes unitários para o módulo ui.py
"""

import pytest
from unittest.mock import patch, MagicMock, call
import sys
import time
import io
from ui import (
    Colors, print_colored, print_welcome, print_help, print_error,
    print_success, print_agent_response, show_thinking_animation,
    get_user_input, clear_screen
)


class TestUI:
    """Testes para as funções de interface do usuário."""

    def test_colors_constants(self):
        """Testa se as constantes de cores estão definidas corretamente."""
        assert Colors.RESET == "\033[0m"
        assert Colors.BOLD == "\033[1m"
        assert Colors.RED == "\033[91m"
        assert Colors.GREEN == "\033[92m"
        assert Colors.YELLOW == "\033[93m"
        assert Colors.BLUE == "\033[94m"
        assert Colors.MAGENTA == "\033[95m"
        assert Colors.CYAN == "\033[96m"
        assert Colors.WHITE == "\033[97m"

    @patch('builtins.print')
    def test_print_colored_regular(self, mock_print):
        """Testa a função print_colored com texto normal."""
        print_colored("Teste", Colors.BLUE)
        mock_print.assert_called_once_with(f"{Colors.BLUE}Teste{Colors.RESET}")

    @patch('builtins.print')
    def test_print_colored_bold(self, mock_print):
        """Testa a função print_colored com texto em negrito."""
        print_colored("Teste", Colors.RED, bold=True)
        mock_print.assert_called_once_with(f"{Colors.BOLD}{Colors.RED}Teste{Colors.RESET}")

    @patch('ui.print_colored')
    def test_print_welcome(self, mock_print_colored):
        """Testa a função print_welcome."""
        with patch('builtins.print') as mock_print:
            print_welcome()
            # Verificar que foram feitas várias chamadas de impressão
            assert mock_print.call_count >= 3
            # Verificar que print_colored foi chamado
            assert mock_print_colored.call_count == 2

            # Verificar uma chamada específica
            mock_print_colored.assert_any_call("\nFarei uma oferta que você não poderá recusar...", Colors.CYAN)

    @patch('ui.print_colored')
    def test_print_help(self, mock_print_colored):
        """Testa a função print_help."""
        print_help()
        # Verificar que print_colored foi chamado várias vezes
        assert mock_print_colored.call_count == 5

        # Verificar algumas chamadas específicas
        mock_print_colored.assert_any_call("\n--- Comandos Disponíveis ---", Colors.YELLOW, bold=True)
        mock_print_colored.assert_any_call("ajuda      - Mostra esta mensagem de ajuda", Colors.WHITE)

    @patch('ui.print_colored')
    def test_print_error(self, mock_print_colored):
        """Testa a função print_error."""
        print_error("Mensagem de erro")
        mock_print_colored.assert_called_once_with("\nErro: Mensagem de erro", Colors.RED, bold=True)

    @patch('ui.print_colored')
    def test_print_success(self, mock_print_colored):
        """Testa a função print_success."""
        print_success("Operação concluída")
        mock_print_colored.assert_called_once_with("\nOperação concluída", Colors.GREEN)

    @patch('ui.print_colored')
    def test_print_agent_response(self, mock_print_colored):
        """Testa a função print_agent_response."""
        # Teste com persona padrão
        print_agent_response("Resposta do agente")
        mock_print_colored.assert_has_calls([
            call("\nDon Corleone:", Colors.YELLOW, bold=True),
            call("Resposta do agente\n", Colors.WHITE)
        ])

        mock_print_colored.reset_mock()

        # Teste com persona personalizada
        print_agent_response("Resposta do agente", persona_name="Sherlock Holmes")
        mock_print_colored.assert_has_calls([
            call("\nSherlock Holmes:", Colors.YELLOW, bold=True),
            call("Resposta do agente\n", Colors.WHITE)
        ])

    @patch('time.time')
    @patch('time.sleep')
    @patch('sys.stdout')
    def test_show_thinking_animation(self, mock_stdout, mock_sleep, mock_time):
        """Testa a função show_thinking_animation."""
        # Configurar mock para time.time() retornar valores incrementais
        # Garantir que o loop execute pelo menos 2 iterações + limpeza final
        mock_time.side_effect = [0, 0.3, 0.6, 0.9, 1.2]  # Simular passagem do tempo

        # Chamar a função com duração curta para o teste
        show_thinking_animation(duration=1.0, message="Processando")

        # Verificar que sys.stdout.write foi chamado pelo menos 2 vezes durante o loop + 1 vez para limpar
        assert mock_stdout.write.call_count >= 3

        # Verificar uma chamada de animação (os pontos podem variar dependendo do valor de i)
        # Em vez de verificar o texto exato, vamos verificar se alguma chamada começa com o padrão correto
        # Isso é mais robusto porque o número de pontos pode variar
        write_calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        assert any(call.startswith("\rProcessando") for call in write_calls)

        # Verificar última chamada (limpar a linha)
        # Como o comprimento da string de espaços pode variar, vamos usar uma abordagem mais flexível
        # ao invés de verificar um valor fixo de 13 espaços
        assert any(call.startswith("\r ") and call.endswith("\r") for call in write_calls)

        # Verificar que sys.stdout.flush foi chamado
        assert mock_stdout.flush.call_count >= 2

    @patch('ui.print_colored')
    @patch('builtins.input', return_value="Teste de entrada")
    def test_get_user_input(self, mock_input, mock_print_colored):
        """Testa a função get_user_input."""
        # Teste com prompt padrão
        result = get_user_input()
        mock_print_colored.assert_called_once_with("Você: ", Colors.GREEN, bold=True)
        assert result == "Teste de entrada"

        mock_print_colored.reset_mock()

        # Teste com prompt personalizado
        result = get_user_input(prompt="Usuário")
        mock_print_colored.assert_called_once_with("Usuário: ", Colors.GREEN, bold=True)
        assert result == "Teste de entrada"

    @patch('builtins.print')
    def test_clear_screen(self, mock_print):
        """Testa a função clear_screen."""
        clear_screen()
        mock_print.assert_called_once_with("\033c", end="")
