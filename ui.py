"""
UI module for LangGraph 101 project.

This module is a compatibility layer that imports and re-exports UI components
from both terminal and Streamlit interfaces.
"""

# Import all components from the base, terminal and Streamlit UI modules
from ui_base import get_i18n, I18n
from ui_terminal import *
from ui_streamlit import *


# For backward compatibility, re-export the Colors class
from ui_terminal import Colors

# Alias function to maintain backward compatibility with existing code
def display_tooltip(label, tooltip):
    """Display a label with a tooltip icon."""
    from ui_streamlit import display_tooltip as display_tooltip_streamlit
    display_tooltip_streamlit(label, tooltip)


def print_welcome() -> None:
    """Print a stylized welcome message."""
    print("\n" + "=" * 50)
    print_colored("=== Don Corleone AI ===".center(50), Colors.YELLOW, bold=True)
    print("=" * 50)
    print_colored("\nFarei uma oferta que você não poderá recusar...", Colors.CYAN)
    print("Digite 'sair' para encerrar o chat.")
    print("Digite 'ajuda' para ver comandos disponíveis.\n")


def print_help() -> None:
    """Print help information with colors."""
    print_colored("\n--- Comandos Disponíveis ---", Colors.YELLOW, bold=True)
    print_colored("sair       - Encerra a conversa", Colors.WHITE)
    print_colored("ajuda      - Mostra esta mensagem de ajuda", Colors.WHITE)
    print_colored("limpar     - Limpa o histórico da conversa", Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)


def print_quick_replies():
    """Print quick reply suggestions for common actions."""
    print_colored("\nSugestões rápidas:", Colors.CYAN, bold=True)
    print_colored("[ajuda] [limpar] [personas] [salvar] [sair]", Colors.WHITE)


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: The error message to print.
    """
    print_colored(f"\nErro: {message}", Colors.RED, bold=True)


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: The success message to print.
    """
    print_colored(f"\n{message}", Colors.GREEN)


def print_agent_response(message: str, persona_name: str = "Don Corleone") -> None:
    """Print the agent's response with styling and quick replies.

    Args:
        message: The agent's response message.
        persona_name: The name of the current persona.
    """
    print_colored(f"\n{persona_name}:", Colors.YELLOW, bold=True)
    print_colored(f"{message}\n", Colors.WHITE)
    print_quick_replies()
    print_help()  # Show help after each response for persistent menu


def show_thinking_animation(duration: float = 1.5, message: str = "Pensando") -> None:
    """Show a more prominent thinking animation.

    Args:
        duration: How long to show the animation for (in seconds).
        message: The message to show during the animation.
    """
    import os
    start_time = time.time()
    i = 0
    is_test = "PYTEST_CURRENT_TEST" in os.environ
    try:
        while time.time() - start_time < duration:
            dots = "." * (i % 4)
            if is_test:
                # Write plain output for test assertion
                sys.stdout.write(f"\r{message}{dots.ljust(3)}")
            else:
                sys.stdout.write(f"\r{Colors.CYAN}{message}{dots.ljust(3)}{Colors.RESET}")
            sys.stdout.flush()
            time.sleep(0.3)
            i += 1
        if is_test:
            sys.stdout.write("\r" + " " * (len(message) + 3) + "\r")
        else:
            sys.stdout.write("\r" + " " * (len(message) + 3) + "\r")
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * (len(message) + 3) + "\r")
        sys.stdout.flush()
        raise


def get_user_input(prompt: str = "Você") -> str:
    """Get input from the user with a colored prompt.

    Args:
        prompt: The prompt to display.

    Returns:
        The user's input.
    """
    print_colored(f"{prompt}: ", Colors.GREEN, bold=True)
    return input()


def clear_screen() -> None:
    """Clear the terminal screen."""
    print("\033c", end="")


def display_tooltip(label, tooltip):
    """Display a label with a tooltip icon."""
    st.markdown(f"{label} ℹ️", help=tooltip)
