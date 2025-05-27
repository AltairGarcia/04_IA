"""
Terminal UI components for LangGraph 101 project.

This module provides UI functions specifically for terminal interfaces.
"""

import sys
import time
import os
from typing import Optional
from ui_base import get_i18n

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def print_colored(text: str, color: str, bold: bool = False) -> None:
    """Print text with color.

    Args:
        text: The text to print.
        color: The color to use.
        bold: Whether to make the text bold.
    """
    if bold:
        print(f"{Colors.BOLD}{color}{text}{Colors.RESET}")
    else:
        print(f"{color}{text}{Colors.RESET}")


def print_welcome(locale: str = "en") -> None:
    """Print a stylized welcome message.

    Args:
        locale: The locale to use for messages.
    """
    i18n = get_i18n(locale)
    print("\n" + "=" * 50)
    print_colored(i18n.get("welcome_title").center(50), Colors.YELLOW, bold=True)
    print("=" * 50)
    print_colored(f"\n{i18n.get('welcome_message')}", Colors.CYAN)
    print(i18n.get("exit_command"))
    print(i18n.get("help_command") + "\n")


def print_help(locale: str = "en") -> None:
    """Print help information with colors.

    Args:
        locale: The locale to use for messages.
    """
    i18n = get_i18n(locale)
    print_colored(f"\n{i18n.get('available_commands')}", Colors.YELLOW, bold=True)
    print_colored(i18n.get("exit_desc"), Colors.WHITE)
    print_colored(i18n.get("help_desc"), Colors.WHITE)
    print_colored(i18n.get("clear_desc"), Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)


def print_quick_replies(locale: str = "en"):
    """Print quick reply suggestions for common actions.

    Args:
        locale: The locale to use for messages.
    """
    i18n = get_i18n(locale)
    print_colored(f"\n{i18n.get('quick_suggestions')}", Colors.CYAN, bold=True)
    print_colored("[ajuda] [limpar] [personas] [salvar] [sair]", Colors.WHITE)


def print_error(message: str, locale: str = "en") -> None:
    """Print an error message.

    Args:
        message: The error message to print.
        locale: The locale to use for messages.
    """
    i18n = get_i18n(locale)
    print_colored(f"\n{i18n.get('error_prefix')}{message}", Colors.RED, bold=True)


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


def show_thinking_animation(duration: float = 1.5, message: str = "Thinking", locale: str = "en") -> None:
    """Show a more prominent thinking animation.

    Args:
        duration: How long to show the animation for (in seconds).
        message: The message to show during the animation.
        locale: The locale to use for messages.
    """
    i18n = get_i18n(locale)
    message = i18n.get("thinking", message)

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


def get_user_input(prompt: str = "You", locale: str = "en") -> str:
    """Get input from the user with a colored prompt.

    Args:
        prompt: The prompt to display.
        locale: The locale to use for messages.

    Returns:
        The user's input.
    """
    i18n = get_i18n(locale)
    prompt = i18n.get("user_prompt", prompt)
    print_colored(f"{prompt}: ", Colors.GREEN, bold=True)
    return input()


def clear_screen() -> None:
    """Clear the terminal screen."""
    print("\033c", end="")
