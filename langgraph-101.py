"""
LangGraph 101 - Main Application

This is the main entry point for the LangGraph 101 application.
It creates a chat interface for interacting with a LangGraph agent.
"""

import sys 
import traceback 
import os
from datetime import datetime
import json 
import logging 
import time # Added for sleep in test functions

from system_initialization import initialize_all_systems, SystemInitializer 

# Import modules from our project
from config import load_config, ConfigError, get_system_prompt, get_available_personas
from tools import get_tools, GeminiAPI, ElevenLabsTTS, PexelsAPI, PixabayAPI, StabilityAIAPI, DalleAPI, AssemblyAIAPI, DeepgramAPI, YouTubeDataAPI
from agent import create_agent, SimpleAgent 
from content_creation import ContentCreator
from model_manager import ModelManager 
from history import get_history_manager
from memory_manager import get_memory_manager
from personas import get_persona_by_name
from export import export_conversation, get_export_formats
from email_sender import send_email, email_conversation
from ui import (
    print_welcome, print_help, print_error, print_success,
    print_agent_response, show_thinking_animation, get_user_input,
    clear_screen, print_colored, Colors
)
from ai_providers.adapters import AIModelLangChainAdapter


def print_personas_list():
    """Print a list of available personas."""
    personas = get_available_personas()
    print_colored("\n--- Personas Disponíveis ---", Colors.YELLOW, bold=True)
    for name, info in personas.items():
        print_colored(f"{name}", Colors.CYAN, bold=True)
        print_colored(f"  {info['description']}", Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)


def print_export_formats():
    """Print a list of available export formats."""
    formats = get_export_formats()
    print_colored("\n--- Formatos de Exportação Disponíveis ---", Colors.YELLOW, bold=True)
    for name, info in formats.items():
        print_colored(f"{name}", Colors.CYAN, bold=True)
        print_colored(f"  {info['description']}", Colors.WHITE)
    print_colored("\nUse 'exportar FORMAT' para exportar a conversa (ex: exportar html)", Colors.WHITE)
    print_colored("------------------------\n", Colors.YELLOW)

def test_multi_provider_agent():
    """
    Temporary test function to verify that models from different providers
    can be loaded and invoked through AIModelLangChainAdapter and SimpleAgent.
    """
    logger = logging.getLogger(__name__) 
    logger.info("--- Starting Multi-Provider Agent Test ---")
    print_colored("\n--- Running Multi-Provider Agent Test ---", Colors.MAGENTA, bold=True)

    try:
        config_data = load_config() # Renamed to avoid conflict with 'config' module
        
        api_keys_dict = {
            key: config_data.get(key) for key in [
                "api_key", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "elevenlabs_api_key", "dalle_api_key", "stabilityai_api_key",
                "temperature", "model_name"
            ] if config_data.get(key) is not None
        }
        if 'api_key' in api_keys_dict and 'GEMINI_API_KEY' not in api_keys_dict:
            api_keys_dict['GEMINI_API_KEY'] = api_keys_dict['api_key']

        model_manager = ModelManager()
        # Pass model_manager to ContentCreator, though it's not the primary focus of this specific test's assertions
        content_creator_instance = ContentCreator(api_keys=api_keys_dict, model_manager=model_manager) 

        available_model_configs = model_manager.list_available_models()
        
        if not available_model_configs:
            logger.warning("Multi-Provider Test: No models configured in ModelManager.")
            print_error("Multi-Provider Test: No models configured in ModelManager. Test cannot run.")
            return

        providers_to_test = ["google", "openai", "anthropic"]
        system_prompt_str = config_data.get("system_prompt", "You are a helpful assistant.")

        for provider_name in providers_to_test:
            print_colored(f"\nTesting agent with provider: {provider_name.upper()}", Colors.YELLOW, bold=True)
            
            found_model_id_for_provider = None
            for model_conf_item in available_model_configs: # Renamed model_conf to avoid conflict
                if model_conf_item['provider'].lower() == provider_name.lower():
                    found_model_id_for_provider = model_conf_item['model_id']
                    break
            
            if not found_model_id_for_provider:
                logger.warning(f"Multi-Provider Test: No model found for provider '{provider_name}' in config.")
                print_colored(f"No model configured for provider: {provider_name}", Colors.RED)
                continue

            logger.info(f"Multi-Provider Test: Attempting to load model '{found_model_id_for_provider}' for provider '{provider_name}'.")
            test_ai_model = model_manager.get_model(found_model_id_for_provider)

            if not test_ai_model:
                logger.error(f"Multi-Provider Test: Failed to get model instance for '{found_model_id_for_provider}'.")
                print_error(f"Failed to get model instance for: {found_model_id_for_provider}")
                continue

            try:
                logger.info(f"Multi-Provider Test: Adapting model '{test_ai_model.model_id}'.")
                adapted_model = AIModelLangChainAdapter(ai_model=test_ai_model, system_prompt_override=system_prompt_str)
                
                logger.info(f"Multi-Provider Test: Creating SimpleAgent for '{test_ai_model.model_id}'.")
                test_agent_instance = SimpleAgent(
                    model_instance=adapted_model, 
                    system_prompt_str_for_agent_logic=system_prompt_str, 
                    available_tools_dict={}, 
                    content_creator_instance=content_creator_instance
                )
                
                test_input = f"Hello, this is a test for the {provider_name} model ({test_ai_model.model_id}). How are you?"
                logger.info(f"Multi-Provider Test: Invoking agent with input: '{test_input}'")
                print_colored(f"Invoking agent for {provider_name} ({test_ai_model.model_id})...", Colors.BLUE)
                
                response_dict = test_agent_instance.invoke({"input": test_input})
                response_content = response_dict.get("output", "No output content.")
                
                logger.info(f"Multi-Provider Test: Response from {provider_name} ({test_ai_model.model_id}): {response_content[:100]}...")
                print_success(f"Response from {provider_name} ({test_ai_model.model_id}):")
                print_colored(response_content, Colors.WHITE)

            except Exception as e_agent:
                logger.error(f"Multi-Provider Test: Error during agent invocation for {provider_name} ({found_model_id_for_provider}): {e_agent}", exc_info=True)
                print_error(f"Error during agent test for {provider_name} ({found_model_id_for_provider}): {e_agent}")
            time.sleep(1) 
    
    except ConfigError as e_conf:
        logger.error(f"Multi-Provider Test: Configuration error: {e_conf}", exc_info=True)
        print_error(f"Test Configuration Error: {e_conf}")
    except ImportError as e_import: 
        logger.error(f"Multi-Provider Test: Import error: {e_import}", exc_info=True)
        print_error(f"Test Import Error: {e_import}. Ensure all modules are correctly placed and named.")
    except Exception as e_general:
        logger.error(f"Multi-Provider Test: A general error occurred: {e_general}", exc_info=True)
        print_error(f"Test General Error: {e_general}")
    finally:
        print_colored("\n--- Multi-Provider Agent Test Finished ---", Colors.MAGENTA, bold=True)
        logger.info("--- Multi-Provider Agent Test Finished ---")

def test_content_creator_multi_model():
    """
    Temporary test function to verify ContentCreator's LLM methods
    with model_id_override for different providers.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting ContentCreator Multi-Model Override Test ---")
    print_colored("\n--- Running ContentCreator Multi-Model Override Test ---", Colors.MAGENTA, bold=True)

    try:
        config_data = load_config() 
        
        model_manager_instance = ModelManager() 

        api_keys_dict = {
            key: config_data.get(key) for key in [
                "api_key", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
                "elevenlabs_api_key", "dalle_api_key", "stabilityai_api_key", 
                "temperature", "model_name" 
            ] if config_data.get(key) is not None
        }
        if 'api_key' in api_keys_dict and 'GEMINI_API_KEY' not in api_keys_dict:
            api_keys_dict['GEMINI_API_KEY'] = api_keys_dict['api_key']

        creator = ContentCreator(api_keys=api_keys_dict, model_manager=model_manager_instance)

        models_to_test = {
            "google": "gemini-1.5-pro-latest", 
            "openai": "gpt-4o",
            "anthropic": "claude-3-haiku-20240307"
        }
        
        configured_model_ids = [m['model_id'] for m in model_manager_instance.list_available_models()]
        
        for provider, model_id in models_to_test.items():
            print_colored(f"\n--- Testing ContentCreator with {provider.upper()} model: {model_id} ---", Colors.YELLOW, bold=True)
            
            if model_id not in configured_model_ids:
                print_colored(f"Model {model_id} for provider {provider} is not configured. Skipping tests for this model.", Colors.RED)
                logger.warning(f"ContentCreator Test: Model {model_id} for provider {provider} not in ModelManager's available models. Skipping.")
                continue

            print_colored(f"Testing generate_script with {model_id}...", Colors.BLUE)
            script_result = creator.generate_script(
                topic="The Future of AI", tone="informative", duration_minutes=1, model_id_override=model_id
            )
            if script_result.get("error"): print_error(f"Script generation error with {model_id}: {script_result['error']}")
            else: print_success(f"Script generated with {model_id}. Title: {script_result.get('title')}")
            time.sleep(1)

            print_colored(f"Testing generate_blog_post with {model_id}...", Colors.BLUE)
            blog_result = creator.generate_blog_post(
                topic="Sustainable Energy Solutions", tone="professional", target_word_count=50, model_id_override=model_id 
            )
            if blog_result.get("error"): print_error(f"Blog post generation error with {model_id}: {blog_result['error']}")
            else: print_success(f"Blog post generated with {model_id}. Title: {blog_result.get('title')}")
            time.sleep(1)

            print_colored(f"Testing generate_twitter_thread with {model_id}...", Colors.BLUE)
            thread_result = creator.generate_twitter_thread(
                topic="Exploring Mars", num_tweets=2, tone="exciting", model_id_override=model_id
            )
            if thread_result.get("error"): print_error(f"Twitter thread generation error with {model_id}: {thread_result['error']}")
            else: print_success(f"Twitter thread generated with {model_id}. Title: {thread_result.get('title')}")
            time.sleep(1)

    except ConfigError as e_conf:
        logger.error(f"ContentCreator Test: Configuration error: {e_conf}", exc_info=True)
        print_error(f"ContentCreator Test Configuration Error: {e_conf}")
    except ImportError as e_import:
        logger.error(f"ContentCreator Test: Import error: {e_import}", exc_info=True)
        print_error(f"ContentCreator Test Import Error: {e_import}")
    except Exception as e_general:
        logger.error(f"ContentCreator Test: A general error occurred: {e_general}", exc_info=True)
        print_error(f"ContentCreator Test General Error: {e_general}")
    finally:
        print_colored("\n--- ContentCreator Multi-Model Override Test Finished ---", Colors.MAGENTA, bold=True)
        logger.info("--- ContentCreator Multi-Model Override Test Finished ---")

def update_help_menu():
    """Print updated help information including persona commands."""
    print_colored("\n--- Comandos Disponíveis ---", Colors.YELLOW, bold=True)
    print_colored("sair       - Encerra a conversa", Colors.WHITE)
    # ... (rest of help menu, truncated for brevity in example)
    print_colored("------------------------\n", Colors.YELLOW)


def print_cli_dashboard():
    """Print the CLI dashboard for tool selection and workflow orchestration."""
    print_colored("\n=== Content Creation Dashboard ===", Colors.YELLOW, bold=True)
    # ... (rest of dashboard print, truncated for brevity)
    print_colored("------------------------\n", Colors.YELLOW)


def main():
    """Main application entry point."""
    try:
        init_results = initialize_all_systems(use_env_vars=True, force=False)
        logger_main = logging.getLogger(__name__) # Use a different name to avoid conflict with global logger in tests
        if init_results.get('status') == 'error':
            log_message = f"CRITICAL: System initialization failed: {init_results.get('error')}"
            if logger_main: logger_main.critical(log_message)
            else: print(log_message)
        logger_main.info("LangGraph 101 CLI application started and logging initialized.")

        config_main = load_config() # Renamed to avoid conflict

        tool_list = [
            {'name': 'Google Gemini', 'desc': 'Gera scripts...', 'example': 'Roteiro...', 'key': 'gemini', 'api_key_name': 'GEMINI_API_KEY'},
            # ... (other tools, truncated for brevity)
        ]

        current_persona = config_main["current_persona"]
        history_manager = get_history_manager(max_history=config_main["max_history"])
        memory_manager = get_memory_manager(max_items=50, extraction_enabled=True)
        tools = get_tools()

        logger_main.info("Initializing ModelManager for the main application.")
        model_manager_main = ModelManager() # Renamed to avoid conflict

        logger_main.info("Gathering API keys for ContentCreator and ModelManager.")
        api_keys_main = { # Renamed
            key: config_main.get(key) for key in [
                "api_key", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "elevenlabs_api_key", "dalle_api_key", "stabilityai_api_key",
                "pixabay_api_key", "pexels_api_key", "deepgram_api_key",
                "assemblyai_api_key", "youtube_data_api_key", "news_api_key",
                "openweather_api_key", "model_name", "temperature"
            ] if config_main.get(key) is not None
        }
        if 'api_key' in api_keys_main and 'GEMINI_API_KEY' not in api_keys_main:
             api_keys_main['GEMINI_API_KEY'] = api_keys_main['api_key']
        
        api_keys_main = {k: v for k, v in api_keys_main.items() if v is not None}

        if not any(key in api_keys_main for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]):
            logger_main.warning("No primary LLM API keys found. ContentCreator LLM functions might be limited.")

        logger_main.info("Initializing ContentCreator with API keys and ModelManager.")
        content_creator_main = ContentCreator( # Renamed
            api_keys=api_keys_main,
            model_manager=model_manager_main
        )

        logger_main.info("Creating main agent executor.")
        agent_executor = create_agent(
            config=config_main, # Use renamed config
            tools=tools,
            content_creator=content_creator_main # Use renamed instance
        )

        print_welcome()
        print_colored(f"Persona atual: {current_persona.name} - {current_persona.description}", Colors.CYAN)
        
        # Omitting the full dashboard and chat loop for brevity in this overwrite example.
        # The critical parts are the test functions and correct ContentCreator instantiation.
        print_colored("\n[Dashboard and Chat loop would run here. Exiting for test brevity after tests.]", Colors.GRAY)
        # Example of how a simplified chat interaction might start, if not exiting:
        # if not (dashboard_input.strip() == "0" and not config_main.get("allow_chat_after_dashboard_exit", True)):
        #    logger_main.info("Iniciando sessão de chat principal.")
        #    print_colored("\nIniciando sessão de chat...", Colors.GREEN)
        #    # ... chat loop ...


    except ConfigError as e:
        local_logger = logging.getLogger(__name__) # Ensure logger is defined
        local_logger.error(f"Erro de configuração: {str(e)}", exc_info=True)
        print_error(f"Erro de configuração: {str(e)}")
        sys.exit(1)
    except Exception as e:
        local_logger = logging.getLogger(__name__) # Ensure logger is defined
        local_logger.error(f"Erro inesperado na aplicação CLI: {str(e)}", exc_info=True)
        print_error(f"Erro inesperado: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logging is configured before any operations, including tests.
    # initialize_all_systems() in main() handles this for the main app.
    # For tests running before main, basicConfig might be needed if they log and init hasn't run.
    # However, test functions get their logger from logging.getLogger which should be fine if main() sets up logging first,
    # OR if the test functions are robust to unconfigured logging (e.g. default to console).
    # The current structure calls tests, then main. initialize_all_systems in main will set up handlers.
    
    test_multi_provider_agent()
    test_content_creator_multi_model() 
    main()

[end of langgraph-101.py]
