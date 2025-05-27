"""
LangChain Integration Module with Robust Error Handling.

This module provides wrapper functions for LangChain operations with
deprecation warning suppression and proper error handling.
"""
import warnings
import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


@contextmanager
def suppress_langchain_warnings():
    """Context manager to suppress LangChain deprecation warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
        warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
        yield


class LangChainWrapper:
    """Wrapper class for LangChain operations with robust error handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def create_chat_model(self, model_name: str, temperature: float, api_key: str):
        """Create a chat model with error handling and warning suppression."""
        try:
            with suppress_langchain_warnings():
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    convert_system_message_to_human=True  # Explicitly set to avoid deprecation
                )
                
                self.logger.info(f"Successfully created chat model: {model_name}")
                return model
                
        except ImportError as e:
            self.logger.error(f"Failed to import LangChain Google GenAI: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to create chat model: {e}")
            raise
    
    def invoke_with_retry(self, model, messages: List[Dict[str, Any]], max_retries: int = 3):
        """Invoke model with retry logic and error handling."""
        for attempt in range(max_retries):
            try:
                with suppress_langchain_warnings():
                    response = model.invoke(messages)
                    self.logger.debug(f"Successfully invoked model on attempt {attempt + 1}")
                    return response
                    
            except Exception as e:
                self.logger.warning(f"Model invocation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                
                # Wait before retry (exponential backoff)
                import time
                wait_time = 2 ** attempt
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    def create_search_tool(self, api_key: str):
        """Create Tavily search tool with error handling."""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                api_wrapper_kwargs={"tavily_api_key": api_key}
            )
            
            self.logger.info("Successfully created Tavily search tool")
            return tool
            
        except ImportError as e:
            self.logger.error(f"Failed to import Tavily search tool: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to create search tool: {e}")
            raise


# Global instance
langchain_wrapper = LangChainWrapper()


def create_robust_chat_model(model_name: str, temperature: float, api_key: str):
    """Create a chat model with robust error handling."""
    return langchain_wrapper.create_chat_model(model_name, temperature, api_key)


def invoke_model_safely(model, messages: List[Dict[str, Any]], max_retries: int = 3):
    """Safely invoke a language model with retry logic."""
    return langchain_wrapper.invoke_with_retry(model, messages, max_retries)


def create_robust_search_tool(api_key: str):
    """Create a search tool with robust error handling."""
    return langchain_wrapper.create_search_tool(api_key)


# Utility function to check LangChain compatibility
def check_langchain_compatibility() -> Dict[str, Any]:
    """Check LangChain installation and compatibility."""
    compatibility_info = {
        "status": "unknown",
        "langchain_version": None,
        "langchain_google_genai_available": False,
        "langchain_community_available": False,
        "warnings": []
    }
    
    try:
        import langchain
        compatibility_info["langchain_version"] = langchain.__version__
        compatibility_info["status"] = "installed"
    except ImportError:
        compatibility_info["status"] = "not_installed"
        compatibility_info["warnings"].append("LangChain not installed")
        return compatibility_info
    
    try:
        import langchain_google_genai
        compatibility_info["langchain_google_genai_available"] = True
    except ImportError:
        compatibility_info["warnings"].append("langchain_google_genai not available")
    
    try:
        import langchain_community
        compatibility_info["langchain_community_available"] = True
    except ImportError:
        compatibility_info["warnings"].append("langchain_community not available")
    
    if not compatibility_info["warnings"]:
        compatibility_info["status"] = "fully_compatible"
    elif compatibility_info["langchain_google_genai_available"]:
        compatibility_info["status"] = "partially_compatible"
    else:
        compatibility_info["status"] = "incompatible"
    
    return compatibility_info


if __name__ == "__main__":
    # Test LangChain compatibility
    compat = check_langchain_compatibility()
    print(f"LangChain Compatibility Status: {compat['status']}")
    if compat["warnings"]:
        print("Warnings:")
        for warning in compat["warnings"]:
            print(f"  - {warning}")
