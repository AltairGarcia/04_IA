"""
Manages AI models, allowing for seamless switching between different
LLMs and other AI model types.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.config import get_config, ModelDetail

logger = logging.getLogger(__name__)

class AIModel(ABC):
    """
    Abstract base class for AI models.
    Defines the common interface for interacting with different models.
    """
    def __init__(self, model_id: str, provider: str, api_key: Optional[str] = None, base_url: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.params = params if params else {}
        logger.info(f"Initializing AIModel: {self.model_id} (Provider: {self.provider})")

    @abstractmethod
    def predict(self, prompt: str, **kwargs) -> Any:
        """
        Generate a prediction or response from the model.
        """
        pass

    @abstractmethod
    async def apredict(self, prompt: str, **kwargs) -> Any:
        """
        Asynchronously generate a prediction or response from the model.
        """
        pass

    def get_details(self) -> Dict[str, Any]:
        """
        Returns details about the model.
        """
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "base_url": self.base_url,
            "params": self.params
        }

class PlaceholderAIModel(AIModel):
    """
    A placeholder implementation of AIModel.
    This should be replaced with actual model client implementations.
    """
    def predict(self, prompt: str, **kwargs) -> str:
        logger.info(f"PlaceholderAIModel ({self.model_id}) received prompt: {prompt[:50]}...")
        return f"Placeholder response for '{self.model_id}': Prompt was '{prompt}' with kwargs {kwargs}"

    async def apredict(self, prompt: str, **kwargs) -> str:
        logger.info(f"PlaceholderAIModel ({self.model_id}) received async prompt: {prompt[:50]}...")
        return f"Async placeholder response for '{self.model_id}': Prompt was '{prompt}' with kwargs {kwargs}"


class ModelManager:
    """
    Manages the selection and instantiation of AI models based on configuration.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.config = get_config()
        self._models: Dict[str, AIModel] = {} # Cache for instantiated models
        self._available_model_configs: List[ModelDetail] = self.config.models.available_models
        self.default_model_id: Optional[str] = self.config.models.default_model_id
        
        logger.info(f"ModelManager initialized. Default model ID: {self.default_model_id}")
        logger.info(f"Available model configurations: {[m.model_id for m in self._available_model_configs]}")
        self._initialized = True

    def _create_model_instance(self, model_config: ModelDetail) -> AIModel:
        """
        Factory method to create model instances using the new AI providers.
        """
        import os
        
        api_key = None
        if model_config.api_key_env_var:
            api_key = self.config.api.get_api_key(model_config.api_key_env_var.replace("_API_KEY", "").lower())
            if not api_key: # Try direct env var if not found via unified config (e.g. for custom keys)
                 api_key = os.getenv(model_config.api_key_env_var)

        # Use the new AI providers
        try:
            if model_config.provider.lower() == "openai":
                from ai_providers.openai_provider import OpenAIProvider
                return OpenAIProvider(
                    model_id=model_config.model_id,
                    api_key=api_key,
                    base_url=model_config.base_url,
                    **model_config.parameters
                )
            elif model_config.provider.lower() == "anthropic":
                from ai_providers.anthropic_provider import AnthropicProvider
                return AnthropicProvider(
                    model_id=model_config.model_id,
                    api_key=api_key,
                    base_url=model_config.base_url,
                    **model_config.parameters
                )
            elif model_config.provider.lower() in ["google", "gemini"]:
                from ai_providers.google_provider import GoogleProvider
                return GoogleProvider(
                    model_id=model_config.model_id,
                    api_key=api_key,
                    base_url=model_config.base_url,
                    **model_config.parameters
                )
            else:
                logger.warning(f"Unknown provider '{model_config.provider}' for model {model_config.model_id}. Using placeholder.")
                return PlaceholderAIModel(
                    model_id=model_config.model_id,
                    provider=model_config.provider,
                    api_key=api_key,
                    base_url=model_config.base_url,
                    params=model_config.parameters
                )
        except ImportError as e:
            logger.error(f"Failed to import provider for {model_config.provider}: {e}")
            logger.warning(f"Creating PlaceholderAIModel for {model_config.model_id}.")
            return PlaceholderAIModel(
                model_id=model_config.model_id,
                provider=model_config.provider,
                api_key=api_key,
                base_url=model_config.base_url,
                params=model_config.parameters
            )

    def get_model(self, model_id: Optional[str] = None) -> Optional[AIModel]:
        """
        Retrieves an instantiated AI model. If model_id is None, returns the default model.
        """
        target_model_id = model_id or self.default_model_id
        
        if not target_model_id:
            logger.error("No model_id specified and no default model configured.")
            return None

        if target_model_id in self._models:
            return self._models[target_model_id]

        model_config = next((m for m in self._available_model_configs if m.model_id == target_model_id), None)
        
        if not model_config:
            logger.error(f"Model configuration for '{target_model_id}' not found.")
            return None
            
        try:
            instance = self._create_model_instance(model_config)
            self._models[target_model_id] = instance
            return instance
        except Exception as e:
            logger.error(f"Failed to create model instance for '{target_model_id}': {e}", exc_info=True)
            return None

    def get_default_model(self) -> Optional[AIModel]:
        """
        Retrieves the default AI model.
        """
        if not self.default_model_id:
            logger.warning("No default model ID is configured.")
            return None
        return self.get_model(self.default_model_id)

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lists available models with their configurations.
        """
        return [
            {
                "model_id": mc.model_id,
                "provider": mc.provider,
                "base_url": mc.base_url,
                "parameters": mc.parameters
            }
            for mc in self._available_model_configs
        ]

# Example usage (for testing purposes)
if __name__ == "__main__":
    import os # Required for _create_model_instance if api_key_env_var is used directly
    logging.basicConfig(level=logging.INFO)
    
    # You might need to set environment variables for API keys for this test to fully work
    # e.g., export GEMINI_API_KEY="your_key_here"
    # os.environ["GEMINI_API_KEY"] = "test_gemini_key" 
    # os.environ["OPENAI_API_KEY"] = "test_openai_key"

    manager = ModelManager()
    
    print("\nAvailable Models:")
    for model_info in manager.list_available_models():
        print(f"  - ID: {model_info['model_id']}, Provider: {model_info['provider']}")

    default_model = manager.get_default_model()
    if default_model:
        print(f"\nDefault Model ({default_model.model_id}):")
        print(f"  Details: {default_model.get_details()}")
        print(f"  Sync Prediction: {default_model.predict('Hello from default model!')}")
    else:
        print("\nNo default model available.")

    # Get a specific model
    specific_model_id = "gpt-4o" # Assuming this is in your default config or MODELS_CONFIG_JSON
    if any(m.model_id == specific_model_id for m in manager._available_model_configs):
        gpt_model = manager.get_model(specific_model_id)
        if gpt_model:
            print(f"\nSpecific Model ({gpt_model.model_id}):")
            print(f"  Details: {gpt_model.get_details()}")
            print(f"  Sync Prediction: {gpt_model.predict('Hello from specific model GPT!')}")
        else:
            print(f"\nCould not load model: {specific_model_id}")
    else:
        print(f"\nModel {specific_model_id} not configured, skipping specific model test.")

    # Test getting a non-existent model
    non_existent_model = manager.get_model("non-existent-model-123")
    if not non_existent_model:
        print("\nSuccessfully handled request for non-existent model.")