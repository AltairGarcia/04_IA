"""
Unit tests for the ModelManager and related classes.
"""
import unittest
import os
from unittest.mock import patch, MagicMock

# Adjust the import path based on your project structure
# This assumes 'core' and 'model_manager.py' are discoverable.
# If tests are in a subfolder, you might need to adjust sys.path or use relative imports.
from model_manager import ModelManager, AIModel, PlaceholderAIModel
from core.config import UnifiedConfig, ModelsConfig, ModelDetail, APIConfig, LangGraphConfig, AppConfig, LoggingConfig, SecurityConfig, DatabaseConfig

# A mock get_config to control the configuration during tests
def get_mock_config(available_models_list=None, default_model_id_override=None):
    mock_config = MagicMock(spec=UnifiedConfig)
    
    mock_config.app = AppConfig()
    mock_config.logging = LoggingConfig()
    mock_config.security = SecurityConfig()
    mock_config.database = DatabaseConfig()

    # Mock APIConfig and its get_api_key method
    mock_api_config = MagicMock(spec=APIConfig)
    def mock_get_api_key(provider_key_name):
        # provider_key_name would be like 'gemini', 'openai'
        env_var_name = provider_key_name.upper() + "_API_KEY"
        return os.getenv(env_var_name, f"mock_{provider_key_name}_key")
    mock_api_config.get_api_key = MagicMock(side_effect=mock_get_api_key)
    mock_config.api = mock_api_config

    # Mock ModelsConfig
    mock_models_config = MagicMock(spec=ModelsConfig)
    if available_models_list is None:
        available_models_list = [
            ModelDetail(model_id="test-gemini", provider="google", api_key_env_var="GEMINI_API_KEY"),
            ModelDetail(model_id="test-openai", provider="openai", api_key_env_var="OPENAI_API_KEY"),
            ModelDetail(model_id="test-local", provider="local", parameters={"path": "/models/local"}),
        ]
    mock_models_config.available_models = available_models_list
    
    if default_model_id_override is not None:
        mock_models_config.default_model_id = default_model_id_override
    elif available_models_list:
        mock_models_config.default_model_id = available_models_list[0].model_id
    else:
        mock_models_config.default_model_id = None
        
    mock_config.models = mock_models_config

    # Mock LangGraphConfig
    mock_langgraph_config = MagicMock(spec=LangGraphConfig)
    mock_langgraph_config.default_model_id = mock_models_config.default_model_id
    mock_config.langgraph = mock_langgraph_config
    
    return mock_config

class TestModelManager(unittest.TestCase):

    def setUp(self):
        # Reset singleton instance for consistent tests
        ModelManager._instance = None 
        # Clear relevant environment variables that might interfere
        self.original_env_vars = {}
        for var in ["GEMINI_API_KEY", "OPENAI_API_KEY", "MODELS_CONFIG_JSON"]:
            if var in os.environ:
                self.original_env_vars[var] = os.environ[var]
                del os.environ[var]

    def tearDown(self):
        # Restore original environment variables
        for var, val in self.original_env_vars.items():
            os.environ[var] = val
        ModelManager._instance = None # Clean up singleton

    @patch('model_manager.get_config')
    def test_initialization_with_default_config(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config()
        
        manager = ModelManager()
        self.assertIsNotNone(manager)
        self.assertEqual(manager.default_model_id, "test-gemini")
        self.assertEqual(len(manager.list_available_models()), 3)

    @patch('model_manager.get_config')
    def test_initialization_with_empty_models_config(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config(available_models_list=[])
        
        manager = ModelManager()
        self.assertIsNone(manager.default_model_id)
        self.assertEqual(len(manager.list_available_models()), 0)

    @patch('model_manager.get_config')
    def test_get_default_model(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config()
        os.environ["GEMINI_API_KEY"] = "actual_gemini_key_for_test"

        manager = ModelManager()
        model = manager.get_default_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, PlaceholderAIModel) # Using Placeholder for now
        self.assertEqual(model.model_id, "test-gemini")
        self.assertEqual(model.provider, "google")
        self.assertEqual(model.api_key, "actual_gemini_key_for_test") # Check if API key is picked up

    @patch('model_manager.get_config')
    def test_get_specific_model(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config()
        os.environ["OPENAI_API_KEY"] = "actual_openai_key_for_test"
        
        manager = ModelManager()
        model = manager.get_model("test-openai")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, PlaceholderAIModel)
        self.assertEqual(model.model_id, "test-openai")
        self.assertEqual(model.provider, "openai")
        self.assertEqual(model.api_key, "actual_openai_key_for_test")

    @patch('model_manager.get_config')
    def test_get_model_caching(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config()
        
        manager = ModelManager()
        model1 = manager.get_model("test-gemini")
        model2 = manager.get_model("test-gemini")
        self.assertIs(model1, model2) # Should be the same instance

    @patch('model_manager.get_config')
    def test_get_non_existent_model(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config()
        
        manager = ModelManager()
        model = manager.get_model("non-existent-model")
        self.assertIsNone(model)

    @patch('model_manager.get_config')
    def test_list_available_models(self, mock_get_config_global):
        custom_models = [
            ModelDetail(model_id="custom-1", provider="custom_provider"),
            ModelDetail(model_id="custom-2", provider="another_provider", base_url="http://localhost:8080")
        ]
        mock_get_config_global.return_value = get_mock_config(available_models_list=custom_models)
        
        manager = ModelManager()
        listed_models = manager.list_available_models()
        self.assertEqual(len(listed_models), 2)
        self.assertEqual(listed_models[0]["model_id"], "custom-1")
        self.assertEqual(listed_models[1]["provider"], "another_provider")

    @patch('model_manager.get_config')
    @patch('model_manager.PlaceholderAIModel') # Mock the actual model class
    def test_model_creation_failure(self, MockPlaceholderAIModel, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config()
        MockPlaceholderAIModel.side_effect = Exception("Failed to create model")
        
        manager = ModelManager()
        model = manager.get_model("test-gemini")
        self.assertIsNone(model)
        # Check if the error was logged (optional, requires logger mocking)

    @patch('model_manager.get_config')
    def test_get_default_model_when_none_configured(self, mock_get_config_global):
        mock_get_config_global.return_value = get_mock_config(available_models_list=[], default_model_id_override=None)
        manager = ModelManager()
        self.assertIsNone(manager.default_model_id)
        model = manager.get_default_model()
        self.assertIsNone(model)

    def test_placeholder_ai_model(self):
        model = PlaceholderAIModel(model_id="placeholder-test", provider="test")
        self.assertEqual(model.model_id, "placeholder-test")
        sync_response = model.predict("Hello")
        self.assertIn("Placeholder response for 'placeholder-test'", sync_response)
        
        # Basic test for async method
        import asyncio
        async_response = asyncio.run(model.apredict("Hello Async"))
        self.assertIn("Async placeholder response for 'placeholder-test'", async_response)
        
        details = model.get_details()
        self.assertEqual(details["model_id"], "placeholder-test")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)