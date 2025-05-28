"""
Adapter to make AIModel instances compatible with LangChain's BaseChatModel.
"""
import sys
from pathlib import Path
from typing import List, Optional, Any, Dict

# Ensure the root directory is in sys.path for model_manager import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model_manager import AIModel # Assuming AIModel is in model_manager.py in root
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage, # Added for more comprehensive message type handling
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun

class AIModelLangChainAdapter(BaseChatModel):
    """
    Adapter to use an AIModel instance as a LangChain BaseChatModel.
    """
    ai_model: AIModel
    system_prompt_override: Optional[str] = None

    def __init__(self, ai_model: AIModel, system_prompt_override: Optional[str] = None):
        super().__init__()
        self.ai_model = ai_model
        self.system_prompt_override = system_prompt_override

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generates a chat response using the wrapped AIModel.
        """
        system_message_str: Optional[str] = self.system_prompt_override
        prompt_parts: List[str] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                # If an explicit system prompt override is given to constructor, it takes precedence.
                # Otherwise, the SystemMessage from the list is used.
                # If multiple SystemMessages, the last one is used (or combined if preferred, here using last).
                if not self.system_prompt_override:
                    system_message_str = message.content
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"AI: {message.content}")
            elif isinstance(message, ChatMessage): # Handle generic ChatMessage
                prompt_parts.append(f"{message.role}: {message.content}")
            else: # Fallback for other message types
                prompt_parts.append(message.content)
        
        prompt_str = "\n".join(prompt_parts)

        # Prepare arguments for ai_model.predict
        predict_kwargs = kwargs.copy()
        if system_message_str:
            predict_kwargs['system_message'] = system_message_str
        if stop:
            predict_kwargs['stop'] = stop
        
        # Assuming ai_model.predict can handle a 'system_message' kwarg.
        # If not, this part would need adjustment based on AIModel's actual interface.
        try:
            response_text = self.ai_model.predict(prompt_str, **predict_kwargs)
        except TypeError as e:
            # This might happen if 'system_message' or 'stop' is not an expected kwarg
            # by the underlying AIModel's predict method.
            # We might need to log this and try without them or adjust AIModel interface.
            if 'system_message' in str(e) and 'system_message' in predict_kwargs:
                del predict_kwargs['system_message']
            if 'stop' in str(e) and 'stop' in predict_kwargs:
                del predict_kwargs['stop']
            # Retry predict call with potentially modified kwargs
            response_text = self.ai_model.predict(prompt_str, **predict_kwargs)


        ai_message = AIMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Asynchronously generates a chat response using the wrapped AIModel.
        """
        system_message_str: Optional[str] = self.system_prompt_override
        prompt_parts: List[str] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                if not self.system_prompt_override:
                    system_message_str = message.content
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"AI: {message.content}")
            elif isinstance(message, ChatMessage):
                prompt_parts.append(f"{message.role}: {message.content}")
            else:
                prompt_parts.append(message.content)

        prompt_str = "\n".join(prompt_parts)
        
        predict_kwargs = kwargs.copy()
        if system_message_str:
            predict_kwargs['system_message'] = system_message_str
        if stop:
            predict_kwargs['stop'] = stop

        # Assuming ai_model.apredict exists and handles similar kwargs
        try:
            response_text = await self.ai_model.apredict(prompt_str, **predict_kwargs)
        except TypeError as e:
            if 'system_message' in str(e) and 'system_message' in predict_kwargs:
                del predict_kwargs['system_message']
            if 'stop' in str(e) and 'stop' in predict_kwargs:
                del predict_kwargs['stop']
            response_text = await self.ai_model.apredict(prompt_str, **predict_kwargs)


        ai_message = AIMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "aimodel_langchain_adapter"

    # For Langchain compatibility, it's good practice to expose some model parameters
    # if the underlying AIModel has them.
    @property
    def model_id(self) -> str:
        return self.ai_model.model_id
    
    @property
    def provider(self) -> str:
        return self.ai_model.provider

    # If your AIModel has identifiable kwargs, you can expose them.
    # This is important for things like LCEL and runnables.
    @property
    def lc_serializable(self) -> bool:
        return True # Mark that this class can be serialized by LangChain

    def get_identifying_params(self) -> Dict[str, Any]:
        """
        Get the identifying parameters.
        """
        return {
            "model_id": self.ai_model.model_id,
            "provider": self.ai_model.provider,
            "system_prompt_override": self.system_prompt_override,
            **(self.ai_model.params if hasattr(self.ai_model, 'params') else {})
        }

    # BaseChatModel requires this if you don't implement _stream or _astream
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> None: # Actually Iterator[ChatGenerationChunk]
        raise NotImplementedError("Streaming is not yet implemented for this adapter.")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> None: # Actually AsyncIterator[ChatGenerationChunk]
        raise NotImplementedError("Async streaming is not yet implemented for this adapter.")

if __name__ == '__main__':
    # This is a placeholder for testing.
    # To run this, you'd need a concrete AIModel implementation and LangChain installed.
    
    # Example of a dummy AIModel for testing
    class DummyAIModel(AIModel):
        def __init__(self, model_id: str, provider: str, params: Optional[Dict[str, Any]] = None):
            super().__init__(model_id, provider, params=params)

        def predict(self, prompt: str, **kwargs) -> Any:
            print(f"DummyAIModel received prompt: {prompt}")
            print(f"DummyAIModel received kwargs: {kwargs}")
            sys_msg = kwargs.get('system_message', 'No system message')
            return f"Response from {self.model_id} to '{prompt}' with system message '{sys_msg}'"

        async def apredict(self, prompt: str, **kwargs) -> Any:
            print(f"DummyAIModel (async) received prompt: {prompt}")
            print(f"DummyAIModel (async) received kwargs: {kwargs}")
            sys_msg = kwargs.get('system_message', 'No system message')
            return f"Async response from {self.model_id} to '{prompt}' with system message '{sys_msg}'"

    print("Testing AIModelLangChainAdapter...")
    dummy_model = DummyAIModel(model_id="dummy-model-001", provider="dummy_provider", params={"temperature": 0.5})
    adapter = AIModelLangChainAdapter(ai_model=dummy_model, system_prompt_override="Be a helpful assistant.")

    # Test _generate
    print("\n--- Testing _generate ---")
    human_message = HumanMessage(content="Hello, how are you?")
    # System message in list, but constructor override takes precedence
    system_message_in_list = SystemMessage(content="This should be ignored.") 
    chat_result = adapter.generate([[human_message, system_message_in_list]], stop=["\nHuman:"])
    
    if chat_result.generations:
        response_message = chat_result.generations[0].message
        print(f"Adapter response: {response_message.content}")
        print(f"Adapter response type: {type(response_message)}")
    else:
        print("No generations in chat_result.")
    
    # Test without system_prompt_override to use the one from messages
    adapter_no_override = AIModelLangChainAdapter(ai_model=dummy_model)
    print("\n--- Testing _generate with system message from list ---")
    system_message_from_list = SystemMessage(content="Be a pirate.")
    chat_result_pirate = adapter_no_override.generate([[human_message, system_message_from_list]])
    if chat_result_pirate.generations:
        print(f"Adapter response (pirate): {chat_result_pirate.generations[0].message.content}")

    # Test identifying params
    print("\n--- Testing identifying_params ---")
    print(adapter.get_identifying_params())
    
    # Test LLM type
    print("\n--- Testing _llm_type ---")
    print(adapter._llm_type)

    # Test async generate (basic)
    import asyncio
    async def test_async():
        print("\n--- Testing _agenerate ---")
        chat_result_async = await adapter.agenerate([[human_message]])
        if chat_result_async.generations:
            response_message_async = chat_result_async.generations[0].message
            print(f"Adapter async response: {response_message_async.content}")
    
    asyncio.run(test_async())

    print("\nAdapter tests complete.")
    # Note: The _stream and _astream methods are not implemented and will raise NotImplementedError if called.
    # LangChain's BaseChatModel.generate will call _generate if _agenerate is not overridden for async calls.
    # However, for true async behavior, _agenerate should be used.
    # The current BaseChatModel structure prefers _agenerate if available.
    # I've added a basic _agenerate implementation.
    # For full LCEL compatibility, things like batching (_generate with List[List[BaseMessage]]) might be needed.
    # Also, the `params` from AIModel are now included in `get_identifying_params`.
    # Added `lc_serializable = True` and `get_identifying_params()` for better LangChain integration.
    # Added ChatMessage handling.
    # Added basic error handling in _generate and _agenerate for missing kwargs in AIModel.predict/apredict.
    
    # To make this test runnable, ensure model_manager.py with AIModel is in the parent directory
    # and langchain_core is installed.
    # Example:
    # model_manager.py content:
    # class AIModel(ABC): ... (as defined in previous tasks)
    # Then run: python ai_providers/adapters.py
    # (This test structure assumes direct execution for demonstration)

    # The current implementation of _generate and _agenerate creates a single prompt string.
    # For models that support a list of messages directly (like OpenAI chat models),
    # it would be more appropriate to pass the messages list if self.ai_model can handle it.
    # This adapter assumes self.ai_model.predict/apredict takes a single string prompt.
    # If AIModel evolves to accept message lists, this adapter should be updated.

```
