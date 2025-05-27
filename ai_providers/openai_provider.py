"""
OpenAI Provider Implementation for LangGraph 101.

Provides integration with OpenAI's GPT models including GPT-4, GPT-3.5-turbo, and others.
Supports both chat completions and legacy completions API.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from model_manager import AIModel

logger = logging.getLogger(__name__)

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI provider."""
    api_key: str
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    
class OpenAIProvider(AIModel):
    """OpenAI GPT models provider."""
    
    SUPPORTED_MODELS = {
        'gpt-4o': {'context_length': 128000, 'supports_functions': True},
        'gpt-4o-mini': {'context_length': 128000, 'supports_functions': True},
        'gpt-4-turbo': {'context_length': 128000, 'supports_functions': True},
        'gpt-4': {'context_length': 8192, 'supports_functions': True},
        'gpt-3.5-turbo': {'context_length': 16385, 'supports_functions': True},
        'gpt-3.5-turbo-16k': {'context_length': 16385, 'supports_functions': True},
    }
    
    def __init__(self, model_id: str, api_key: str, **kwargs):
        super().__init__(model_id, "openai", api_key, **kwargs)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.config = OpenAIConfig(
            api_key=api_key,
            organization=kwargs.get('organization'),
            base_url=kwargs.get('base_url'),
            timeout=kwargs.get('timeout', 30.0),
            max_retries=kwargs.get('max_retries', 3)
        )
        
        # Initialize OpenAI clients
        client_kwargs = {
            'api_key': self.config.api_key,
            'timeout': self.config.timeout,
            'max_retries': self.config.max_retries
        }
        
        if self.config.organization:
            client_kwargs['organization'] = self.config.organization
        if self.config.base_url:
            client_kwargs['base_url'] = self.config.base_url
            
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        
        # Validate model
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported models list. Proceeding anyway.")
        
        logger.info(f"OpenAI provider initialized for model: {model_id}")
    
    def predict(self, prompt: str, **kwargs) -> str:
        """Generate a synchronous prediction using OpenAI."""
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, kwargs.get('system_message'))
            
            # Extract OpenAI-specific parameters
            model_params = self._extract_model_params(kwargs)
            
            # Make API call
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **model_params
            )
            
            # Log performance metrics
            self._log_performance_metrics(start_time, response.usage)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI prediction failed: {e}")
            raise
    
    async def apredict(self, prompt: str, **kwargs) -> str:
        """Generate an asynchronous prediction using OpenAI."""
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, kwargs.get('system_message'))
            
            # Extract OpenAI-specific parameters
            model_params = self._extract_model_params(kwargs)
            
            # Make async API call
            start_time = time.time()
            response = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **model_params
            )
            
            # Log performance metrics
            self._log_performance_metrics(start_time, response.usage)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI async prediction failed: {e}")
            raise
    
    def _prepare_messages(self, prompt: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI chat completions."""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _extract_model_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate OpenAI-specific parameters."""
        params = {}
        
        # Standard parameters
        if 'temperature' in kwargs:
            params['temperature'] = max(0.0, min(2.0, kwargs['temperature']))
        
        if 'max_tokens' in kwargs:
            params['max_tokens'] = min(kwargs['max_tokens'], 
                                     self.SUPPORTED_MODELS.get(self.model_id, {}).get('context_length', 4096))
        
        if 'top_p' in kwargs:
            params['top_p'] = max(0.0, min(1.0, kwargs['top_p']))
        
        if 'frequency_penalty' in kwargs:
            params['frequency_penalty'] = max(-2.0, min(2.0, kwargs['frequency_penalty']))
        
        if 'presence_penalty' in kwargs:
            params['presence_penalty'] = max(-2.0, min(2.0, kwargs['presence_penalty']))
        
        if 'stop' in kwargs:
            params['stop'] = kwargs['stop']
          # Function calling support
        if 'functions' in kwargs and self.SUPPORTED_MODELS.get(self.model_id, {}).get('supports_functions'):
            params['functions'] = kwargs['functions']
            if 'function_call' in kwargs:
                params['function_call'] = kwargs['function_call']
        
        return params
    
    def _log_performance_metrics(self, start_time: float, usage: Any):
        """Log performance metrics for analytics."""
        try:
            from analytics.analytics_logger import get_analytics_logger
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            
            analytics_logger = get_analytics_logger()
            
            # Calculate cost estimate
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            cost_estimate = self.estimate_cost(prompt_tokens, completion_tokens) if usage else 0
            
            analytics_logger.log_event(
                event_type='model_interaction',
                model_id=self.model_id,
                response_time_ms=response_time_ms,
                tokens_used=usage.total_tokens if usage else 0,
                cost_estimate=cost_estimate,
                details={
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'provider': self.provider,
                    'model_capabilities': self.SUPPORTED_MODELS.get(self.model_id, {})
                }
            )
            
        except Exception as e:
            logger.debug(f"Failed to log performance metrics: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = self.get_details()
        info.update(self.SUPPORTED_MODELS.get(self.model_id, {}))
        info['provider_specific'] = {
            'api_version': 'v1',
            'billing_model': 'token-based',
            'rate_limits': 'varies by tier'
        }
        return info
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for a given token usage (approximate)."""
        # Approximate pricing as of 2024 (subject to change)
        pricing = {
            'gpt-4o': {'input': 0.005, 'output': 0.015},  # per 1K tokens
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
        }
        
        model_pricing = pricing.get(self.model_id, {'input': 0.001, 'output': 0.002})
        
        input_cost = (prompt_tokens / 1000) * model_pricing['input']
        output_cost = (completion_tokens / 1000) * model_pricing['output']
        
        return input_cost + output_cost
    
    def supports_streaming(self) -> bool:
        """Check if model supports streaming responses."""
        return True
    
    def stream_predict(self, prompt: str, **kwargs):
        """Stream prediction responses (generator)."""
        try:
            messages = self._prepare_messages(prompt, kwargs.get('system_message'))
            model_params = self._extract_model_params(kwargs)
            model_params['stream'] = True
            
            start_time = time.time()
            stream = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **model_params
            )
            
            full_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content
            
            # Log metrics after streaming is complete
            self._log_performance_metrics(start_time, None)  # Usage not available in streaming
            
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
