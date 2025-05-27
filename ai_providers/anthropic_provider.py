"""
Anthropic Provider Implementation for LangGraph 101.

Provides integration with Anthropic's Claude models including Claude-3, Claude-2, and Instant variants.
Supports the latest Anthropic API with proper message formatting.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from model_manager import AIModel

logger = logging.getLogger(__name__)

@dataclass
class AnthropicConfig:
    """Configuration for Anthropic provider."""
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3

class AnthropicProvider(AIModel):
    """Anthropic Claude models provider."""
    
    SUPPORTED_MODELS = {
        'claude-3-5-sonnet-20241022': {
            'context_length': 200000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_tools': True
        },
        'claude-3-5-haiku-20241022': {
            'context_length': 200000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_tools': True
        },
        'claude-3-opus-20240229': {
            'context_length': 200000,
            'max_output': 4096,
            'supports_vision': True,
            'supports_tools': True
        },
        'claude-3-sonnet-20240229': {
            'context_length': 200000,
            'max_output': 4096,
            'supports_vision': True,
            'supports_tools': True
        },
        'claude-3-haiku-20240307': {
            'context_length': 200000,
            'max_output': 4096,
            'supports_vision': True,
            'supports_tools': True
        },
        'claude-2.1': {
            'context_length': 200000,
            'max_output': 4096,
            'supports_vision': False,
            'supports_tools': False
        },
        'claude-2.0': {
            'context_length': 100000,
            'max_output': 4096,
            'supports_vision': False,
            'supports_tools': False
        },
        'claude-instant-1.2': {
            'context_length': 100000,
            'max_output': 4096,
            'supports_vision': False,
            'supports_tools': False
        }
    }
    
    def __init__(self, model_id: str, api_key: str, **kwargs):
        super().__init__(model_id, "anthropic", api_key, **kwargs)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not found. Install with: pip install anthropic")
        
        self.config = AnthropicConfig(
            api_key=api_key,
            base_url=kwargs.get('base_url'),
            timeout=kwargs.get('timeout', 60.0),
            max_retries=kwargs.get('max_retries', 3)
        )
        
        # Initialize Anthropic clients
        client_kwargs = {
            'api_key': self.config.api_key,
            'timeout': self.config.timeout,
            'max_retries': self.config.max_retries
        }
        
        if self.config.base_url:
            client_kwargs['base_url'] = self.config.base_url
            
        self.client = Anthropic(**client_kwargs)
        self.async_client = AsyncAnthropic(**client_kwargs)
        
        # Validate model
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported models list. Proceeding anyway.")
        
        logger.info(f"Anthropic provider initialized for model: {model_id}")
    
    def predict(self, prompt: str, **kwargs) -> str:
        """Generate a synchronous prediction using Anthropic Claude."""
        try:
            # Prepare messages and system prompt
            messages, system_prompt = self._prepare_messages(prompt, kwargs.get('system_message'))
            
            # Extract Anthropic-specific parameters
            model_params = self._extract_model_params(kwargs)
            
            # Make API call
            start_time = time.time()
            
            api_kwargs = {
                'model': self.model_id,
                'messages': messages,
                **model_params
            }
            
            if system_prompt:
                api_kwargs['system'] = system_prompt
            
            response = self.client.messages.create(**api_kwargs)
            
            # Log performance metrics
            self._log_performance_metrics(start_time, response.usage)
            
            # Extract content from response
            return self._extract_content(response)
            
        except Exception as e:
            logger.error(f"Anthropic prediction failed: {e}")
            raise
    
    async def apredict(self, prompt: str, **kwargs) -> str:
        """Generate an asynchronous prediction using Anthropic Claude."""
        try:
            # Prepare messages and system prompt
            messages, system_prompt = self._prepare_messages(prompt, kwargs.get('system_message'))
            
            # Extract Anthropic-specific parameters
            model_params = self._extract_model_params(kwargs)
            
            # Make async API call
            start_time = time.time()
            
            api_kwargs = {
                'model': self.model_id,
                'messages': messages,
                **model_params
            }
            
            if system_prompt:
                api_kwargs['system'] = system_prompt
            
            response = await self.async_client.messages.create(**api_kwargs)
            
            # Log performance metrics
            self._log_performance_metrics(start_time, response.usage)
            
            # Extract content from response
            return self._extract_content(response)
            
        except Exception as e:
            logger.error(f"Anthropic async prediction failed: {e}")
            raise
    
    def _prepare_messages(self, prompt: str, system_message: Optional[str] = None) -> tuple:
        """Prepare messages for Anthropic API."""
        messages = [{"role": "user", "content": prompt}]
        
        # Anthropic uses separate system parameter, not in messages
        return messages, system_message
    
    def _extract_model_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate Anthropic-specific parameters."""
        params = {}
        
        # Standard parameters
        max_tokens = kwargs.get('max_tokens', 1024)
        model_info = self.SUPPORTED_MODELS.get(self.model_id, {})
        max_output = model_info.get('max_output', 4096)
        params['max_tokens'] = min(max_tokens, max_output)
        
        if 'temperature' in kwargs:
            params['temperature'] = max(0.0, min(1.0, kwargs['temperature']))
        
        if 'top_p' in kwargs:
            params['top_p'] = max(0.0, min(1.0, kwargs['top_p']))
        
        if 'top_k' in kwargs:
            params['top_k'] = max(1, kwargs['top_k'])
        
        if 'stop_sequences' in kwargs:
            params['stop_sequences'] = kwargs['stop_sequences']
        elif 'stop' in kwargs:
            # Convert OpenAI-style stop to Anthropic stop_sequences
            stop = kwargs['stop']
            if isinstance(stop, str):
                params['stop_sequences'] = [stop]
            elif isinstance(stop, list):
                params['stop_sequences'] = stop
        
        # Tool use support (for newer models)
        if 'tools' in kwargs and model_info.get('supports_tools', False):
            params['tools'] = kwargs['tools']
            if 'tool_choice' in kwargs:
                params['tool_choice'] = kwargs['tool_choice']
        
        return params
    
    def _extract_content(self, response) -> str:
        """Extract text content from Anthropic response."""
        if hasattr(response, 'content') and response.content:
            # Handle multiple content blocks
            content_parts = []
            for block in response.content:
                if hasattr(block, 'text'):
                    content_parts.append(block.text)
                elif hasattr(block, 'type') and block.type == 'text':
                    content_parts.append(block.text if hasattr(block, 'text') else str(block))
            return ''.join(content_parts)
        return str(response)
    
    def _log_performance_metrics(self, start_time: float, usage: Any):
        """Log performance metrics for analytics."""
        try:
            from analytics.analytics_logger import get_analytics_logger
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            
            analytics_logger = get_analytics_logger()
            
            # Extract token information
            input_tokens = getattr(usage, 'input_tokens', 0) if usage else 0
            output_tokens = getattr(usage, 'output_tokens', 0) if usage else 0
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost estimate
            cost_estimate = self.estimate_cost(input_tokens, output_tokens) if usage else 0
            
            analytics_logger.log_event(
                event_type='model_interaction',
                model_id=self.model_id,
                response_time_ms=response_time_ms,
                tokens_used=total_tokens,
                cost_estimate=cost_estimate,
                details={
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
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
            'api_version': 'messages',
            'billing_model': 'token-based',
            'rate_limits': 'varies by tier',
            'safety_features': 'constitutional_ai'
        }
        return info
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given token usage (approximate)."""
        # Approximate pricing as of 2024 (subject to change)
        pricing = {
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'claude-3-5-haiku-20241022': {'input': 0.0008, 'output': 0.004},
            'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
            'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
            'claude-2.1': {'input': 0.008, 'output': 0.024},
            'claude-2.0': {'input': 0.008, 'output': 0.024},
            'claude-instant-1.2': {'input': 0.0008, 'output': 0.0024},
        }
        
        model_pricing = pricing.get(self.model_id, {'input': 0.001, 'output': 0.003})
        
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        
        return input_cost + output_cost
    
    def supports_streaming(self) -> bool:
        """Check if model supports streaming responses."""
        return True
    
    def stream_predict(self, prompt: str, **kwargs):
        """Stream prediction responses (generator)."""
        try:
            messages, system_prompt = self._prepare_messages(prompt, kwargs.get('system_message'))
            model_params = self._extract_model_params(kwargs)
            model_params['stream'] = True
            
            api_kwargs = {
                'model': self.model_id,
                'messages': messages,
                **model_params
            }
            
            if system_prompt:
                api_kwargs['system'] = system_prompt
            
            start_time = time.time()
            
            with self.client.messages.stream(**api_kwargs) as stream:
                for chunk in stream:
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text
            
            # Log metrics after streaming is complete
            self._log_performance_metrics(start_time, None)
            
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    def supports_vision(self) -> bool:
        """Check if model supports vision/image analysis."""
        return self.SUPPORTED_MODELS.get(self.model_id, {}).get('supports_vision', False)
    
    def supports_tools(self) -> bool:
        """Check if model supports tool/function calling."""
        return self.SUPPORTED_MODELS.get(self.model_id, {}).get('supports_tools', False)
