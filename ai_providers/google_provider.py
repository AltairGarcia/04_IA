"""
Google Provider Implementation for LangGraph 101 with Enhanced Analytics.

Provides integration with Google's Gemini models using both the direct REST API
and LangChain's ChatGoogleGenerativeAI wrapper.
Supports various Gemini models including gemini-2.0-flash, gemini-1.5-pro, etc.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

import requests
import json

from model_manager import AIModel

logger = logging.getLogger(__name__)

@dataclass
class GoogleConfig:
    """Configuration for Google provider."""
    api_key: str
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    max_output_tokens: int = 8192
    timeout: float = 60.0
    max_retries: int = 3
    base_url: Optional[str] = None
    use_langchain: bool = True

class GoogleProvider(AIModel):
    """Google Gemini models provider with enhanced analytics."""
    
    SUPPORTED_MODELS = {
        'gemini-2.0-flash': {
            'context_length': 1000000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_function_calling': True,
            'supports_system_instructions': True
        },
        'gemini-1.5-pro': {
            'context_length': 2000000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_function_calling': True,
            'supports_system_instructions': True
        },
        'gemini-1.5-pro-latest': {
            'context_length': 2000000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_function_calling': True,
            'supports_system_instructions': True
        },
        'gemini-1.5-flash': {
            'context_length': 1000000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_function_calling': True,
            'supports_system_instructions': True
        },
        'gemini-1.5-flash-latest': {
            'context_length': 1000000,
            'max_output': 8192,
            'supports_vision': True,
            'supports_function_calling': True,
            'supports_system_instructions': True
        },
        'gemini-pro': {
            'context_length': 30720,
            'max_output': 2048,
            'supports_vision': False,
            'supports_function_calling': True,
            'supports_system_instructions': False
        },
        'gemini-pro-vision': {
            'context_length': 16384,
            'max_output': 2048,
            'supports_vision': True,
            'supports_function_calling': False,
            'supports_system_instructions': False
        }
    }
    
    def __init__(self, model_id: str, api_key: str, **kwargs):
        super().__init__(model_id, "google", api_key, **kwargs)
        
        if not GOOGLE_AVAILABLE:
            logger.warning("Google AI packages not fully available. Using fallback REST API.")
        
        self.config = GoogleConfig(
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7),
            top_k=kwargs.get('top_k', 40),
            top_p=kwargs.get('top_p', 0.95),
            max_output_tokens=kwargs.get('max_output_tokens', 8192),
            timeout=kwargs.get('timeout', 60.0),
            max_retries=kwargs.get('max_retries', 3),
            base_url=kwargs.get('base_url'),
            use_langchain=kwargs.get('use_langchain', True) and GOOGLE_AVAILABLE
        )
        
        # Initialize Google AI client
        if GOOGLE_AVAILABLE:
            genai.configure(api_key=self.config.api_key)
            
        # Initialize LangChain client if available
        if self.config.use_langchain and GOOGLE_AVAILABLE:
            try:
                self.langchain_client = ChatGoogleGenerativeAI(
                    model=model_id,
                    google_api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain Google client: {e}")
                self.langchain_client = None
        else:
            self.langchain_client = None
            
        # REST API fallback configuration
        self.rest_base_url = self.config.base_url or f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        
        # Validate model
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported models list. Proceeding anyway.")
        
        logger.info(f"Google provider initialized for model: {model_id}")
    
    def predict(self, prompt: str, **kwargs) -> str:
        """Generate a synchronous prediction using Google Gemini."""
        try:
            # Try LangChain first if available
            if self.langchain_client:
                return self._predict_langchain(prompt, **kwargs)
            else:
                return self._predict_rest_api(prompt, **kwargs)
                
        except Exception as e:
            logger.error(f"Google prediction failed: {e}")
            # Log failed prediction
            self._log_failed_prediction(prompt, str(e))
            raise
    
    async def apredict(self, prompt: str, **kwargs) -> str:
        """Generate an asynchronous prediction using Google Gemini."""
        try:
            # For async, we'll use the REST API approach
            return await self._apredict_rest_api(prompt, **kwargs)
            
        except Exception as e:
            logger.error(f"Google async prediction failed: {e}")
            # Log failed prediction
            self._log_failed_prediction(prompt, str(e))
            raise
    
    def _predict_langchain(self, prompt: str, **kwargs) -> str:
        """Use LangChain for prediction."""
        start_time = time.time()
        
        # Prepare system message if provided
        system_message = kwargs.get('system_message')
        
        if system_message and self.SUPPORTED_MODELS.get(self.model_id, {}).get('supports_system_instructions'):
            # Use system instructions for supported models
            full_prompt = f"System: {system_message}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        
        # Make API call with timeout config
        response = self.langchain_client.invoke(
            full_prompt,
            config={"request_timeout": self.config.timeout}
        )
        
        # Log performance metrics
        self._log_performance_metrics(start_time, len(prompt.split()), len(response.content.split()), prompt, response.content)
        
        return response.content
    
    def _predict_rest_api(self, prompt: str, **kwargs) -> str:
        """Use REST API for prediction."""
        start_time = time.time()
        
        # Prepare request payload
        payload = self._prepare_rest_payload(prompt, kwargs)
        
        # Make API call with retries
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.rest_base_url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.config.api_key
                    },
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = self._extract_content_from_response(data)
                    
                    # Log performance metrics
                    self._log_performance_metrics(start_time, len(prompt.split()), len(content.split()), prompt, content)
                    
                    return content
                    
                elif response.status_code in (429, 500, 502, 503, 504):
                    # Retry on transient errors
                    logger.warning(f"Google API transient error (status {response.status_code}), attempt {attempt}/{self.config.max_retries}")
                    if attempt < self.config.max_retries:
                        time.sleep(2 * attempt)
                        continue
                    
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Google API network error: {e}, attempt {attempt}/{self.config.max_retries}")
                if attempt < self.config.max_retries:
                    time.sleep(2 * attempt)
                    continue
                raise
        
        raise Exception(f"Google API failed after {self.config.max_retries} attempts")
    
    async def _apredict_rest_api(self, prompt: str, **kwargs) -> str:
        """Async REST API prediction."""
        import aiohttp
        
        start_time = time.time()
        payload = self._prepare_rest_payload(prompt, kwargs)
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    async with session.post(
                        self.rest_base_url,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.config.api_key
                        },
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            content = self._extract_content_from_response(data)
                            
                            # Log performance metrics
                            self._log_performance_metrics(start_time, len(prompt.split()), len(content.split()), prompt, content)
                            
                            return content
                            
                        elif response.status in (429, 500, 502, 503, 504):
                            logger.warning(f"Google API transient error (status {response.status}), attempt {attempt}/{self.config.max_retries}")
                            if attempt < self.config.max_retries:
                                await asyncio.sleep(2 * attempt)
                                continue
                        
                        response.raise_for_status()
                        
                except Exception as e:
                    logger.warning(f"Google API async error: {e}, attempt {attempt}/{self.config.max_retries}")
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(2 * attempt)
                        continue
                    raise
        
        raise Exception(f"Google async API failed after {self.config.max_retries} attempts")
    
    def _prepare_rest_payload(self, prompt: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare REST API payload."""
        # Handle system message for supported models
        system_message = kwargs.get('system_message')
        
        if system_message and self.SUPPORTED_MODELS.get(self.model_id, {}).get('supports_system_instructions'):
            # For newer models, use system instructions
            payload = {
                "system_instruction": {"parts": [{"text": system_message}]},
                "contents": [{"parts": [{"text": prompt}]}]
            }
        else:
            # For older models or when system message should be part of content
            if system_message:
                combined_prompt = f"System: {system_message}\n\nUser: {prompt}"
            else:
                combined_prompt = prompt
            payload = {
                "contents": [{"parts": [{"text": combined_prompt}]}]
            }
        
        # Add generation config
        generation_config = {
            "temperature": kwargs.get('temperature', self.config.temperature),
            "topK": kwargs.get('top_k', self.config.top_k),
            "topP": kwargs.get('top_p', self.config.top_p),
            "maxOutputTokens": kwargs.get('max_output_tokens', self.config.max_output_tokens)
        }
        
        # Add safety settings if specified
        if 'safety_settings' in kwargs:
            payload["safetySettings"] = kwargs['safety_settings']
        
        payload["generationConfig"] = generation_config
        
        return payload
    
    def _extract_content_from_response(self, data: Dict[str, Any]) -> str:
        """Extract text content from Google API response."""
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to extract content from Google response: {e}")
            logger.error(f"Response data: {data}")
            raise ValueError(f"Invalid response format from Google API: {e}")
    
    def _log_performance_metrics(self, start_time: float, prompt_tokens: int, completion_tokens: int, prompt: str = "", response: str = ""):
        """Log performance metrics for enhanced analytics."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Import enhanced analytics logger
        try:
            from analytics.analytics_logger import AnalyticsLogger
            analytics_logger = AnalyticsLogger()
            
            # Log model performance with enhanced metrics
            analytics_logger.log_model_performance({
                'provider': 'google',
                'model': self.model_id,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'response_time_ms': duration_ms,
                'cost_estimate': self.estimate_cost(prompt_tokens, completion_tokens),
                'success': True,
                'model_capabilities': {
                    'supports_vision': self.supports_feature('vision'),
                    'supports_function_calling': self.supports_feature('function_calling'),
                    'supports_system_instructions': self.supports_feature('system_instructions'),
                    'supports_streaming': self.supports_feature('streaming'),
                    'context_length': self.SUPPORTED_MODELS.get(self.model_id, {}).get('context_length', 0),
                    'max_output': self.SUPPORTED_MODELS.get(self.model_id, {}).get('max_output', 0)
                }
            })
            
            # Log API call event
            analytics_logger.log_api_call({
                'provider': 'google',
                'model': self.model_id,
                'method': 'generate_content',
                'response_time_ms': duration_ms,
                'status': 'success',
                'tokens_used': prompt_tokens + completion_tokens,
                'cost': self.estimate_cost(prompt_tokens, completion_tokens)
            })
            
            # Log user interaction
            analytics_logger.log_user_interaction({
                'interaction_type': 'model_query',
                'model_provider': 'google',
                'model_name': self.model_id,
                'input_length': len(prompt),
                'output_length': len(response),
                'processing_time_ms': duration_ms,
                'success': True
            })
            
        except ImportError:
            logger.warning("Enhanced analytics logger not available for performance tracking")
    
    def _log_failed_prediction(self, prompt: str, error_message: str):
        """Log failed prediction attempts for analytics."""
        try:
            from analytics.analytics_logger import AnalyticsLogger
            analytics_logger = AnalyticsLogger()
            
            # Log failed API call
            analytics_logger.log_api_call({
                'provider': 'google',
                'model': self.model_id,
                'method': 'generate_content',
                'status': 'failed',
                'error': error_message
            })
            
            # Log failed user interaction
            analytics_logger.log_user_interaction({
                'interaction_type': 'model_query',
                'model_provider': 'google',
                'model_name': self.model_id,
                'input_length': len(prompt),
                'success': False,
                'error': error_message
            })
            
        except ImportError:
            logger.warning("Enhanced analytics logger not available for error tracking")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = self.get_details()
        info.update(self.SUPPORTED_MODELS.get(self.model_id, {}))
        info['provider_specific'] = {
            'api_version': 'v1beta',
            'billing_model': 'token-based',
            'rate_limits': 'varies by model',
            'safety_features': 'built_in_safety_filters',
            'supports_streaming': True,
            'langchain_integration': self.config.use_langchain
        }
        return info
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for a given token usage (approximate)."""
        # Google Gemini pricing (approximate, as of 2024)
        pricing = {
            'gemini-2.0-flash': {'input': 0.075, 'output': 0.30},  # per 1M tokens
            'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
            'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
            'gemini-pro': {'input': 0.50, 'output': 1.50}
        }
        
        model_pricing = pricing.get(self.model_id, pricing['gemini-1.5-flash'])
        
        input_cost = (prompt_tokens / 1_000_000) * model_pricing['input']
        output_cost = (completion_tokens / 1_000_000) * model_pricing['output']
        
        return input_cost + output_cost
    
    def stream_predict(self, prompt: str, **kwargs):
        """Generate streaming predictions (if supported)."""
        if not GOOGLE_AVAILABLE:
            raise NotImplementedError("Streaming requires google-generativeai package")
        
        start_time = time.time()
        
        try:
            model = genai.GenerativeModel(self.model_id)
            
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', self.config.temperature),
                top_k=kwargs.get('top_k', self.config.top_k),
                top_p=kwargs.get('top_p', self.config.top_p),
                max_output_tokens=kwargs.get('max_output_tokens', self.config.max_output_tokens)
            )
            
            # Generate streaming response
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Log streaming metrics after completion
            self._log_performance_metrics(start_time, len(prompt.split()), len(full_response.split()), prompt, full_response)
                    
        except Exception as e:
            logger.error(f"Google streaming prediction failed: {e}")
            self._log_failed_prediction(prompt, str(e))
            raise
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the model supports a specific feature."""
        model_info = self.SUPPORTED_MODELS.get(self.model_id, {})
        
        feature_map = {
            'vision': 'supports_vision',
            'function_calling': 'supports_function_calling',
            'system_instructions': 'supports_system_instructions',
            'streaming': True  # All models support streaming
        }
        
        if feature in feature_map:
            if feature == 'streaming':
                return True
            return model_info.get(feature_map[feature], False)
        
        return False

    def get_safety_settings(self, safety_level: str = 'default') -> List[Dict[str, Any]]:
        """Get safety settings for content filtering."""
        safety_levels = {
            'strict': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}
            ],
            'default': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ],
            'permissive': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }
        return safety_levels.get(safety_level, safety_levels['default'])

    def batch_predict(self, prompts: List[str], **kwargs) -> List[str]:
        """Process multiple prompts in batch for efficiency."""
        if not prompts:
            return []
        
        # For batch processing, we'll process them sequentially with optimizations
        # Google's API doesn't support true batch processing like OpenAI
        results = []
        batch_start_time = time.time()
        
        try:
            from analytics.analytics_logger import AnalyticsLogger
            analytics_logger = AnalyticsLogger()
            
            # Log batch processing start
            analytics_logger.log_batch_operation({
                'provider': 'google',
                'model': self.model_id,
                'batch_size': len(prompts),
                'operation': 'batch_predict_start'
            })
            
        except ImportError:
            pass
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.predict(prompt, **kwargs)
                results.append(result)
                
                # Add small delay to avoid rate limiting
                if i < len(prompts) - 1:  # Don't delay after last item
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Batch prediction failed for prompt {i}: {e}")
                results.append(f"Error: {str(e)}")
        
        # Log batch completion
        try:
            batch_duration = time.time() - batch_start_time
            analytics_logger.log_batch_operation({
                'provider': 'google',
                'model': self.model_id,
                'batch_size': len(prompts),
                'operation': 'batch_predict_complete',
                'duration_seconds': batch_duration,
                'success_count': len([r for r in results if not r.startswith('Error:')]),
                'error_count': len([r for r in results if r.startswith('Error:')])
            })
        except:
            pass
        
        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider instance."""
        try:
            from analytics.analytics_logger import AnalyticsLogger
            analytics_logger = AnalyticsLogger()
            
            # Get recent stats for this provider/model combination
            stats = analytics_logger.get_provider_stats('google', self.model_id)
            
            return {
                'provider': 'google',
                'model': self.model_id,
                'total_requests': stats.get('total_requests', 0),
                'total_tokens': stats.get('total_tokens', 0),
                'average_response_time': stats.get('avg_response_time', 0),
                'error_rate': stats.get('error_rate', 0),
                'estimated_cost': stats.get('total_cost', 0),
                'last_used': stats.get('last_used'),
                'capabilities': {
                    'vision': self.supports_feature('vision'),
                    'function_calling': self.supports_feature('function_calling'),
                    'system_instructions': self.supports_feature('system_instructions'),
                    'streaming': self.supports_feature('streaming')
                }
            }
        except ImportError:
            return {
                'provider': 'google',
                'model': self.model_id,
                'message': 'Analytics not available'
            }

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics for monitoring dashboard."""
        try:
            from analytics.real_time_analytics import RealTimeAnalytics
            real_time_analytics = RealTimeAnalytics()
            
            # Get current metrics for this provider
            current_metrics = real_time_analytics.get_current_metrics()
            
            # Filter metrics for this provider
            provider_metrics = {
                'requests_per_minute': 0,
                'avg_response_time': 0,
                'error_rate': 0,
                'active_sessions': 0
            }
            
            # Extract provider-specific metrics from current_metrics
            if hasattr(current_metrics, 'provider_usage'):
                google_usage = current_metrics.provider_usage.get('google', {})
                provider_metrics.update({
                    'requests_per_minute': google_usage.get('requests_per_minute', 0),
                    'avg_response_time': google_usage.get('avg_response_time', 0),
                    'error_rate': google_usage.get('error_rate', 0)
                })
            
            return provider_metrics
            
        except ImportError:
            return {'message': 'Real-time analytics not available'}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the Google provider."""
        from datetime import datetime
        
        health_status = {
            'provider': 'google',
            'model': self.model_id,
            'status': 'unknown',
            'response_time': None,
            'last_check': datetime.now().isoformat(),
            'errors': []
        }
        
        try:
            # Simple health check with a minimal request
            start_time = time.time()
            
            test_prompt = "Hello"
            if self.langchain_client:
                # Use LangChain client for health check
                response = self.langchain_client.invoke(test_prompt)
                if response and response.content:
                    health_status['status'] = 'healthy'
                else:
                    health_status['status'] = 'degraded'
                    health_status['errors'].append('Empty response from model')
            else:
                # Use REST API for health check
                payload = self._prepare_rest_payload(test_prompt, {})
                response = requests.post(
                    self.rest_base_url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.config.api_key
                    },
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    health_status['status'] = 'healthy'
                else:
                    health_status['status'] = 'unhealthy'
                    health_status['errors'].append(f'HTTP {response.status_code}')
            
            health_status['response_time'] = (time.time() - start_time) * 1000
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append(str(e))
            logger.error(f"Google provider health check failed: {e}")
        
        # Log health check result
        try:
            from analytics.analytics_logger import AnalyticsLogger
            analytics_logger = AnalyticsLogger()
            analytics_logger.log_provider_health({
                'provider': 'google',
                'model': self.model_id,
                'status': health_status['status'],
                'response_time': health_status['response_time'],
                'errors': health_status['errors']
            })
        except ImportError:
            pass
        
        return health_status
