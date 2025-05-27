"""
Model Selection and Performance Tracking for LangGraph 101.

This module provides intelligent model selection based on task requirements,
performance history, and cost considerations. It also tracks model performance
metrics for continuous optimization.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_id: str
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    avg_cost: float = 0.0
    avg_quality_score: float = 0.0
    last_used: Optional[datetime] = None
    total_tokens: int = 0
    error_rate: float = 0.0

@dataclass
class TaskRequirements:
    """Requirements for a specific task."""
    task_type: str  # 'text_generation', 'code', 'creative', 'analysis', etc.
    complexity: str  # 'simple', 'medium', 'complex'
    max_cost_per_request: Optional[float] = None
    max_latency_seconds: Optional[float] = None
    min_quality_score: Optional[float] = None
    required_features: List[str] = None  # ['vision', 'function_calling', etc.]
    context_length_needed: Optional[int] = None

class ModelPerformanceTracker:
    """Tracks and analyzes model performance metrics."""
    
    def __init__(self, data_dir: str = "analytics_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.performance_file = self.data_dir / "model_performance.json"
        self.performance_data: Dict[str, ModelPerformance] = {}
        self.load_performance_data()
    
    def load_performance_data(self):
        """Load performance data from file."""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    
                for model_key, perf_data in data.items():
                    # Convert last_used string back to datetime
                    if perf_data.get('last_used'):
                        perf_data['last_used'] = datetime.fromisoformat(perf_data['last_used'])
                    
                    self.performance_data[model_key] = ModelPerformance(**perf_data)
                    
                logger.info(f"Loaded performance data for {len(self.performance_data)} models")
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            self.performance_data = {}
    
    def save_performance_data(self):
        """Save performance data to file."""
        try:
            # Convert to dict and handle datetime serialization
            data = {}
            for model_key, perf in self.performance_data.items():
                perf_dict = asdict(perf)
                if perf_dict.get('last_used'):
                    perf_dict['last_used'] = perf_dict['last_used'].isoformat()
                data[model_key] = perf_dict
            
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def record_request(self, provider: str, model_id: str, success: bool, 
                      latency: float, cost: float = 0.0, quality_score: float = 0.0,
                      tokens_used: int = 0):
        """Record a model request and its metrics."""
        model_key = f"{provider}:{model_id}"
        
        if model_key not in self.performance_data:
            self.performance_data[model_key] = ModelPerformance(
                model_id=model_id,
                provider=provider
            )
        
        perf = self.performance_data[model_key]
        perf.total_requests += 1
        perf.last_used = datetime.now()
        perf.total_tokens += tokens_used
        
        if success:
            perf.successful_requests += 1
            
            # Update running averages
            if perf.successful_requests == 1:
                perf.avg_latency = latency
                perf.avg_cost = cost
                if quality_score > 0:
                    perf.avg_quality_score = quality_score
            else:
                # Running average calculation
                prev_count = perf.successful_requests - 1
                perf.avg_latency = (perf.avg_latency * prev_count + latency) / perf.successful_requests
                perf.avg_cost = (perf.avg_cost * prev_count + cost) / perf.successful_requests
                
                if quality_score > 0:
                    if perf.avg_quality_score == 0:
                        perf.avg_quality_score = quality_score
                    else:
                        perf.avg_quality_score = (perf.avg_quality_score * prev_count + quality_score) / perf.successful_requests
        else:
            perf.failed_requests += 1
        
        # Update error rate
        perf.error_rate = perf.failed_requests / perf.total_requests
        
        # Save data periodically
        if perf.total_requests % 10 == 0:
            self.save_performance_data()
    
    def get_model_performance(self, provider: str, model_id: str) -> Optional[ModelPerformance]:
        """Get performance data for a specific model."""
        model_key = f"{provider}:{model_id}"
        return self.performance_data.get(model_key)
    
    def get_top_models(self, task_requirements: TaskRequirements, limit: int = 5) -> List[Tuple[str, float]]:
        """Get top performing models for given requirements."""
        scored_models = []
        
        for model_key, perf in self.performance_data.items():
            if perf.total_requests < 3:  # Skip models with insufficient data
                continue
                
            score = self._calculate_model_score(perf, task_requirements)
            if score > 0:
                scored_models.append((model_key, score))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[:limit]
    
    def _calculate_model_score(self, perf: ModelPerformance, req: TaskRequirements) -> float:
        """Calculate a score for a model based on requirements."""
        if perf.error_rate > 0.1:  # Skip models with high error rates
            return 0.0
        
        score = 100.0  # Start with base score
        
        # Latency factor
        if req.max_latency_seconds and perf.avg_latency > req.max_latency_seconds:
            return 0.0  # Hard requirement
        score += max(0, (5.0 - perf.avg_latency) * 5)  # Faster is better
        
        # Cost factor
        if req.max_cost_per_request and perf.avg_cost > req.max_cost_per_request:
            return 0.0  # Hard requirement
        score += max(0, (0.01 - perf.avg_cost) * 1000)  # Cheaper is better
        
        # Quality factor
        if req.min_quality_score and perf.avg_quality_score < req.min_quality_score:
            return 0.0  # Hard requirement
        if perf.avg_quality_score > 0:
            score += perf.avg_quality_score * 20
        
        # Success rate factor
        success_rate = perf.successful_requests / perf.total_requests
        score += success_rate * 30
        
        # Recency factor (prefer recently used models)
        if perf.last_used:
            days_since_used = (datetime.now() - perf.last_used).days
            score += max(0, (30 - days_since_used) * 0.5)
        
        return score
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all model performance data."""
        if not self.performance_data:
            return {"total_models": 0, "summary": "No performance data available"}
        
        total_requests = sum(p.total_requests for p in self.performance_data.values())
        total_successful = sum(p.successful_requests for p in self.performance_data.values())
        
        avg_latencies = [p.avg_latency for p in self.performance_data.values() if p.avg_latency > 0]
        avg_costs = [p.avg_cost for p in self.performance_data.values() if p.avg_cost > 0]
        
        return {
            "total_models": len(self.performance_data),
            "total_requests": total_requests,
            "overall_success_rate": total_successful / total_requests if total_requests > 0 else 0,
            "avg_latency": statistics.mean(avg_latencies) if avg_latencies else 0,
            "avg_cost": statistics.mean(avg_costs) if avg_costs else 0,
            "models_by_provider": self._get_models_by_provider()
        }
    
    def _get_models_by_provider(self) -> Dict[str, List[str]]:
        """Group models by provider."""
        by_provider = {}
        for model_key, perf in self.performance_data.items():
            if perf.provider not in by_provider:
                by_provider[perf.provider] = []
            by_provider[perf.provider].append(perf.model_id)
        return by_provider

class ModelSelector:
    """Intelligent model selection based on task requirements and performance."""
    
    def __init__(self, performance_tracker: ModelPerformanceTracker):
        self.performance_tracker = performance_tracker
        self.model_capabilities = self._load_model_capabilities()
    
    def _load_model_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Load model capabilities from providers."""
        # This would be populated from the actual provider implementations
        return {
            "openai:gpt-4o": {
                "context_length": 128000,
                "supports_functions": True,
                "supports_vision": False,
                "task_suitability": ["code", "analysis", "creative", "text_generation"],
                "complexity_levels": ["simple", "medium", "complex"]
            },
            "anthropic:claude-3-5-sonnet-20241022": {
                "context_length": 200000,
                "supports_tools": True,
                "supports_vision": True,
                "task_suitability": ["analysis", "creative", "text_generation", "code"],
                "complexity_levels": ["medium", "complex"]
            },
            "google:gemini-2.0-flash": {
                "context_length": 1000000,
                "supports_function_calling": True,
                "supports_vision": True,
                "task_suitability": ["text_generation", "creative", "analysis"],
                "complexity_levels": ["simple", "medium", "complex"]
            }
        }
    
    def select_model(self, task_requirements: TaskRequirements) -> Optional[Tuple[str, str]]:
        """Select the best model for given requirements.
        
        Returns:
            Tuple of (provider, model_id) or None if no suitable model found
        """
        # Get eligible models based on capabilities
        eligible_models = self._filter_by_capabilities(task_requirements)
        
        if not eligible_models:
            logger.warning(f"No models found with required capabilities for task: {task_requirements.task_type}")
            return None
        
        # Get performance-based rankings
        top_performers = self.performance_tracker.get_top_models(task_requirements, limit=10)
        
        # Find intersection of eligible and top performing models
        top_eligible = []
        for model_key, score in top_performers:
            if model_key in eligible_models:
                top_eligible.append((model_key, score))
        
        if top_eligible:
            # Return the top performer
            best_model_key = top_eligible[0][0]
            provider, model_id = best_model_key.split(':', 1)
            logger.info(f"Selected model {model_id} from {provider} based on performance (score: {top_eligible[0][1]:.2f})")
            return provider, model_id
        
        # Fallback: select first eligible model
        first_eligible = list(eligible_models)[0]
        provider, model_id = first_eligible.split(':', 1)
        logger.info(f"Selected model {model_id} from {provider} as fallback (no performance data)")
        return provider, model_id
    
    def _filter_by_capabilities(self, req: TaskRequirements) -> set:
        """Filter models by capability requirements."""
        eligible = set()
        
        for model_key, capabilities in self.model_capabilities.items():
            # Check context length requirement
            if req.context_length_needed:
                if capabilities.get("context_length", 0) < req.context_length_needed:
                    continue
            
            # Check task suitability
            if req.task_type not in capabilities.get("task_suitability", []):
                continue
            
            # Check complexity level
            if req.complexity not in capabilities.get("complexity_levels", []):
                continue
            
            # Check required features
            if req.required_features:
                missing_features = []
                for feature in req.required_features:
                    if feature == "vision" and not capabilities.get("supports_vision", False):
                        missing_features.append(feature)
                    elif feature == "functions" and not capabilities.get("supports_functions", False):
                        missing_features.append(feature)
                    elif feature == "tools" and not capabilities.get("supports_tools", False):
                        missing_features.append(feature)
                    elif feature == "function_calling" and not capabilities.get("supports_function_calling", False):
                        missing_features.append(feature)
                
                if missing_features:
                    continue
            
            eligible.add(model_key)
        
        return eligible
    
    def suggest_alternatives(self, failed_model: str, task_requirements: TaskRequirements) -> List[Tuple[str, str]]:
        """Suggest alternative models when one fails."""
        # Remove the failed model from consideration temporarily
        original_perf = self.performance_tracker.performance_data.get(failed_model)
        if original_perf:
            # Temporarily mark as failed to lower its score
            original_perf.failed_requests += 1
            original_perf.total_requests += 1
            original_perf.error_rate = original_perf.failed_requests / original_perf.total_requests
        
        try:
            # Get alternative models
            alternatives = []
            top_models = self.performance_tracker.get_top_models(task_requirements, limit=3)
            
            for model_key, score in top_models:
                if model_key != failed_model:
                    provider, model_id = model_key.split(':', 1)
                    alternatives.append((provider, model_id))
            
            return alternatives
            
        finally:
            # Restore original performance data
            if original_perf:
                original_perf.failed_requests -= 1
                original_perf.total_requests -= 1
                if original_perf.total_requests > 0:
                    original_perf.error_rate = original_perf.failed_requests / original_perf.total_requests
                else:
                    original_perf.error_rate = 0.0
    
    def update_model_capabilities(self, provider: str, model_id: str, capabilities: Dict[str, Any]):
        """Update capabilities for a model."""
        model_key = f"{provider}:{model_id}"
        self.model_capabilities[model_key] = capabilities
        logger.info(f"Updated capabilities for {model_key}")
    
    def get_model_recommendations(self, task_type: str) -> List[Dict[str, Any]]:
        """Get model recommendations for a specific task type."""
        recommendations = []
        
        for model_key, capabilities in self.model_capabilities.items():
            if task_type in capabilities.get("task_suitability", []):
                provider, model_id = model_key.split(':', 1)
                perf = self.performance_tracker.get_model_performance(provider, model_id)
                
                recommendation = {
                    "provider": provider,
                    "model_id": model_id,
                    "capabilities": capabilities,
                    "performance": perf if perf else None
                }
                recommendations.append(recommendation)
        
        return recommendations
