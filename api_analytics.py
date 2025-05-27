"""
API Analytics module for LangGraph 101 project.

This module provides functionality for tracking API usage and performance metrics.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import time
import logging
from datetime import datetime, timedelta
from functools import wraps

# Initialize logger
logger = logging.getLogger(__name__)

class APIAnalytics:
    """Track and analyze API usage and performance."""

    def __init__(self, analytics_dir: Optional[str] = None):
        """Initialize the APIAnalytics tracker.

        Args:
            analytics_dir: Directory to store analytics data
        """
        self.analytics_dir = analytics_dir or os.path.join(os.path.dirname(__file__), "analytics_data")
        os.makedirs(self.analytics_dir, exist_ok=True)

        self.daily_usage_file = os.path.join(self.analytics_dir, "daily_usage.json")
        self.performance_file = os.path.join(self.analytics_dir, "api_performance.json")

        # Initialize data files if they don't exist
        self._initialize_data_files()

    def _initialize_data_files(self) -> None:
        """Initialize data files if they don't exist."""
        today = datetime.now().strftime("%Y-%m-%d")
        # Initialize daily usage file
        if not os.path.exists(self.daily_usage_file):
            initial_data = {
                today: {
                    "gemini": {"calls": 0, "tokens": 0, "errors": 0, "estimated_cost": 0.0},
                    "elevenlabs": {"calls": 0, "seconds": 0, "errors": 0, "estimated_cost": 0.0},
                    "dalle": {"calls": 0, "images": 0, "errors": 0, "estimated_cost": 0.0},
                    "stabilityai": {"calls": 0, "images": 0, "errors": 0, "estimated_cost": 0.0},
                    "assemblyai": {"calls": 0, "seconds": 0, "errors": 0, "estimated_cost": 0.0},                    "deepgram": {"calls": 0, "seconds": 0, "errors": 0, "estimated_cost": 0.0},
                    "youtube_data": {"calls": 0, "queries": 0, "errors": 0},
                    "tavily": {"calls": 0, "queries": 0, "errors": 0, "estimated_cost": 0.0},
                    "arxiv": {"calls": 0, "queries": 0, "errors": 0},
                    "wikipedia": {"calls": 0, "queries": 0, "errors": 0},
                    "newsapi": {"calls": 0, "queries": 0, "errors": 0, "estimated_cost": 0.0},
                    "openweather": {"calls": 0, "queries": 0, "errors": 0, "estimated_cost": 0.0}
                }
            }
            with open(self.daily_usage_file, "w") as f:
                json.dump(initial_data, f, indent=2)
          # Initialize performance file
        if not os.path.exists(self.performance_file):
            initial_data = {
                "gemini": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "elevenlabs": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "dalle": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "stabilityai": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "assemblyai": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "deepgram": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "youtube_data": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "tavily": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "newsapi": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0},
                "openweather": {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0}
            }
            with open(self.performance_file, "w") as f:
                json.dump(initial_data, f, indent=2)

    def track_api_call(self, api_name: str, success: bool, latency: float, **kwargs) -> None:
        """Track an API call with its performance metrics.

        Args:
            api_name: Name of the API (e.g., 'gemini', 'elevenlabs')
            success: Whether the call was successful
            latency: Time taken for the API call (in seconds)
            **kwargs: Additional metrics to track (tokens, seconds, images, etc.)
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Update daily usage
        try:
            with open(self.daily_usage_file, "r") as f:
                daily_data = json.load(f)

            # Add today if it doesn't exist
            if today not in daily_data:                daily_data[today] = {
                    "gemini": {"calls": 0, "tokens": 0, "errors": 0, "estimated_cost": 0.0},
                    "elevenlabs": {"calls": 0, "seconds": 0, "errors": 0, "estimated_cost": 0.0},
                    "dalle": {"calls": 0, "images": 0, "errors": 0, "estimated_cost": 0.0},
                    "stabilityai": {"calls": 0, "images": 0, "errors": 0, "estimated_cost": 0.0},
                    "assemblyai": {"calls": 0, "seconds": 0, "errors": 0, "estimated_cost": 0.0},
                    "deepgram": {"calls": 0, "seconds": 0, "errors": 0, "estimated_cost": 0.0},
                    "youtube_data": {"calls": 0, "queries": 0, "errors": 0},
                    "tavily": {"calls": 0, "queries": 0, "errors": 0},
                    "arxiv": {"calls": 0, "queries": 0, "errors": 0},
                    "wikipedia": {"calls": 0, "queries": 0, "errors": 0},
                    "newsapi": {"calls": 0, "queries": 0, "errors": 0},
                    "openweather": {"calls": 0, "queries": 0, "errors": 0}
                }
              # Initialize API if it doesn't exist
            if api_name not in daily_data[today]:
                if api_name in ["gemini", "dalle", "stabilityai", "elevenlabs", "assemblyai", "deepgram", "newsapi", "openweather", "tavily"]:
                    daily_data[today][api_name] = {"calls": 0, "errors": 0, "estimated_cost": 0.0}
                else:
                    daily_data[today][api_name] = {"calls": 0, "errors": 0}

                # Add specific counters based on API type
                if api_name in ["gemini"]:
                    daily_data[today][api_name]["tokens"] = 0
                elif api_name in ["elevenlabs", "assemblyai", "deepgram"]:
                    daily_data[today][api_name]["seconds"] = 0
                elif api_name in ["dalle", "stabilityai"]:
                    daily_data[today][api_name]["images"] = 0
                else:
                    daily_data[today][api_name]["queries"] = 0

            # Update metrics
            daily_data[today][api_name]["calls"] += 1
            if not success:
                daily_data[today][api_name]["errors"] += 1

            # Update specific metrics
            for key, value in kwargs.items():
                if key in daily_data[today][api_name]:
                    daily_data[today][api_name][key] += value

            # Update estimated costs
            if "estimated_cost" in daily_data[today][api_name]:
                cost = self._calculate_estimated_cost(api_name, kwargs)
                daily_data[today][api_name]["estimated_cost"] += cost

            # Save data
            with open(self.daily_usage_file, "w") as f:
                json.dump(daily_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update daily usage analytics: {str(e)}")

        # Update performance metrics
        try:
            with open(self.performance_file, "r") as f:
                perf_data = json.load(f)

            # Initialize API if it doesn't exist
            if api_name not in perf_data:
                perf_data[api_name] = {"avg_latency": 0.0, "calls": 0, "success_rate": 1.0}

            # Update metrics using weighted average
            current_calls = perf_data[api_name]["calls"]
            current_latency = perf_data[api_name]["avg_latency"]
            current_success_rate = perf_data[api_name]["success_rate"]

            # Update latency with weighted average
            total_calls = current_calls + 1
            perf_data[api_name]["avg_latency"] = (current_latency * current_calls + latency) / total_calls

            # Update success rate
            success_count = current_success_rate * current_calls + (1 if success else 0)
            perf_data[api_name]["success_rate"] = success_count / total_calls

            # Update call count
            perf_data[api_name]["calls"] = total_calls

            # Save data
            with open(self.performance_file, "w") as f:
                json.dump(perf_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update performance analytics: {str(e)}")

    def _calculate_estimated_cost(self, api_name: str, metrics: Dict[str, Any]) -> float:
        """Calculate estimated cost for an API call.

        Args:
            api_name: Name of the API
            metrics: Metrics for the API call

        Returns:
            Estimated cost in USD        """
        # These are approximate costs, may need regular updates
        if api_name == "gemini":
            tokens = metrics.get("tokens", 0)
            # Gemini Pro: ~$0.00025 per 1K tokens (input+output)
            return (tokens / 1000) * 0.00025

        elif api_name == "elevenlabs":
            seconds = metrics.get("seconds", 0)
            # Typical cost is ~$0.006 per second
            return seconds * 0.006

        elif api_name == "dalle":
            images = metrics.get("images", 0)
            # DALL-E 3: ~$0.04 per image at 1024×1024
            return images * 0.04

        elif api_name == "stabilityai":
            images = metrics.get("images", 0)
            # Stable Diffusion API: ~$0.02 per image
            return images * 0.02

        elif api_name == "assemblyai":
            seconds = metrics.get("seconds", 0)
            # AssemblyAI: ~$0.00025 per second
            return seconds * 0.00025

        elif api_name == "deepgram":
            seconds = metrics.get("seconds", 0)
            # Deepgram: ~$0.0004 per second for premium model
            return seconds * 0.0004

        elif api_name == "newsapi":
            queries = metrics.get("queries", 0)
            # NewsAPI: ~$0.005 per query
            return queries * 0.005

        elif api_name == "openweather":
            queries = metrics.get("queries", 0)
            # OpenWeather API: ~$0.0001 per query
            return queries * 0.0001

        elif api_name == "tavily":
            queries = metrics.get("queries", 0)
            # Tavily API: ~$0.01 per search query
            return queries * 0.01

        return 0.0

    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get usage summary for the specified number of days.

        Args:
            days: Number of days to include in summary

        Returns:
            Dictionary with usage summary
        """
        try:
            with open(self.daily_usage_file, "r") as f:
                daily_data = json.load(f)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days-1)

            # Filter data for the specified date range
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_range.append(date_str)
                current_date += timedelta(days=1)

            # Initialize summary data
            summary = {
                "date_range": date_range,
                "total_calls": 0,
                "total_cost": 0.0,
                "by_api": {},
                "by_date": {}
            }

            # Calculate totals
            for date_str in date_range:
                if date_str in daily_data:
                    date_total_calls = 0
                    date_total_cost = 0.0

                    for api_name, metrics in daily_data[date_str].items():
                        # Initialize API in summary if not exists
                        if api_name not in summary["by_api"]:
                            summary["by_api"][api_name] = {
                                "calls": 0,
                                "errors": 0,
                                "estimated_cost": 0.0
                            }

                        # Update API totals
                        calls = metrics.get("calls", 0)
                        errors = metrics.get("errors", 0)
                        cost = metrics.get("estimated_cost", 0.0)

                        summary["by_api"][api_name]["calls"] += calls
                        summary["by_api"][api_name]["errors"] += errors
                        summary["by_api"][api_name]["estimated_cost"] += cost

                        # Add specific metrics if they exist
                        for key in ["tokens", "seconds", "images", "queries"]:
                            if key in metrics:
                                if key not in summary["by_api"][api_name]:
                                    summary["by_api"][api_name][key] = 0
                                summary["by_api"][api_name][key] += metrics[key]

                        # Update totals
                        date_total_calls += calls
                        date_total_cost += cost

                    # Add daily summary
                    summary["by_date"][date_str] = {
                        "calls": date_total_calls,
                        "estimated_cost": date_total_cost
                    }

                    # Update overall totals
                    summary["total_calls"] += date_total_calls
                    summary["total_cost"] += date_total_cost

            # Calculate success rates
            for api_name, metrics in summary["by_api"].items():
                calls = metrics.get("calls", 0)
                errors = metrics.get("errors", 0)

                if calls > 0:
                    metrics["success_rate"] = (calls - errors) / calls
                else:
                    metrics["success_rate"] = 1.0

            return summary

        except Exception as e:
            logger.error(f"Failed to generate usage summary: {str(e)}")
            return {
                "error": str(e),
                "date_range": [],
                "total_calls": 0,
                "total_cost": 0.0,
                "by_api": {},
                "by_date": {}
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all APIs.

        Returns:
            Dictionary with performance summary
        """
        try:
            with open(self.performance_file, "r") as f:
                perf_data = json.load(f)

            # Calculate overall averages
            total_calls = sum(api["calls"] for api in perf_data.values())
            weighted_latency = sum(api["avg_latency"] * api["calls"] for api in perf_data.values())
            weighted_success_rate = sum(api["success_rate"] * api["calls"] for api in perf_data.values())

            overall_latency = weighted_latency / total_calls if total_calls > 0 else 0
            overall_success_rate = weighted_success_rate / total_calls if total_calls > 0 else 1.0

            return {
                "overall": {
                    "avg_latency": overall_latency,
                    "success_rate": overall_success_rate,
                    "total_calls": total_calls
                },
                "by_api": perf_data
            }

        except Exception as e:
            logger.error(f"Failed to generate performance summary: {str(e)}")
            return {
                "error": str(e),
                "overall": {
                    "avg_latency": 0.0,
                    "success_rate": 0.0,
                    "total_calls": 0
                },
                "by_api": {}
            }


# Decorator to track API calls
def track_api_usage(api_name: str):
    """Decorator to track API usage and performance.

    Args:
        api_name: Name of the API

    Returns:
        Decorated function
    """
    analytics = APIAnalytics()

    # Add a class attribute for API name override (used for dynamic providers)
    track_api_usage.override_api_name = None

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            metrics = {}

            # Use overridden API name if provided (useful for dynamic providers like transcription)
            current_api_name = track_api_usage.override_api_name or api_name
            # Reset the override after using it
            track_api_usage.override_api_name = None

            try:
                result = func(*args, **kwargs)
                  # Extract metrics from result if available
                if isinstance(result, dict):
                    if current_api_name == "gemini" and "token_count" in result:
                        metrics["tokens"] = result["token_count"]
                    elif current_api_name in ["elevenlabs", "assemblyai", "deepgram"] and "duration_seconds" in result:
                        metrics["seconds"] = result["duration_seconds"]
                    elif current_api_name in ["dalle", "stabilityai"] and "images" in result:
                        metrics["images"] = result.get("images", 1)  # Default to 1 if not specified
                    elif current_api_name in ["tavily", "newsapi", "openweather", "wikipedia", "arxiv"] and "queries" in result:
                        metrics["queries"] = result["queries"]
                    elif "response" in result:  # For API tools that return response wrapper
                        metrics["queries"] = result.get("queries", 1)
                    else:
                        metrics["queries"] = 1

                return result

            except Exception as e:
                success = False
                raise

            finally:
                latency = time.time() - start_time
                analytics.track_api_call(api_name, success, latency, **metrics)

        return wrapper

    return decorator


# Singleton instance for use throughout the application
_api_analytics_instance = None

def get_api_analytics() -> APIAnalytics:
    """Get the singleton APIAnalytics instance.

    Returns:
        APIAnalytics instance
    """
    global _api_analytics_instance

    if _api_analytics_instance is None:
        _api_analytics_instance = APIAnalytics()

    return _api_analytics_instance
