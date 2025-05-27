"""
Workflow Optimization Enhancement for Content Creation System

This module provides advanced optimizations for the content creation workflow,
including batch processing, intelligent caching, parallel execution, and
performance monitoring.
"""

import asyncio
import concurrent.futures
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json
import os
from functools import wraps

from performance_optimization import (
    cached, async_cached, get_cache, get_task_queue,
    TaskQueue, PerformanceCache
)

logger = logging.getLogger(__name__)

class WorkflowOptimizer:
    """Advanced optimization for content creation workflows."""

    def __init__(self, content_creator):
        """Initialize the workflow optimizer.

        Args:
            content_creator: ContentCreator instance to optimize
        """
        self.content_creator = content_creator
        self.cache = get_cache()
        self.task_queue = get_task_queue(max_workers=6)
        self.task_queue.start()

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "parallel_operations": 0,
            "batch_operations": 0,
            "average_response_time": 0.0,
            "api_usage": {}
        }

    def performance_monitor(self, operation_name: str):
        """Decorator to monitor performance of operations."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                self.metrics["total_requests"] += 1

                try:
                    result = await func(*args, **kwargs)

                    # Update metrics
                    execution_time = time.time() - start_time
                    self._update_average_response_time(execution_time)

                    logger.info(f"{operation_name} completed in {execution_time:.2f}s")
                    return result

                except Exception as e:
                    logger.error(f"{operation_name} failed: {str(e)}")
                    raise

            return wrapper
        return decorator

    def _update_average_response_time(self, new_time: float):
        """Update the average response time metric."""
        current_avg = self.metrics["average_response_time"]
        total_requests = self.metrics["total_requests"]

        # Calculate new average
        self.metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + new_time) / total_requests
        )

    @async_cached(expiry=1800)  # Cache for 30 minutes
    async def optimized_image_search(self, query: str, sources: List[str] = None,
                                   count_per_source: int = 5) -> Dict[str, Any]:
        """Optimized image search with parallel API calls and intelligent caching.

        Args:
            query: Search query
            sources: List of sources to search (default: ['pexels', 'pixabay'])
            count_per_source: Number of images per source

        Returns:
            Aggregated results from all sources
        """
        if sources is None:
            sources = ['pexels', 'pixabay']

        logger.info(f"Starting optimized image search: '{query}' from {sources}")

        # Parallel image search across sources
        tasks = []
        for source in sources:
            task = asyncio.create_task(
                self._async_image_search(query, source, count_per_source)
            )
            tasks.append((source, task))

        # Wait for all searches to complete
        results = {"query": query, "sources": {}, "total_count": 0, "aggregated_results": []}

        for source, task in tasks:
            try:
                source_results = await task
                results["sources"][source] = {
                    "count": len(source_results),
                    "images": source_results
                }
                results["aggregated_results"].extend(source_results)
                results["total_count"] += len(source_results)

            except Exception as e:
                logger.error(f"Error searching {source}: {str(e)}")
                results["sources"][source] = {"error": str(e), "count": 0}

        self.metrics["parallel_operations"] += 1
        logger.info(f"Optimized image search completed: {results['total_count']} total images")

        return results

    async def _async_image_search(self, query: str, source: str, count: int) -> List[Dict]:
        """Async wrapper for image search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.content_creator.search_images,
            query, source, count
        )

    @async_cached(expiry=3600)  # Cache for 1 hour
    async def optimized_content_bundle(self, topic: str, include_images: bool = True,
                                     include_web_research: bool = True,
                                     include_script: bool = True) -> Dict[str, Any]:
        """Generate a complete content bundle with parallel processing.

        Args:
            topic: Content topic
            include_images: Whether to include image search
            include_web_research: Whether to include web research
            include_script: Whether to include script generation

        Returns:
            Complete content bundle
        """
        logger.info(f"Creating optimized content bundle for: '{topic}'")
        start_time = time.time()

        # Prepare parallel tasks
        tasks = {}

        if include_script:
            tasks["script"] = asyncio.create_task(
                self._async_generate_script(topic)
            )

        if include_web_research:
            tasks["web_research"] = asyncio.create_task(
                self._async_web_search(f"{topic} latest trends insights")
            )

        if include_images:
            tasks["images"] = asyncio.create_task(
                self.optimized_image_search(topic, count_per_source=3)
            )

        # Wait for all tasks to complete
        results = {
            "topic": topic,
            "created_at": datetime.now().isoformat(),
            "processing_time": 0,
            "components": {}
        }

        for component, task in tasks.items():
            try:
                results["components"][component] = await task
                logger.info(f"Completed {component} generation")
            except Exception as e:
                logger.error(f"Error generating {component}: {str(e)}")
                results["components"][component] = {"error": str(e)}

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time

        self.metrics["parallel_operations"] += 1
        logger.info(f"Content bundle created in {processing_time:.2f}s")

        return results

    async def _async_generate_script(self, topic: str) -> Dict[str, Any]:
        """Async wrapper for script generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.content_creator.generate_script,
            topic, "engaging and informative", 3
        )

    async def _async_web_search(self, query: str) -> Dict[str, Any]:
        """Async wrapper for web search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.content_creator.search_web,
            query
        )

    @async_cached(expiry=600)  # Cache for 10 minutes
    async def batch_image_search(self, queries: List[str],
                               source: str = "pexels",
                               count_per_query: int = 3) -> Dict[str, Any]:
        """Perform batch image searches with optimized API usage.

        Args:
            queries: List of search queries
            source: Image source to use
            count_per_query: Number of images per query

        Returns:
            Batch search results
        """
        logger.info(f"Starting batch image search for {len(queries)} queries")

        # Create semaphore to limit concurrent API calls (avoid rate limiting)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent API calls

        async def search_with_semaphore(query: str) -> Tuple[str, List[Dict]]:
            async with semaphore:
                try:
                    images = await self._async_image_search(query, source, count_per_query)
                    return query, images
                except Exception as e:
                    logger.error(f"Batch search error for '{query}': {str(e)}")
                    return query, []

        # Execute all searches concurrently with rate limiting
        tasks = [search_with_semaphore(query) for query in queries]
        search_results = await asyncio.gather(*tasks)

        # Organize results
        results = {
            "batch_size": len(queries),
            "source": source,
            "total_images": 0,
            "queries": {}
        }

        for query, images in search_results:
            results["queries"][query] = {
                "count": len(images),
                "images": images
            }
            results["total_images"] += len(images)

        self.metrics["batch_operations"] += 1
        logger.info(f"Batch search completed: {results['total_images']} total images")

        return results

    def smart_cache_warmup(self, common_queries: List[str]):
        """Pre-warm cache with common queries to improve response times.

        Args:
            common_queries: List of frequently used search queries
        """
        logger.info(f"Starting cache warmup for {len(common_queries)} queries")

        async def warmup_task():
            for query in common_queries:
                try:
                    # Pre-cache image searches
                    await self.optimized_image_search(query, count_per_source=2)

                    # Pre-cache web searches
                    await self._async_web_search(f"{query} trends")

                    logger.debug(f"Cache warmed for: '{query}'")

                except Exception as e:
                    logger.warning(f"Cache warmup failed for '{query}': {str(e)}")

                # Small delay to avoid overwhelming APIs
                await asyncio.sleep(0.5)

        # Run warmup in background
        asyncio.create_task(warmup_task())
        logger.info("Cache warmup started in background")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_stats = self.cache.get_stats()

        return {
            "workflow_metrics": self.metrics.copy(),
            "cache_metrics": cache_stats,
            "cache_hit_rate": (
                cache_stats["hits"] / max(cache_stats["hits"] + cache_stats["misses"], 1)
            ) * 100,
            "timestamp": datetime.now().isoformat()
        }

    def export_performance_report(self, filepath: str = None) -> str:
        """Export detailed performance report.

        Args:
            filepath: Optional file path to save report

        Returns:
            Report content as string
        """
        if filepath is None:
            filepath = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = self.get_performance_metrics()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report exported to: {filepath}")
        return json.dumps(report, indent=2)

    async def intelligent_content_suggestions(self, base_topic: str,
                                            count: int = 5) -> List[Dict[str, Any]]:
        """Generate intelligent content suggestions based on web research.

        Args:
            base_topic: Base topic for suggestions
            count: Number of suggestions to generate

        Returns:
            List of content suggestions with supporting data
        """
        logger.info(f"Generating content suggestions for: '{base_topic}'")

        # Research related topics and trends
        research_queries = [
            f"{base_topic} trends 2024",
            f"{base_topic} best practices",
            f"{base_topic} common questions",
            f"{base_topic} latest developments"
        ]

        # Parallel web research
        research_tasks = [
            self._async_web_search(query) for query in research_queries
        ]
        research_results = await asyncio.gather(*research_tasks, return_exceptions=True)

        # Generate suggestions based on research
        suggestions = []
        suggestion_topics = [
            f"Complete Guide to {base_topic}",
            f"{base_topic}: Common Mistakes to Avoid",
            f"Latest Trends in {base_topic}",
            f"{base_topic} for Beginners",
            f"Advanced {base_topic} Techniques"
        ]

        for i, topic in enumerate(suggestion_topics[:count]):
            # Get supporting images for each suggestion
            image_task = self.optimized_image_search(topic, count_per_source=2)
            images = await image_task

            suggestion = {
                "title": topic,
                "suggested_duration": "3-5 minutes",
                "difficulty": ["Beginner", "Intermediate", "Advanced"][i % 3],
                "supporting_images": images.get("aggregated_results", [])[:3],
                "research_data": research_results[i % len(research_results)] if research_results else None,
                "estimated_engagement": ["High", "Medium", "High", "Medium", "High"][i % 5]
            }

            suggestions.append(suggestion)

        logger.info(f"Generated {len(suggestions)} content suggestions")
        return suggestions

    def cleanup_resources(self):
        """Clean up resources and stop background tasks."""
        try:
            if self.task_queue:
                self.task_queue.stop()
            logger.info("Workflow optimizer resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")

# Factory function for easy integration
def create_workflow_optimizer(content_creator) -> WorkflowOptimizer:
    """Create and configure a workflow optimizer.

    Args:
        content_creator: ContentCreator instance

    Returns:
        Configured WorkflowOptimizer instance
    """
    optimizer = WorkflowOptimizer(content_creator)

    # Pre-warm cache with common topics
    common_topics = [
        "artificial intelligence",
        "technology trends",
        "business strategy",
        "digital marketing",
        "content creation"
    ]

    optimizer.smart_cache_warmup(common_topics)

    return optimizer
