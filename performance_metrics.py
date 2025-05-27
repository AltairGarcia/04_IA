"""Performance metrics tracking module"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Track and analyze system performance metrics"""

    operation_timings: Dict[str, List[float]] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    resource_usage: Dict[str, List[float]] = field(default_factory=dict)
    cache_stats: Dict[str, int] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)

    def record_operation_timing(self, component: str, operation: str, duration_ms: float,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the duration of an operation"""
        key = f"{component}:{operation}"
        if key not in self.operation_timings:
            self.operation_timings[key] = []
        self.operation_timings[key].append(duration_ms)
        self.last_update = datetime.now()

    def record_error(self, component: str, error_type: str) -> None:
        """Record an error occurrence"""
        key = f"{component}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.last_update = datetime.now()

    def record_resource_usage(self, resource: str, usage: float) -> None:
        """Record resource usage measurement"""
        if resource not in self.resource_usage:
            self.resource_usage[resource] = []
        self.resource_usage[resource].append(usage)
        self.last_update = datetime.now()

    def record_cache_stats(self, hits: int, misses: int, size: int) -> None:
        """Record cache performance statistics"""
        self.cache_stats["hits"] = self.cache_stats.get("hits", 0) + hits
        self.cache_stats["misses"] = self.cache_stats.get("misses", 0) + misses
        self.cache_stats["size"] = size
        self.last_update = datetime.now()

    def get_summary(self) -> dict:
        """Get a summary of all performance metrics"""
        summary = {
            "operation_timings": {},
            "error_counts": self.error_counts,
            "resource_usage": {},
            "cache_stats": self.cache_stats,
            "last_update": self.last_update.isoformat()
        }

        # Calculate operation timing stats
        for key, values in self.operation_timings.items():
            if values:
                summary["operation_timings"][key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "latest": values[-1]
                }

        # Calculate resource usage stats
        for key, values in self.resource_usage.items():
            if values:
                summary["resource_usage"][key] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "current": values[-1],
                    "samples": len(values)
                }

        return summary

# Global metrics instance
_metrics = PerformanceMetrics()

def get_metrics() -> PerformanceMetrics:
    """Get the global metrics tracker instance"""
    return _metrics
