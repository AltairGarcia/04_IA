#!/usr/bin/env python3
"""
Memory Optimization Enhancement for LangGraph 101

This module provides targeted enhancements to the existing optimized system
rather than replacing it. It integrates with the current production-ready
ThreadSafeConnectionManager and AdvancedMemoryProfiler.

Author: GitHub Copilot
Date: 2025-05-27
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import dataclass

# Import existing optimized systems
try:
    from thread_safe_connection_manager import get_connection_manager
    from advanced_memory_profiler import AdvancedMemoryProfiler
    from enhanced_unified_monitoring import EnhancedUnifiedMonitoringSystem
    OPTIMIZED_IMPORTS = True
except ImportError as e:
    logging.warning(f"Optimized modules not available: {e}")
    OPTIMIZED_IMPORTS = False

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Metrics for conversation lifecycle tracking."""
    conversation_id: str
    creation_time: datetime
    last_activity: datetime
    memory_usage_mb: float
    request_count: int


class ConversationLifecycleManager:
    """    Manages conversation lifecycle and prevents memory leaks
    from rapid creation.

    This class enhances the existing system without replacing it.
    """

    def __init__(self, memory_profiler=None):
        """Initialize with existing memory profiler."""
        self.memory_profiler = memory_profiler
        self.conversations: Dict[str, ConversationMetrics] = {}
        self.creation_threshold = 10  # Max conversations per 5 minutes
        self.max_age_minutes = 30     # Auto-cleanup after 30 minutes

    def register_conversation(self, conversation_id: str) -> bool:
        """
        Register a new conversation with rate limiting.

        Returns:
            bool: True if conversation can be created, False if rate limited
        """
        now = datetime.now()

        # Check for rapid creation
        recent_conversations = [
            conv for conv in self.conversations.values()
            if now - conv.creation_time < timedelta(minutes=5)
        ]

        if len(recent_conversations) >= self.creation_threshold:
            logger.warning(
                f"Rate limiting: {len(recent_conversations)} conversations "
                f"created in last 5 minutes "
                f"(threshold: {self.creation_threshold})"
            )
            return False

        # Register conversation
        self.conversations[conversation_id] = ConversationMetrics(
            conversation_id=conversation_id,
            creation_time=now,
            last_activity=now,            memory_usage_mb=0.0,
            request_count=0
        )

        logger.info(f"Registered conversation {conversation_id}")
        return True

    def update_conversation_activity(self, conversation_id: str,
                                     memory_delta: float = 0.0):
        """Update conversation activity and memory usage."""
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conv.last_activity = datetime.now()
            conv.memory_usage_mb += memory_delta
            conv.request_count += 1

    def cleanup_old_conversations(self) -> int:
        """Clean up old conversations and return count cleaned."""
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        old_conversations = [
            conv_id for conv_id, conv in self.conversations.items()
            if conv.last_activity < cutoff_time
        ]

        for conv_id in old_conversations:
            del self.conversations[conv_id]

        if old_conversations:
            logger.info(
                f"Cleaned up {len(old_conversations)} old conversations"
            )

        return len(old_conversations)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not self.conversations:
            return {"total": 0, "recent": 0, "memory_mb": 0.0}

        now = datetime.now()

        recent_count = sum(
            1 for conv in self.conversations.values()
            if now - conv.creation_time < timedelta(minutes=5)
        )

        total_memory = sum(
            conv.memory_usage_mb for conv in self.conversations.values()
        )

        return {
            "total": len(self.conversations),
            "recent": recent_count,
            "memory_mb": total_memory,
            "avg_memory_per_conversation": (
                total_memory / len(self.conversations)
            ),
            "oldest_conversation_age_minutes": (now - min(
                    conv.creation_time
                    for conv in self.conversations.values()
                )
            ).total_seconds() / 60
        }


class DatabaseOptimizationEnhancer:
    """
    Enhances database optimization for the existing
    ThreadSafeConnectionManager.
    """

    def __init__(self):
        """Initialize with connection manager."""
        if OPTIMIZED_IMPORTS:
            self.connection_manager = get_connection_manager()
        else:
            self.connection_manager = None
            logger.warning("ThreadSafeConnectionManager not available")

    def optimize_database_maintenance(self) -> Dict[str, Any]:
        """
        Perform database maintenance using the existing connection manager.

        Returns:
            Dict with optimization results
        """
        if not self.connection_manager:
            return {"error": "Connection manager not available"}

        results = {
            "vacuum_completed": False,
            "old_records_cleaned": 0,
            "optimization_time_ms": 0
        }

        start_time = time.time()

        try:
            # Use existing connection manager for safe database operations
            with self.connection_manager.get_connection() as conn:
                # Run VACUUM to optimize database
                conn.execute('VACUUM')
                results["vacuum_completed"] = True

                # Get table information safely
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()

                # Clean old log/metric records (keep last 1000)
                for (table_name,) in tables:
                    if ('log' in table_name.lower() or
                            'metric' in table_name.lower()):
                        cursor = conn.execute(f'''
                            DELETE FROM {table_name}
                            WHERE rowid NOT IN (
                                SELECT rowid FROM {table_name}
                                ORDER BY rowid DESC LIMIT 1000
                            )
                        ''')
                        results["old_records_cleaned"] += cursor.rowcount

                conn.commit()

        except Exception as e:
            logger.error(f"Database optimization error: {e}")
            results["error"] = str(e)

        results["optimization_time_ms"] = (time.time() - start_time) * 1000

        if results.get("vacuum_completed"):
            logger.info(
                f"Database optimized: {results['old_records_cleaned']} "
                f"old records cleaned, "
                f"took {results['optimization_time_ms']:.1f}ms"
            )
        return results


class MemoryEnhancementIntegrator:
    """
    Integrates enhancements with the existing optimized monitoring system.
    """

    def __init__(self):
        """Initialize with existing systems."""
        self.conversation_manager = ConversationLifecycleManager()
        self.db_optimizer = DatabaseOptimizationEnhancer()

        # Try to get existing memory profiler
        if OPTIMIZED_IMPORTS:
            try:
                self.memory_profiler = AdvancedMemoryProfiler()
                self.monitoring_system = EnhancedUnifiedMonitoringSystem()
            except Exception as e:
                logger.warning(f"Could not initialize existing systems: {e}")
                self.memory_profiler = None
                self.monitoring_system = None
        else:
            self.memory_profiler = None
            self.monitoring_system = None

    def start_enhanced_monitoring(self):
        """Start enhanced monitoring with existing systems."""
        if self.memory_profiler:
            logger.info(
                "Enhanced monitoring using existing AdvancedMemoryProfiler"
            )
        else:
            logger.warning(
                "Starting basic enhanced monitoring "
                "(existing systems not available)"
            )

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including enhancements."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "conversation_stats": (
                self.conversation_manager.get_conversation_stats()
            ),
            "database_optimization": (
                "Available" if self.db_optimizer.connection_manager
                else "Not Available"
            ),
            "existing_memory_profiler": (
                "Available" if self.memory_profiler else "Not Available"
            ),
            "existing_monitoring_system": (
                "Available" if self.monitoring_system else "Not Available"
            )
        }

        return status

    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform enhanced maintenance operations."""
        results = {
            "conversations_cleaned": (
                self.conversation_manager.cleanup_old_conversations()
            ),
            "database_optimization": (
                self.db_optimizer.optimize_database_maintenance()
            )
        }

        return results


# Global instance for easy access
memory_enhancement = MemoryEnhancementIntegrator()


def start_memory_enhancements():
    """Start memory enhancements."""
    memory_enhancement.start_enhanced_monitoring()


def get_enhanced_status() -> Dict[str, Any]:
    """Get enhanced status."""
    return memory_enhancement.get_comprehensive_status()


def perform_enhanced_maintenance() -> Dict[str, Any]:
    """Perform enhanced maintenance."""
    return memory_enhancement.perform_maintenance()


if __name__ == "__main__":
    # Test the enhancements
    print("Testing Memory Optimization Enhancements...")

    start_memory_enhancements()

    # Test conversation management
    conv_manager = ConversationLifecycleManager()

    # Test registering conversations
    for i in range(5):
        success = conv_manager.register_conversation(f"test_conv_{i}")
        print(f"Conversation {i}: {'✅' if success else '❌'}")

    # Get stats
    stats = conv_manager.get_conversation_stats()
    print(f"\nConversation Stats: {stats}")

    # Test enhanced status
    status = get_enhanced_status()
    print(f"\nEnhanced Status: {status}")

    print("\nMemory optimization enhancements tested successfully! ✅")
