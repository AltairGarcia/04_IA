"""
Legacy Database Manager for Content Creation System

This module has been updated to use the unified database system.
Provides backward compatibility while leveraging the new unified architecture.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from core.database import UnifiedDatabaseManager, get_database_manager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Legacy database management class - now uses unified database system."""

    def __init__(self, db_path: str = "content_creation.db"):
        """Initialize database manager using unified system.

        Args:
            db_path: Path to SQLite database file (kept for compatibility)
        """
        self.db_path = db_path
        self.unified_db = get_database_manager()
        logger.info(f"DatabaseManager initialized using unified system (legacy path: {db_path})")

    def save_content_creation(self, topic: str, content_type: str,
                            components: Dict[str, Any], processing_time: float,
                            quality_score: Optional[int] = None,
                            tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> int:
        """Save content creation record using unified database system.

        Args:
            topic: Content topic
            content_type: Type of content created
            components: Content components generated
            processing_time: Time taken to create content
            quality_score: Quality score (0-100)
            tags: Content tags
            metadata: Additional metadata

        Returns:
            Content ID
        """
        try:
            # Use unified database system
            content_data = {
                'topic': topic,
                'content_type': content_type,
                'processing_time': processing_time,
                'quality_score': quality_score,
                'components': json.dumps(components) if components else None,
                'tags': json.dumps(tags) if tags else None,
                'metadata': json.dumps(metadata) if metadata else None,
                'created_at': datetime.now().isoformat()
            }
            
            content_id = self.unified_db.save_content(content_data)
            logger.info(f"Content creation saved with ID: {content_id}")
            return content_id
            
        except Exception as e:
            logger.error(f"Error saving content creation: {e}")
            raise

    def save_performance_metric(self, operation_type: str, response_time: float,
                              cache_hit: bool = False, api_provider: str = None,
                              success: bool = True, error_message: str = None,
                              resource_usage: Dict[str, Any] = None):
        """Save performance metric using unified database system.

        Args:
            operation_type: Type of operation performed
            response_time: Time taken for operation
            cache_hit: Whether cache was hit
            api_provider: API provider used
            success: Whether operation succeeded
            error_message: Error message if failed
            resource_usage: Resource usage information
        """
        try:
            # Use unified database system
            metric_data = {
                'operation_type': operation_type,
                'response_time': response_time,
                'cache_hit': cache_hit,
                'api_provider': api_provider,
                'success': success,
                'error_message': error_message,
                'resource_usage': json.dumps(resource_usage) if resource_usage else None,
                'timestamp': datetime.now().isoformat()
            }
            
            self.unified_db.save_metrics(metric_data)
            logger.info(f"Performance metric saved: {operation_type}")
            
        except Exception as e:
            logger.error(f"Error saving performance metric: {e}")
            raise

    def get_content_history(self, limit: int = 100, content_type: str = None) -> List[Dict[str, Any]]:
        """Get content creation history using unified database system.

        Args:
            limit: Maximum number of records to return
            content_type: Filter by content type

        Returns:
            List of content history records
        """
        try:
            filters = {}
            if content_type:
                filters['content_type'] = content_type
            
            content_list = self.unified_db.get_content_list(filters=filters, limit=limit)
            
            # Convert to legacy format
            history = []
            for content in content_list:
                history_item = {
                    'id': content.get('id'),
                    'topic': content.get('topic'),
                    'content_type': content.get('content_type'),
                    'created_at': content.get('created_at'),
                    'processing_time': content.get('processing_time'),
                    'quality_score': content.get('quality_score'),
                    'components': json.loads(content.get('components', '{}')) if content.get('components') else {},
                    'tags': json.loads(content.get('tags', '[]')) if content.get('tags') else [],
                    'metadata': json.loads(content.get('metadata', '{}')) if content.get('metadata') else {}
                }
                history.append(history_item)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting content history: {e}")
            return []

    def get_user_preference(self, preference_key: str, user_id: str = "default") -> Optional[str]:
        """Get user preference using unified database system.

        Args:
            preference_key: Preference key to retrieve
            user_id: User ID (default: "default")

        Returns:
            Preference value or None if not found
        """
        try:
            return self.unified_db.get_user_preference(user_id, preference_key)
        except Exception as e:
            logger.error(f"Error getting user preference: {e}")
            return None

    def set_user_preference(self, preference_key: str, preference_value: str, user_id: str = "default"):
        """Set user preference using unified database system.

        Args:
            preference_key: Preference key to set
            preference_value: Preference value
            user_id: User ID (default: "default")
        """
        try:
            self.unified_db.set_user_preference(user_id, preference_key, preference_value)
            logger.info(f"User preference set: {preference_key}")
        except Exception as e:
            logger.error(f"Error setting user preference: {e}")
            raise

    def get_analytics_data(self, metric_name: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """Get analytics data using unified database system.

        Args:
            metric_name: Filter by metric name
            days: Number of days to look back

        Returns:
            List of analytics records
        """
        try:
            filters = {}
            if metric_name:
                filters['metric_name'] = metric_name
            
            # Calculate date range
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            filters['start_date'] = start_date
            
            return self.unified_db.get_analytics_data(filters)
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return []

    def save_template(self, name: str, category: str, template_data: Dict[str, Any]) -> int:
        """Save content template using unified database system.

        Args:
            name: Template name
            category: Template category
            template_data: Template data

        Returns:
            Template ID
        """
        try:
            template_record = {
                'name': name,
                'category': category,
                'template_data': json.dumps(template_data),
                'usage_count': 0,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            template_id = self.unified_db.save_template(template_record)
            logger.info(f"Template saved with ID: {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Error saving template: {e}")
            raise

    def get_templates(self, category: str = None) -> List[Dict[str, Any]]:
        """Get content templates using unified database system.

        Args:
            category: Filter by category

        Returns:
            List of template records
        """
        try:
            filters = {}
            if category:
                filters['category'] = category
            
            templates = self.unified_db.get_templates(filters)
            
            # Convert template_data from JSON string to dict
            for template in templates:
                if 'template_data' in template and template['template_data']:
                    template['template_data'] = json.loads(template['template_data'])
            
            return templates
            
        except Exception as e:
            logger.error(f"Error getting templates: {e}")
            return []

    def update_template_usage(self, template_id: int):
        """Update template usage count.

        Args:
            template_id: Template ID to update
        """
        try:
            self.unified_db.update_template_usage(template_id)
            logger.info(f"Template usage updated for ID: {template_id}")
        except Exception as e:
            logger.error(f"Error updating template usage: {e}")

    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary using unified database system.

        Args:
            days: Number of days to analyze

        Returns:
            Performance summary dictionary
        """
        try:
            return self.unified_db.get_performance_summary(days)
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def cleanup_old_data(self, days: int = 90):
        """Clean up old data using unified database system.

        Args:
            days: Keep data newer than this many days
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            deleted_count = self.unified_db.cleanup_old_data(cutoff_date)
            logger.info(f"Cleaned up {deleted_count} old records older than {days} days")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0

    def export_data(self, output_format: str = "json") -> str:
        """Export database data using unified database system.

        Args:
            output_format: Export format ("json" or "csv")

        Returns:
            Export file path
        """
        try:
            return self.unified_db.export_data(output_format)
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics.

        Returns:
            Database information dictionary
        """
        try:
            return self.unified_db.get_database_info()
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}

# Backward compatibility functions
def get_database_manager(db_path: str = "content_creation.db") -> DatabaseManager:
    """Get database manager instance.
    
    Args:
        db_path: Database path (kept for compatibility)
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_path)

# Legacy aliases for backward compatibility
ContentDatabase = DatabaseManager
create_database_manager = get_database_manager
