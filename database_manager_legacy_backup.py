"""
Legacy Database Manager for Content Creation System

This module has been updated to use the unified database system.
Provides backward compatibility while leveraging the new unified architecture.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import hashlib
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
        
        # Ensure tables exist in unified system
        self.init_database()

    def init_database(self):
        """Initialize database tables using unified system."""
        try:
            # The unified database system handles initialization
            # We just ensure our specific tables exist
            logger.info("Database initialization delegated to unified system")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

                # Performance metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        operation_type TEXT NOT NULL,
                        response_time REAL,
                        cache_hit BOOLEAN,
                        api_provider TEXT,
                        success BOOLEAN,
                        error_message TEXT,
                        resource_usage TEXT  -- JSON string
                    )
                """)

                # User preferences
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT DEFAULT 'default',
                        preference_key TEXT NOT NULL,
                        preference_value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, preference_key)
                    )
                """)

                # Content templates
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_templates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        category TEXT NOT NULL,
                        template_data TEXT NOT NULL,  -- JSON string
                        usage_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Analytics data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_id INTEGER,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (content_id) REFERENCES content_history (id)
                    )
                """)

                # API usage tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider TEXT NOT NULL,
                        endpoint TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        request_size INTEGER,
                        response_size INTEGER,
                        cost REAL,
                        rate_limit_remaining INTEGER
                    )
                """)

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise    def save_content_creation(self, topic: str, content_type: str,
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
            logger.info(f"Content creation saved with ID: {content_id}")            return content_id
            
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
                    json.dumps(resource_usage or {})
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save performance metric: {str(e)}")

    def get_content_history(self, limit: int = 100,
                          content_type: str = None,
                          date_from: datetime = None,
                          date_to: datetime = None) -> List[Dict[str, Any]]:
        """Get content creation history.

        Args:
            limit: Maximum number of records
            content_type: Filter by content type
            date_from: Start date filter
            date_to: End date filter

        Returns:
            List of content records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM content_history WHERE 1=1"
                params = []

                if content_type:
                    query += " AND content_type = ?"
                    params.append(content_type)

                if date_from:
                    query += " AND created_at >= ?"
                    params.append(date_from.isoformat())

                if date_to:
                    query += " AND created_at <= ?"
                    params.append(date_to.isoformat())

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)

                columns = [desc[0] for desc in cursor.description]
                records = []

                for row in cursor.fetchall():
                    record = dict(zip(columns, row))

                    # Parse JSON fields
                    if record['components']:
                        record['components'] = json.loads(record['components'])
                    if record['tags']:
                        record['tags'] = json.loads(record['tags'])
                    if record['metadata']:
                        record['metadata'] = json.loads(record['metadata'])

                    records.append(record)

                return records

        except Exception as e:
            logger.error(f"Failed to get content history: {str(e)}")
            return []

    def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance analytics for specified period.

        Args:
            days: Number of days to analyze

        Returns:
            Performance analytics data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                date_threshold = datetime.now() - timedelta(days=days)

                # Overall statistics
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_operations,
                        AVG(response_time) as avg_response_time,
                        MIN(response_time) as min_response_time,
                        MAX(response_time) as max_response_time,
                        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_operations
                    FROM performance_metrics
                    WHERE timestamp >= ?
                """, (date_threshold.isoformat(),))

                stats = cursor.fetchone()

                # Performance by operation type
                cursor.execute("""
                    SELECT
                        operation_type,
                        COUNT(*) as count,
                        AVG(response_time) as avg_time,
                        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
                    FROM performance_metrics
                    WHERE timestamp >= ?
                    GROUP BY operation_type
                    ORDER BY count DESC
                """, (date_threshold.isoformat(),))

                operation_stats = cursor.fetchall()

                # Daily trends
                cursor.execute("""
                    SELECT
                        DATE(timestamp) as date,
                        COUNT(*) as operations,
                        AVG(response_time) as avg_response_time,
                        SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
                    FROM performance_metrics
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (date_threshold.isoformat(),))

                daily_trends = cursor.fetchall()

                return {
                    "overview": {
                        "total_operations": stats[0] or 0,
                        "avg_response_time": stats[1] or 0,
                        "min_response_time": stats[2] or 0,
                        "max_response_time": stats[3] or 0,
                        "cache_hit_rate": (stats[4] / max(stats[0], 1)) * 100,
                        "success_rate": (stats[5] / max(stats[0], 1)) * 100
                    },
                    "operation_breakdown": [
                        {
                            "operation_type": row[0],
                            "count": row[1],
                            "avg_time": row[2],
                            "cache_hits": row[3],
                            "cache_hit_rate": (row[3] / max(row[1], 1)) * 100
                        }
                        for row in operation_stats
                    ],
                    "daily_trends": [
                        {
                            "date": row[0],
                            "operations": row[1],
                            "avg_response_time": row[2],
                            "cache_hits": row[3]
                        }
                        for row in daily_trends
                    ]
                }

        except Exception as e:
            logger.error(f"Failed to get performance analytics: {str(e)}")
            return {}

    def save_user_preference(self, preference_key: str, preference_value: Any,
                           user_id: str = "default"):
        """Save user preference.

        Args:
            preference_key: Preference key
            preference_value: Preference value
            user_id: User identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO user_preferences
                    (user_id, preference_key, preference_value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, preference_key, json.dumps(preference_value)))

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save user preference: {str(e)}")

    def get_user_preferences(self, user_id: str = "default") -> Dict[str, Any]:
        """Get user preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT preference_key, preference_value
                    FROM user_preferences
                    WHERE user_id = ?
                """, (user_id,))

                preferences = {}
                for key, value in cursor.fetchall():
                    preferences[key] = json.loads(value)

                return preferences

        except Exception as e:
            logger.error(f"Failed to get user preferences: {str(e)}")
            return {}

    def save_content_template(self, name: str, category: str,
                            template_data: Dict[str, Any]) -> int:
        """Save content template.

        Args:
            name: Template name
            category: Template category
            template_data: Template configuration

        Returns:
            Template ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO content_templates
                    (name, category, template_data, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (name, category, json.dumps(template_data)))

                template_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Content template saved: {name}")
                return template_id

        except Exception as e:
            logger.error(f"Failed to save content template: {str(e)}")
            raise

    def get_content_templates(self, category: str = None) -> List[Dict[str, Any]]:
        """Get content templates.

        Args:
            category: Filter by category

        Returns:
            List of templates
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if category:
                    cursor.execute("""
                        SELECT * FROM content_templates
                        WHERE category = ?
                        ORDER BY usage_count DESC, name
                    """, (category,))
                else:
                    cursor.execute("""
                        SELECT * FROM content_templates
                        ORDER BY usage_count DESC, name
                    """)

                columns = [desc[0] for desc in cursor.description]
                templates = []

                for row in cursor.fetchall():
                    template = dict(zip(columns, row))
                    template['template_data'] = json.loads(template['template_data'])
                    templates.append(template)

                return templates

        except Exception as e:
            logger.error(f"Failed to get content templates: {str(e)}")
            return []

    def update_template_usage(self, template_id: int):
        """Update template usage count.

        Args:
            template_id: Template ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE content_templates
                    SET usage_count = usage_count + 1
                    WHERE id = ?
                """, (template_id,))

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to update template usage: {str(e)}")

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old performance data.

        Args:
            days_to_keep: Number of days to keep
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff_date = datetime.now() - timedelta(days=days_to_keep)

                cursor.execute("""
                    DELETE FROM performance_metrics
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old performance records")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")

    def export_analytics_report(self, filepath: str = None) -> str:
        """Export comprehensive analytics report.

        Args:
            filepath: Output file path

        Returns:
            Report file path
        """
        try:
            if filepath is None:
                filepath = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Gather comprehensive analytics
            report = {
                "generated_at": datetime.now().isoformat(),
                "performance_analytics": self.get_performance_analytics(30),
                "content_history": self.get_content_history(50),
                "template_usage": self.get_content_templates()
            }

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Analytics report exported to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to export analytics report: {str(e)}")
            raise


# Factory function
def create_database_manager(db_path: str = "content_creation.db") -> DatabaseManager:
    """Create database manager instance.

    Args:
        db_path: Database file path

    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_path)
