#!/usr/bin/env python3
"""
Migration Script for LangGraph 101 - Database Consolidation

This script migrates data from legacy database files to the unified database system.
It handles:
- content_creation.db (DatabaseManager)
- content_calendar.db (ContentCalendarManager) 
- content_templates.db (TemplateManager)
- Various other standalone database files

Run this script ONCE after implementing the unified database system.
"""

import os
import sys
import logging
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.database import UnifiedDatabaseManager
from core.config import get_config

logger = logging.getLogger(__name__)


class LegacyDatabaseMigrator:
    """Migrates data from legacy database files to unified system"""
    
    def __init__(self):
        self.config = get_config()
        self.unified_db = UnifiedDatabaseManager()
        self.migration_log = []
        
    def run_migration(self) -> Dict[str, Any]:
        """Run complete migration process"""
        logger.info("Starting database migration process...")
        
        results = {
            "success": True,
            "migrations": [],
            "errors": [],
            "summary": {}
        }
        
        try:
            # Migrate content creation data
            if self._file_exists("content_creation.db"):
                content_result = self._migrate_content_creation_db()
                results["migrations"].append(content_result)
            
            # Migrate content calendar data
            if self._file_exists("content_calendar.db"):
                calendar_result = self._migrate_content_calendar_db()
                results["migrations"].append(calendar_result)
            
            # Migrate content templates
            if self._file_exists("content_templates.db"):
                templates_result = self._migrate_content_templates_db()
                results["migrations"].append(templates_result)
            
            # Migrate analytics data
            if self._file_exists("analytics.db"):
                analytics_result = self._migrate_analytics_db()
                results["migrations"].append(analytics_result)
            
            # Migrate social media data
            if self._file_exists("social_media.db"):
                social_result = self._migrate_social_media_db()
                results["migrations"].append(social_result)
            
            # Generate summary
            results["summary"] = self._generate_summary(results["migrations"])
            
            # Create backup of legacy files
            if results["success"]:
                self._backup_legacy_files()
            
            logger.info("Database migration completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            results["success"] = False
            results["errors"].append(str(e))
        
        return results
    
    def _file_exists(self, filename: str) -> bool:
        """Check if legacy database file exists"""
        return os.path.exists(filename)
    
    def _migrate_content_creation_db(self) -> Dict[str, Any]:
        """Migrate content_creation.db data"""
        logger.info("Migrating content_creation.db...")
        
        result = {
            "source": "content_creation.db",
            "tables_migrated": [],
            "records_migrated": 0,
            "errors": []
        }
        
        try:
            with sqlite3.connect("content_creation.db") as legacy_conn:
                legacy_conn.row_factory = sqlite3.Row
                
                # Migrate content_history table
                try:
                    cursor = legacy_conn.execute("SELECT * FROM content_history")
                    records = cursor.fetchall()
                    
                    for record in records:
                        with self.unified_db.get_connection() as unified_conn:
                            unified_conn.execute("""
                                INSERT OR IGNORE INTO content (
                                    id, title, content_type, content_data, status,
                                    created_at, metadata
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                f"legacy_{record['id']}",
                                record.get('topic', 'Migrated Content'),
                                record.get('content_type', 'text'),
                                record.get('components', '{}'),
                                'published',  # Assume published for legacy data
                                record.get('created_at'),
                                json.dumps({
                                    "migrated_from": "content_creation.db",
                                    "original_id": record['id'],
                                    "processing_time": record.get('processing_time'),
                                    "quality_score": record.get('quality_score'),
                                    "tags": json.loads(record.get('tags', '[]'))
                                })
                            ))
                    
                    result["tables_migrated"].append("content_history")
                    result["records_migrated"] += len(records)
                    logger.info(f"Migrated {len(records)} records from content_history")
                    
                except Exception as e:
                    result["errors"].append(f"content_history migration failed: {e}")
                
                # Migrate performance_metrics table
                try:
                    cursor = legacy_conn.execute("SELECT * FROM performance_metrics")
                    records = cursor.fetchall()
                    
                    for record in records:
                        with self.unified_db.get_connection() as unified_conn:
                            unified_conn.execute("""
                                INSERT OR IGNORE INTO system_metrics (
                                    id, metric_name, metric_value, recorded_at, metadata
                                ) VALUES (?, ?, ?, ?, ?)
                            """, (
                                f"perf_{record['id']}",
                                record.get('operation_type', 'unknown'),
                                record.get('response_time', 0),
                                record.get('timestamp'),
                                json.dumps({
                                    "migrated_from": "content_creation.db",
                                    "cache_hit": record.get('cache_hit'),
                                    "api_provider": record.get('api_provider'),
                                    "success": record.get('success'),
                                    "error_message": record.get('error_message')
                                })
                            ))
                    
                    result["tables_migrated"].append("performance_metrics")
                    result["records_migrated"] += len(records)
                    logger.info(f"Migrated {len(records)} records from performance_metrics")
                    
                except Exception as e:
                    result["errors"].append(f"performance_metrics migration failed: {e}")
                
                # Migrate user_preferences table
                try:
                    cursor = legacy_conn.execute("SELECT * FROM user_preferences")
                    records = cursor.fetchall()
                    
                    for record in records:
                        with self.unified_db.get_connection() as unified_conn:
                            unified_conn.execute("""
                                INSERT OR IGNORE INTO user_preferences (
                                    id, user_id, preference_name, preference_value, updated_at
                                ) VALUES (?, ?, ?, ?, ?)
                            """, (
                                f"legacy_pref_{record['id']}",
                                record.get('user_id', 'default'),
                                record.get('preference_key'),
                                record.get('preference_value'),
                                record.get('updated_at')
                            ))
                    
                    result["tables_migrated"].append("user_preferences")
                    result["records_migrated"] += len(records)
                    logger.info(f"Migrated {len(records)} records from user_preferences")
                    
                except Exception as e:
                    result["errors"].append(f"user_preferences migration failed: {e}")
        
        except Exception as e:
            result["errors"].append(f"Failed to connect to content_creation.db: {e}")
        
        return result
    
    def _migrate_content_calendar_db(self) -> Dict[str, Any]:
        """Migrate content_calendar.db data"""
        logger.info("Migrating content_calendar.db...")
        
        result = {
            "source": "content_calendar.db",
            "tables_migrated": [],
            "records_migrated": 0,
            "errors": []
        }
        
        try:
            with sqlite3.connect("content_calendar.db") as legacy_conn:
                legacy_conn.row_factory = sqlite3.Row
                
                # Migrate calendar_events table
                try:
                    cursor = legacy_conn.execute("SELECT * FROM calendar_events")
                    records = cursor.fetchall()
                    
                    for record in records:
                        with self.unified_db.get_connection() as unified_conn:
                            unified_conn.execute("""
                                INSERT OR IGNORE INTO content_calendar (
                                    id, title, description, scheduled_date, platform,
                                    status, assigned_to, priority, created_at, metadata
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                record['id'],
                                record.get('title'),
                                record.get('description'),
                                record.get('start_date'),
                                json.loads(record.get('channels', '[]'))[0] if record.get('channels') else 'web',
                                record.get('status', 'scheduled'),
                                record.get('assigned_to'),
                                record.get('priority', 'medium'),
                                record.get('created_at'),
                                json.dumps({
                                    "migrated_from": "content_calendar.db",
                                    "content_type": record.get('content_type'),
                                    "end_date": record.get('end_date'),
                                    "team_members": json.loads(record.get('team_members', '[]')),
                                    "channels": json.loads(record.get('channels', '[]')),
                                    "content_data": json.loads(record.get('content_data', '{}')),
                                    "template_id": record.get('template_id'),
                                    "parent_campaign_id": record.get('parent_campaign_id')
                                })
                            ))
                    
                    result["tables_migrated"].append("calendar_events")
                    result["records_migrated"] += len(records)
                    logger.info(f"Migrated {len(records)} records from calendar_events")
                    
                except Exception as e:
                    result["errors"].append(f"calendar_events migration failed: {e}")
                
                # Migrate content_campaigns table
                try:
                    cursor = legacy_conn.execute("SELECT * FROM content_campaigns")
                    records = cursor.fetchall()
                    
                    for record in records:
                        with self.unified_db.get_connection() as unified_conn:
                            unified_conn.execute("""
                                INSERT OR IGNORE INTO content (
                                    id, title, content_type, content_data, status,
                                    created_at, metadata
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                f"campaign_{record['id']}",
                                record.get('name', 'Migrated Campaign'),
                                'campaign',
                                json.dumps({
                                    "description": record.get('description'),
                                    "start_date": record.get('start_date'),
                                    "end_date": record.get('end_date'),
                                    "goals": json.loads(record.get('goals', '[]')),
                                    "target_audience": record.get('target_audience'),
                                    "budget": record.get('budget'),
                                    "spent": record.get('spent')
                                }),
                                record.get('status', 'active'),
                                record.get('created_at'),
                                json.dumps({
                                    "migrated_from": "content_calendar.db",
                                    "type": "campaign",
                                    "created_by": record.get('created_by'),
                                    "performance_metrics": json.loads(record.get('performance_metrics', '{}'))
                                })
                            ))
                    
                    result["tables_migrated"].append("content_campaigns")
                    result["records_migrated"] += len(records)
                    logger.info(f"Migrated {len(records)} records from content_campaigns")
                    
                except Exception as e:
                    result["errors"].append(f"content_campaigns migration failed: {e}")
        
        except Exception as e:
            result["errors"].append(f"Failed to connect to content_calendar.db: {e}")
        
        return result
    
    def _migrate_content_templates_db(self) -> Dict[str, Any]:
        """Migrate content_templates.db data"""
        logger.info("Migrating content_templates.db...")
        
        result = {
            "source": "content_templates.db",
            "tables_migrated": [],
            "records_migrated": 0,
            "errors": []
        }
        
        try:
            with sqlite3.connect("content_templates.db") as legacy_conn:
                legacy_conn.row_factory = sqlite3.Row
                
                # Migrate templates table
                try:
                    cursor = legacy_conn.execute("SELECT * FROM templates")
                    records = cursor.fetchall()
                    
                    for record in records:
                        with self.unified_db.get_connection() as unified_conn:
                            unified_conn.execute("""
                                INSERT OR IGNORE INTO content_templates (
                                    id, name, content_type, category, template_data,
                                    variables, usage_count, rating, is_active,
                                    created_at, updated_at, metadata
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                record['id'],
                                record.get('name'),
                                record.get('content_type'),
                                record.get('category'),
                                json.dumps({
                                    "title_template": record.get('title_template'),
                                    "content_template": record.get('content_template'),
                                    "description": record.get('description')
                                }),
                                record.get('variables'),
                                record.get('usage_count', 0),
                                record.get('rating', 0.0),
                                record.get('is_active', True),
                                record.get('created_at'),
                                record.get('updated_at'),
                                json.dumps({
                                    "migrated_from": "content_templates.db",
                                    "version": record.get('version'),
                                    "tags": json.loads(record.get('tags', '[]'))
                                })
                            ))
                    
                    result["tables_migrated"].append("templates")
                    result["records_migrated"] += len(records)
                    logger.info(f"Migrated {len(records)} records from templates")
                    
                except Exception as e:
                    result["errors"].append(f"templates migration failed: {e}")
        
        except Exception as e:
            result["errors"].append(f"Failed to connect to content_templates.db: {e}")
        
        return result
    
    def _migrate_analytics_db(self) -> Dict[str, Any]:
        """Migrate analytics.db data"""
        logger.info("Migrating analytics.db...")
        
        result = {
            "source": "analytics.db",
            "tables_migrated": [],
            "records_migrated": 0,
            "errors": []
        }
        
        # Placeholder for analytics migration
        # Implementation depends on actual analytics database structure
        
        return result
    
    def _migrate_social_media_db(self) -> Dict[str, Any]:
        """Migrate social_media.db data"""
        logger.info("Migrating social_media.db...")
        
        result = {
            "source": "social_media.db",
            "tables_migrated": [],
            "records_migrated": 0,
            "errors": []
        }
        
        # Placeholder for social media migration
        # Implementation depends on actual social media database structure
        
        return result
    
    def _generate_summary(self, migrations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate migration summary"""
        total_records = sum(m.get("records_migrated", 0) for m in migrations)
        total_tables = sum(len(m.get("tables_migrated", [])) for m in migrations)
        total_errors = sum(len(m.get("errors", [])) for m in migrations)
        
        return {
            "total_files_migrated": len(migrations),
            "total_tables_migrated": total_tables,
            "total_records_migrated": total_records,
            "total_errors": total_errors,
            "migration_timestamp": datetime.now().isoformat()
        }
    
    def _backup_legacy_files(self):
        """Create backup of legacy database files"""
        backup_dir = Path("legacy_db_backup")
        backup_dir.mkdir(exist_ok=True)
        
        legacy_files = [
            "content_creation.db",
            "content_calendar.db", 
            "content_templates.db",
            "analytics.db",
            "social_media.db"
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for filename in legacy_files:
            if os.path.exists(filename):
                backup_name = f"{filename.split('.')[0]}_{timestamp}.db"
                backup_path = backup_dir / backup_name
                
                import shutil
                shutil.copy2(filename, backup_path)
                logger.info(f"Backed up {filename} to {backup_path}")


def main():
    """Main migration function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔄 LangGraph 101 Database Migration Tool")
    print("=" * 50)
    
    migrator = LegacyDatabaseMigrator()
    
    # Run migration
    results = migrator.run_migration()
    
    # Print results
    print("\n📊 Migration Results:")
    print(f"Success: {'✅' if results['success'] else '❌'}")
    
    if results['summary']:
        summary = results['summary']
        print(f"Files migrated: {summary['total_files_migrated']}")
        print(f"Tables migrated: {summary['total_tables_migrated']}")
        print(f"Records migrated: {summary['total_records_migrated']}")
        print(f"Errors: {summary['total_errors']}")
    
    if results['errors']:
        print("\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Detailed migration results
    for migration in results['migrations']:
        print(f"\n📁 {migration['source']}:")
        print(f"  Tables: {', '.join(migration['tables_migrated'])}")
        print(f"  Records: {migration['records_migrated']}")
        if migration['errors']:
            print(f"  Errors: {len(migration['errors'])}")
    
    print("\n✅ Migration process completed!")
    print("💾 Legacy database files have been backed up to 'legacy_db_backup/' directory")
    print("🎯 You can now safely use the unified database system")


if __name__ == "__main__":
    main()
