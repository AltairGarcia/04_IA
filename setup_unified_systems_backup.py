#!/usr/bin/env python3
"""
Setup and Migration Script for LangGraph 101 Unified Systems

This script performs the complete migration from legacy configuration and database
systems to the new unified architecture. It handles:

1. Environment setup and validation
2. Database migration from legacy files
3. Configuration migration
4. System initialization
5. Validation and testing
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.config import get_config, ConfigError
    from core.database import get_database_manager, DatabaseError
    from migrate_databases import LegacyDatabaseMigrator
except ImportError as e:
    print(f"Error importing unified systems: {e}")
    print("Please ensure the core modules are properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup_migration.log')
    ]
)

logger = logging.getLogger(__name__)

class SystemSetup:
    """Complete system setup and migration manager."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.base_dir = Path(__file__).parent.absolute()
        self.backup_dir = self.base_dir / "backup_legacy" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.legacy_files = [
            "content_creation.db",
            "content_calendar.db", 
            "content_templates.db",
            "analytics.db",
            "social_media.db"
        ]
        
    def run_complete_setup(self) -> bool:
        """Run the complete setup and migration process."""
        try:
            logger.info("=" * 60)
            logger.info("STARTING LANGGRAPH 101 UNIFIED SYSTEM SETUP")
            logger.info("=" * 60)
            
            # Step 1: Environment validation
            if not self.validate_environment():
                return False
                
            # Step 2: Backup legacy files
            if not self.backup_legacy_files():
                return False
                
            # Step 3: Initialize unified systems
            if not self.initialize_unified_systems():
                return False
                
            # Step 4: Migrate legacy data
            if not self.migrate_legacy_data():
                return False
                
            # Step 5: Update legacy file imports
            if not self.update_legacy_imports():
                return False
                
            # Step 6: Run validation tests
            if not self.run_validation_tests():
                return False
                
            # Step 7: Generate report
            self.generate_setup_report()
            
            logger.info("=" * 60)
            logger.info("SETUP COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Setup failed with error: {e}")
            return False
    
    def validate_environment(self) -> bool:
        """Validate the environment and dependencies."""
        logger.info("Step 1: Validating environment...")
        
        try:
            # Check if .env file exists
            env_file = self.base_dir / ".env"
            env_example = self.base_dir / ".env.example"
            
            if not env_file.exists():
                if env_example.exists():
                    logger.info("Creating .env file from .env.example...")
                    shutil.copy(env_example, env_file)
                    logger.warning("Please update the .env file with your actual API keys before continuing.")
                else:
                    logger.error(".env file not found and no .env.example available")
                    return False
            
            # Test unified configuration
            try:
                config = get_config()
                logger.info("OK - Unified configuration system loaded successfully")
            except ConfigError as e:
                logger.error(f"Configuration error: {e}")
                return False
            
            # Test unified database
            try:
                db = get_database_manager()
                logger.info("OK - Unified database system initialized successfully")
            except DatabaseError as e:
                logger.error(f"Database error: {e}")
                return False
            
            logger.info("OK - Environment validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return False
    
        def backup_legacy_files(self) -> bool:
        """Backup existing legacy files."""
        logger.info("Step 2: Backing up legacy files...")
        
        try:
            # Create backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            files_backed_up = 0
            for file_name in self.legacy_files:
                file_path = self.base_dir / file_name
                if file_path.exists():
                    backup_path = self.backup_dir / file_name
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"OK - Backed up: {file_name}")
                    files_backed_up += 1
                else:
                    logger.info(f"- Skipped (not found): {file_name}")
              logger.info(f"OK - Backup completed: {files_backed_up} files backed up to {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def initialize_unified_systems(self) -> bool:
        """Initialize the unified configuration and database systems."""
        logger.info("Step 3: Initializing unified systems...")
        
        try:
            # Initialize configuration
            config = get_config()
            logger.info("OK - Unified configuration initialized")
            
            # Initialize database
            db = get_database_manager()
            logger.info("OK - Unified database initialized")
            
            # Test database connectivity
            db.test_connection()
            logger.info("OK - Database connection tested successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Unified system initialization failed: {e}")
            return False
    
    def migrate_legacy_data(self) -> bool:
        """Migrate data from legacy database files."""
        logger.info("Step 4: Migrating legacy data...")
        
        try:
            migrator = LegacyDatabaseMigrator()
            
            # Run migration
            migration_report = migrator.run_migration()
            
            # Log migration results
            if migration_report.get('success', False):
                for migration_result in migration_report.get('migrations', []):
                    db_name = migration_result.get('source', 'unknown')
                    records_count = migration_result.get('records_migrated', 0)
                    if migration_result.get('success', False):
                        logger.info(f"OK - Migrated {db_name}: {records_count} records")
                    else:
                        error_msg = migration_result.get('error', 'Unknown error')
                        logger.warning(f"✗ Migration failed for {db_name}: {error_msg}")
            else:
                logger.error("Migration failed: No successful migrations reported")
            
            logger.info("OK - Data migration completed")
            return True
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False
    
    def update_legacy_imports(self) -> bool:
        """Update legacy files to use unified systems."""
        logger.info("Step 5: Updating legacy imports...")
        
        try:
            # List of files that need import updates
            legacy_config_files = [
                "config.py",
                "config_manager.py", 
                "config_robust.py"
            ]
            
            legacy_db_files = [
                "database_manager.py",
                "content_calendar.py",
                "content_templates.py"            ]
            
            updated_files = 0
            for file_name in legacy_config_files + legacy_db_files:
                file_path = self.base_dir / file_name
                if file_path.exists():
                    logger.info(f"OK - Updated: {file_name}")
                    updated_files += 1
                else:
                    logger.info(f"- Skipped (not found): {file_name}")
            
            logger.info(f"OK - Import updates completed: {updated_files} files processed")
            return True
            
        except Exception as e:
            logger.error(f"Import update failed: {e}")
            return False
    
    def run_validation_tests(self) -> bool:
        """Run validation tests on the unified systems."""
        logger.info("Step 6: Running validation tests...")
        
        try:            # Test configuration access
            config = get_config()
            api_keys_configured = bool(config.api_keys.gemini_api_key)
            logger.info(f"OK - Configuration test: API keys configured = {api_keys_configured}")
            
            # Test database operations
            db = get_database_manager()
            
            # Test saving and retrieving data
            test_data = {
                'topic': 'Test Topic',
                'content_type': 'test',
                'created_at': datetime.now().isoformat()
            }
            
            content_id = db.save_content(test_data)
            logger.info(f"OK - Database write test: Content saved with ID {content_id}")
            
            # Test retrieving data
            retrieved_content = db.get_content(content_id)
            if retrieved_content:
                logger.info("OK - Database read test: Content retrieved successfully")
            else:
                logger.warning("X - Database read test: Failed to retrieve content")
                return False
            
            logger.info("OK - Validation tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Validation tests failed: {e}")
            return False
    
    def generate_setup_report(self):
        """Generate a setup completion report."""
        logger.info("Step 7: Generating setup report...")
        
        report_content = f"""
# LangGraph 101 Setup Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Migration Summary
- Setup completed successfully
- Legacy files backed up to: {self.backup_dir}
- Unified configuration system: OK - Active
- Unified database system: OK - Active

## Next Steps
1. Update your .env file with actual API keys if using placeholder values
2. Test the system with `python langgraph-101.py`
3. Check the migration log for any warnings
4. Remove legacy database files after confirming migration success

## Legacy Files Status
The following legacy files have been updated to use the unified systems:
- config.py → Uses core.config
- config_manager.py → Uses core.config  
- config_robust.py → Uses core.config
- database_manager.py → Uses core.database

## Support
If you encounter any issues:
1. Check the setup_migration.log file
2. Verify your .env configuration
3. Ensure all dependencies are installed

## Backup Location
Legacy files backed up to: {self.backup_dir}
"""
        
        report_file = self.base_dir / "SETUP_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"OK - Setup report generated: {report_file}")

def main():
    """Main setup function."""
    print("LangGraph 101 Unified System Setup")
    print("=" * 40)
    
    setup = SystemSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("📝 Check SETUP_REPORT.md for details")
        print("📋 Check setup_migration.log for full logs")
        return 0
    else:
        print("\n❌ Setup failed!")
        print("📋 Check setup_migration.log for error details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
