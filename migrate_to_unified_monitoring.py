#!/usr/bin/env python3
"""
Memory and Performance Optimization Migration Script

This script implements the comprehensive fixes for the LangGraph 101 project:
1. Replaces multiple monitoring instances with unified system
2. Optimizes memory usage and prevents leaks
3. Implements proper cleanup mechanisms
4. Provides performance improvement tracking

Usage:
    python migrate_to_unified_monitoring.py [options]

Options:
    --backup       Create backup of existing monitoring files
    --force        Force migration even if issues are detected
    --dry-run      Show what would be done without making changes
    --verbose      Enable verbose logging

Author: GitHub Copilot
Date: 2025-05-27
"""

import os
import sys
import shutil
import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages the migration from multiple monitoring systems to unified monitoring."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False, force: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.force = force
        self.backup_dir = Path("monitoring_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.migration_log = []
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def log_action(self, action: str, details: str = "", success: bool = True):
        """Log migration action."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "success": success
        }
        self.migration_log.append(entry)
        
        if success:
            logger.info(f"✓ {action}: {details}")
        else:
            logger.error(f"✗ {action}: {details}")
    
    def create_backup(self) -> bool:
        """Create backup of existing monitoring files."""
        try:
            if self.dry_run:
                self.log_action("BACKUP", f"Would create backup in {self.backup_dir}")
                return True
            
            self.backup_dir.mkdir(exist_ok=True)
            
            # Files to backup
            monitoring_files = [
                "monitoring_dashboard.py",
                "performance_assessment.py", 
                "performance_test_simple.py",
                "performance_optimization.py",
                "memory_manager.py",
                "enhanced_cache_manager.py",
                "memory_optimization_fix.py"
            ]
            
            backed_up = 0
            for file_name in monitoring_files:
                source_path = Path(file_name)
                if source_path.exists():
                    backup_path = self.backup_dir / file_name
                    shutil.copy2(source_path, backup_path)
                    backed_up += 1
                    logger.debug(f"Backed up {file_name}")
            
            self.log_action("BACKUP", f"Backed up {backed_up} files to {self.backup_dir}")
            return True
            
        except Exception as e:
            self.log_action("BACKUP", f"Failed: {e}", False)
            return False
    
    def analyze_current_system(self) -> Dict[str, Any]:
        """Analyze current monitoring system state."""
        analysis = {
            "monitoring_files": [],
            "database_files": [],
            "memory_usage": None,
            "active_threads": None,
            "issues": []
        }
        
        try:
            # Check for monitoring files
            monitoring_patterns = [
                "*monitoring*.py",
                "*performance*.py", 
                "*memory*.py"
            ]
            
            for pattern in monitoring_patterns:
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        analysis["monitoring_files"].append(str(file_path))
            
            # Check for database files
            for db_file in Path(".").glob("*.db"):
                analysis["database_files"].append(str(db_file))
            
            # Check memory usage
            try:
                import psutil
                memory = psutil.virtual_memory()
                analysis["memory_usage"] = memory.percent
                analysis["active_threads"] = len(psutil.Process().threads())
                
                if memory.percent > 85:
                    analysis["issues"].append(f"High memory usage: {memory.percent:.1f}%")
                
                if analysis["active_threads"] > 20:
                    analysis["issues"].append(f"High thread count: {analysis['active_threads']}")
                    
            except ImportError:
                analysis["issues"].append("psutil not available for system analysis")
            
            # Check for duplicate monitoring classes
            monitoring_classes = []
            for file_path in analysis["monitoring_files"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "class PerformanceMonitor" in content:
                            monitoring_classes.append(file_path)
                except Exception:
                    pass
            
            if len(monitoring_classes) > 1:
                analysis["issues"].append(f"Multiple PerformanceMonitor classes found: {monitoring_classes}")
            
            self.log_action("ANALYSIS", f"Found {len(analysis['monitoring_files'])} monitoring files, {len(analysis['issues'])} issues")
            
        except Exception as e:
            self.log_action("ANALYSIS", f"Failed: {e}", False)
            analysis["issues"].append(f"Analysis error: {e}")
        
        return analysis
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        required_modules = [
            "psutil",
            "sqlite3", 
            "threading",
            "weakref",
            "collections"
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            self.log_action("DEPENDENCIES", f"Missing modules: {missing}", False)
            return False
        
        self.log_action("DEPENDENCIES", "All required modules available")
        return True
    
    def deploy_unified_system(self) -> bool:
        """Deploy the unified monitoring system."""
        try:
            if self.dry_run:
                self.log_action("DEPLOY", "Would deploy unified monitoring system")
                return True
            
            # Check if unified monitoring files exist
            unified_files = [
                "unified_monitoring_system.py",
                "monitoring_integration_patch.py"
            ]
            
            missing_files = [f for f in unified_files if not Path(f).exists()]
            if missing_files:
                self.log_action("DEPLOY", f"Missing unified system files: {missing_files}", False)
                return False
            
            # Test import of unified system
            try:
                sys.path.insert(0, str(Path.cwd()))
                import unified_monitoring_system
                import monitoring_integration_patch
                
                # Test initialization
                monitor = unified_monitoring_system.get_unified_monitor()
                self.log_action("DEPLOY", "Unified monitoring system deployed successfully")
                return True
                
            except Exception as e:
                self.log_action("DEPLOY", f"Failed to import unified system: {e}", False)
                return False
            
        except Exception as e:
            self.log_action("DEPLOY", f"Deployment failed: {e}", False)
            return False
    
    def apply_integration_patches(self) -> bool:
        """Apply integration patches to existing code."""
        try:
            if self.dry_run:
                self.log_action("PATCHES", "Would apply integration patches")
                return True
            
            # Import and apply patches
            import monitoring_integration_patch
            
            # Apply all patches
            results = monitoring_integration_patch.apply_all_patches()
            
            successful_patches = sum(results.values())
            total_patches = len(results)
            
            if successful_patches > 0:
                self.log_action("PATCHES", f"Applied {successful_patches}/{total_patches} patches successfully")
                return True
            else:
                self.log_action("PATCHES", "No patches were applied", False)
                return False
            
        except Exception as e:
            self.log_action("PATCHES", f"Failed to apply patches: {e}", False)
            return False
    
    def test_unified_system(self) -> bool:
        """Test the unified monitoring system."""
        try:
            if self.dry_run:
                self.log_action("TEST", "Would test unified monitoring system")
                return True
            
            # Import unified system
            import unified_monitoring_system
            
            # Get monitor instance
            monitor = unified_monitoring_system.get_unified_monitor()
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Wait a bit for it to collect metrics
            time.sleep(5)
            
            # Get status
            status = monitor.get_current_status()
            
            # Check if system is working
            if "error" in status:
                self.log_action("TEST", f"System error: {status['error']}", False)
                return False
            
            memory_percent = status.get("memory_percent", 0)
            self.log_action("TEST", f"System working - Memory: {memory_percent:.1f}%")
            
            # Stop monitoring for now
            monitor.stop_monitoring()
            
            return True
            
        except Exception as e:
            self.log_action("TEST", f"Testing failed: {e}", False)
            return False
    
    def optimize_databases(self) -> bool:
        """Optimize existing database files."""
        try:
            if self.dry_run:
                self.log_action("DB_OPTIMIZE", "Would optimize database files")
                return True
            
            import sqlite3
            
            db_files = list(Path(".").glob("*.db"))
            optimized = 0
            
            for db_file in db_files:
                try:
                    with sqlite3.connect(str(db_file)) as conn:
                        # Optimize database
                        conn.execute("VACUUM")
                        conn.execute("ANALYZE")
                        
                        # Clean old records if it's a monitoring database
                        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                        
                        for (table_name,) in tables:
                            if any(keyword in table_name.lower() for keyword in ['metric', 'alert', 'log', 'performance']):
                                # Keep only last 1000 records
                                conn.execute(f"""
                                    DELETE FROM {table_name} 
                                    WHERE rowid NOT IN (
                                        SELECT rowid FROM {table_name} 
                                        ORDER BY rowid DESC LIMIT 1000
                                    )
                                """)
                        
                        conn.commit()
                        optimized += 1
                        
                except sqlite3.Error as e:
                    logger.warning(f"Could not optimize {db_file}: {e}")
            
            self.log_action("DB_OPTIMIZE", f"Optimized {optimized} database files")
            return True
            
        except Exception as e:
            self.log_action("DB_OPTIMIZE", f"Failed: {e}", False)
            return False
    
    def create_startup_script(self) -> bool:
        """Create a startup script for the new monitoring system."""
        try:
            if self.dry_run:
                self.log_action("STARTUP_SCRIPT", "Would create startup script")
                return True
            
            startup_script = """#!/usr/bin/env python3
'''
Startup script for unified monitoring system.
This script should be imported at the beginning of your application.
'''

import logging
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import unified monitoring system
    from unified_monitoring_system import start_unified_monitoring, stop_unified_monitoring
    from monitoring_integration_patch import apply_all_patches
    
    # Apply integration patches
    logger.info("Applying monitoring integration patches...")
    patch_results = apply_all_patches()
    
    # Start unified monitoring
    logger.info("Starting unified monitoring system...")
    start_unified_monitoring()
    
    # Register cleanup function
    atexit.register(stop_unified_monitoring)
    
    logger.info("Unified monitoring system started successfully")
    
except Exception as e:
    logger.error(f"Failed to start unified monitoring: {e}")
    # Don't fail the application startup if monitoring fails
"""
            
            with open("start_unified_monitoring.py", "w", encoding='utf-8') as f:
                f.write(startup_script)
            
            self.log_action("STARTUP_SCRIPT", "Created start_unified_monitoring.py")
            return True
            
        except Exception as e:
            self.log_action("STARTUP_SCRIPT", f"Failed: {e}", False)
            return False
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration report."""
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "actions_performed": len(self.migration_log),
            "successful_actions": len([a for a in self.migration_log if a["success"]]),
            "failed_actions": len([a for a in self.migration_log if not a["success"]]),
            "backup_location": str(self.backup_dir) if self.backup_dir.exists() else None,
            "actions": self.migration_log
        }
        
        # Add performance analysis
        try:
            import psutil
            memory = psutil.virtual_memory()
            report["post_migration_memory"] = memory.percent
            report["post_migration_threads"] = len(psutil.Process().threads())
        except ImportError:
            pass
        
        return report
    
    def save_migration_report(self, report: Dict[str, Any]):
        """Save migration report to file."""
        report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Migration report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save migration report: {e}")
    
    def run_migration(self, create_backup: bool = True) -> bool:
        """Run the complete migration process."""
        logger.info("Starting migration to unified monitoring system...")
        
        try:
            # 1. Create backup
            if create_backup and not self.create_backup():
                if not self.force:
                    logger.error("Backup failed and --force not specified. Aborting.")
                    return False
            
            # 2. Analyze current system
            analysis = self.analyze_current_system()
            if analysis["issues"] and not self.force:
                logger.warning("Issues detected in current system:")
                for issue in analysis["issues"]:
                    logger.warning(f"  - {issue}")
                logger.warning("Use --force to proceed anyway")
                return False
            
            # 3. Check dependencies
            if not self.check_dependencies():
                logger.error("Dependency check failed")
                return False
            
            # 4. Deploy unified system
            if not self.deploy_unified_system():
                logger.error("Failed to deploy unified system")
                return False
            
            # 5. Apply integration patches
            if not self.apply_integration_patches():
                logger.error("Failed to apply integration patches")
                return False
            
            # 6. Test unified system
            if not self.test_unified_system():
                logger.error("Unified system test failed")
                return False
            
            # 7. Optimize databases
            self.optimize_databases()
            
            # 8. Create startup script
            self.create_startup_script()
            
            # 9. Generate and save report
            report = self.generate_migration_report()
            self.save_migration_report(report)
            
            logger.info("Migration completed successfully!")
            return True
            
        except Exception as e:
            self.log_action("MIGRATION", f"Migration failed: {e}", False)
            logger.error(f"Migration failed: {e}")
            return False


def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate to unified monitoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--backup", action="store_true", 
                       help="Create backup of existing monitoring files")
    parser.add_argument("--force", action="store_true",
                       help="Force migration even if issues are detected")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup creation")
    
    args = parser.parse_args()
    
    # Create migration manager
    migration_manager = MigrationManager(
        dry_run=args.dry_run,
        verbose=args.verbose,
        force=args.force
    )
    
    # Run migration
    success = migration_manager.run_migration(
        create_backup=args.backup and not args.no_backup
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Import 'start_unified_monitoring.py' at the beginning of your application")
        print("2. Remove old monitoring system imports")
        print("3. Monitor system performance and adjust thresholds as needed")
        print("4. Review the migration report for detailed information")
        
        if args.dry_run:
            print("\nNote: This was a dry run. Run without --dry-run to apply changes.")
    else:
        print("\n" + "="*60)
        print("❌ MIGRATION FAILED")
        print("="*60)
        print("\nCheck the migration log for details.")
        print("Consider using --force if you want to proceed despite issues.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
