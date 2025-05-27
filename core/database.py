"""
Unified Database Management System for LangGraph 101

This module provides a centralized database management system that consolidates
all database operations, handles migrations, and ensures data consistency.
"""

import sqlite3
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, ContextManager
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception raised for database errors."""
    pass


@dataclass
class TableSchema:
    """Database table schema definition"""
    name: str
    columns: Dict[str, str]
    indexes: List[str] = None
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.indexes is None:
            self.indexes = []
        if self.constraints is None:
            self.constraints = []


class DatabaseMigration:
    """Base class for database migrations"""
    
    @property
    def version(self) -> str:
        """Migration version identifier"""
        raise NotImplementedError
    
    @property
    def description(self) -> str:
        """Migration description"""
        raise NotImplementedError
    
    def up(self, connection: sqlite3.Connection) -> None:
        """Apply migration"""
        raise NotImplementedError
    
    def down(self, connection: sqlite3.Connection) -> None:
        """Rollback migration"""
        raise NotImplementedError


class UnifiedDatabaseManager:
    """
    Unified database manager that consolidates all database operations
    and provides a single interface for database access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, database_url: str = None):
        """Singleton implementation with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, database_url: str = None):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.database_url = database_url or "sqlite:///data/langgraph_101.db"
        self.db_path = self._extract_db_path()
        self._connection_pool = {}
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._migrations = []
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        self._register_migrations()
        self._run_migrations()
    
    def _extract_db_path(self) -> str:
        """Extract database file path from URL"""
        if self.database_url.startswith("sqlite:///"):
            return self.database_url.replace("sqlite:///", "")
        elif self.database_url.startswith("sqlite://"):
            return self.database_url.replace("sqlite://", "")
        else:
            return self.database_url
    
    @contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """Get a database connection with automatic cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def _initialize_database(self):
        """Initialize database with all required tables"""
        logger.info("Initializing unified database...")
        
        with self.get_connection() as conn:
            # Create migrations table first
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create all core tables
            self._create_core_tables(conn)
            self._create_content_tables(conn)
            self._create_analytics_tables(conn)
            self._create_security_tables(conn)
            self._create_system_tables(conn)
            
            conn.commit()
            logger.info("Database initialization completed")
    
    def _create_core_tables(self, conn: sqlite3.Connection):
        """Create core application tables"""
        
        # Conversations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                persona_name TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                metadata TEXT
            )
        """)
        
        # Messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        """)
        
        # Memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                source_message_id TEXT,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE,
                FOREIGN KEY (source_message_id) REFERENCES messages (id) ON DELETE SET NULL
            )
        """)
        
        # Agent tasks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                task_type TEXT NOT NULL,
                parameters TEXT,
                status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes for core tables
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memory_conversation_id ON memory(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)",
            "CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory(importance)",
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_id ON agent_tasks(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_agent_tasks_created_at ON agent_tasks(created_at)"
        ]
        
        for index in indexes:
            conn.execute(index)
    
    def _create_content_tables(self, conn: sqlite3.Connection):
        """Create content management tables"""
        
        # Content table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                content_data TEXT NOT NULL,
                status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_at TIMESTAMP,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # Content templates table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                content_type TEXT NOT NULL,
                category TEXT NOT NULL,
                template_data TEXT NOT NULL,
                variables TEXT,
                usage_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                metadata TEXT
            )
        """)
        
        # Content calendar table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_calendar (
                id TEXT PRIMARY KEY,
                content_id TEXT,
                title TEXT NOT NULL,
                description TEXT,
                scheduled_date TIMESTAMP NOT NULL,
                platform TEXT NOT NULL,
                status TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'published', 'cancelled')),
                assigned_to TEXT,
                priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (content_id) REFERENCES content (id) ON DELETE CASCADE
            )
        """)
        
        # Content history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_history (
                id TEXT PRIMARY KEY,
                content_id TEXT,
                action TEXT NOT NULL,
                old_data TEXT,
                new_data TEXT,
                changed_by TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (content_id) REFERENCES content (id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for content tables
        content_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type)",
            "CREATE INDEX IF NOT EXISTS idx_content_status ON content(status)",
            "CREATE INDEX IF NOT EXISTS idx_content_created_at ON content(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_content_templates_category ON content_templates(category)",
            "CREATE INDEX IF NOT EXISTS idx_content_templates_usage ON content_templates(usage_count)",
            "CREATE INDEX IF NOT EXISTS idx_content_calendar_scheduled_date ON content_calendar(scheduled_date)",
            "CREATE INDEX IF NOT EXISTS idx_content_calendar_platform ON content_calendar(platform)",
            "CREATE INDEX IF NOT EXISTS idx_content_calendar_status ON content_calendar(status)"
        ]
        
        for index in content_indexes:
            conn.execute(index)
    
    def _create_analytics_tables(self, conn: sqlite3.Connection):
        """Create analytics and performance tables"""
        
        # Performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                response_time REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1,
                api_provider TEXT,
                cache_hit BOOLEAN DEFAULT 0,
                resource_usage TEXT,
                error_message TEXT,
                metadata TEXT
            )
        """)
        
        # Content analytics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_analytics (
                id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                platform TEXT,
                metadata TEXT,
                FOREIGN KEY (content_id) REFERENCES content (id) ON DELETE CASCADE
            )
        """)
        
        # API usage tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                endpoint TEXT,
                request_size INTEGER,
                response_size INTEGER,
                cost REAL,
                rate_limit_remaining INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes for analytics tables
        analytics_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation_type ON performance_metrics(operation_type)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_content_analytics_content_id ON content_analytics(content_id)",
            "CREATE INDEX IF NOT EXISTS idx_content_analytics_metric_name ON content_analytics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_provider ON api_usage(provider)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp)"
        ]
        
        for index in analytics_indexes:
            conn.execute(index)
    
    def _create_security_tables(self, conn: sqlite3.Connection):
        """Create security and user management tables"""
        
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer')),
                is_active BOOLEAN DEFAULT 1,
                is_verified BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # User sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        
        # API keys table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                permissions TEXT,
                expires_at TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        
        # User preferences table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, preference_key)
            )
        """)
        
        # Create indexes for security tables
        security_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)",
            "CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id)"
        ]
        
        for index in security_indexes:
            conn.execute(index)
    
    def _create_system_tables(self, conn: sqlite3.Connection):
        """Create system monitoring and logging tables"""
        
        # System logs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id TEXT PRIMARY KEY,
                level TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                module TEXT,
                function_name TEXT,
                line_number INTEGER,
                exception_info TEXT,
                extra_data TEXT
            )
        """)
        
        # System metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hostname TEXT,
                process_id INTEGER,
                thread_id TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes for system tables
        system_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_logger_name ON system_logs(logger_name)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON system_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)"
        ]
        
        for index in system_indexes:
            conn.execute(index)
    
    def _register_migrations(self):
        """Register database migrations"""
        # Add migrations here as they're created
        pass
    
    def _run_migrations(self):
        """Run pending database migrations"""
        with self.get_connection() as conn:
            # Get applied migrations
            applied = set()
            try:
                cursor = conn.execute("SELECT version FROM migrations")
                applied = {row[0] for row in cursor.fetchall()}
            except sqlite3.OperationalError:
                # Migrations table doesn't exist yet
                pass
            
            # Run pending migrations
            for migration in self._migrations:
                if migration.version not in applied:
                    logger.info(f"Running migration {migration.version}: {migration.description}")
                    try:
                        migration.up(conn)
                        conn.execute(
                            "INSERT INTO migrations (version, description) VALUES (?, ?)",
                            (migration.version, migration.description)
                        )
                        conn.commit()
                        logger.info(f"Migration {migration.version} completed successfully")
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Migration {migration.version} failed: {e}")
                        raise DatabaseError(f"Migration failed: {e}")
    
    # Data access methods
    
    def execute_query(self, query: str, params: Tuple = None, fetch: bool = True) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            if fetch:
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return []
    
    def insert_record(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a record into a table"""
        columns = list(data.keys())
        placeholders = ['?' for _ in columns]
        values = list(data.values())
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, values)
            conn.commit()
            return str(cursor.lastrowid)
    
    def update_record(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """Update a record in a table"""
        set_clauses = [f"{key} = ?" for key in data.keys()]
        values = list(data.values()) + [record_id]
        
        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = ?"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_record(self, table: str, record_id: str) -> bool:
        """Delete a record from a table"""
        query = f"DELETE FROM {table} WHERE id = ?"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (record_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_record(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a single record by ID"""
        query = f"SELECT * FROM {table} WHERE id = ?"
        results = self.execute_query(query, (record_id,))
        return results[0] if results else None
    
    def get_records(self, table: str, conditions: Dict[str, Any] = None, limit: int = None, order_by: str = None) -> List[Dict[str, Any]]:
        """Get multiple records with optional filtering"""
        query = f"SELECT * FROM {table}"
        params = []
        
        if conditions:
            where_clauses = [f"{key} = ?" for key in conditions.keys()]
            query += f" WHERE {' AND '.join(where_clauses)}"
            params.extend(conditions.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, tuple(params))
    
    # Specialized methods for common operations
    
    def save_conversation(self, conversation_id: str, persona_name: str, title: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Save a conversation"""
        data = {
            'id': conversation_id,
            'persona_name': persona_name,
            'title': title,
            'metadata': json.dumps(metadata) if metadata else None
        }
        try:
            self.insert_record('conversations', data)
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False
    
    def save_message(self, message_id: str, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Save a message"""
        data = {
            'id': message_id,
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'metadata': json.dumps(metadata) if metadata else None
        }
        try:
            self.insert_record('messages', data)
            # Update conversation timestamp
            self.update_record('conversations', conversation_id, {'updated_at': datetime.now().isoformat()})
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def save_memory(self, memory_id: str, conversation_id: str, category: str, content: str, 
                   importance: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """Save a memory"""
        data = {
            'id': memory_id,
            'conversation_id': conversation_id,
            'category': category,
            'content': content,
            'importance': importance,
            'metadata': json.dumps(metadata) if metadata else None
        }
        try:
            self.insert_record('memory', data)
            return True
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return False
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation"""
        return self.get_records('messages', {'conversation_id': conversation_id}, order_by='timestamp ASC')
    
    def get_conversation_memories(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a conversation"""
        return self.get_records('memory', {'conversation_id': conversation_id}, order_by='importance DESC')
    
    # Performance and analytics methods
    
    def log_performance_metric(self, operation_type: str, response_time: float, success: bool = True, 
                             api_provider: str = None, cache_hit: bool = False, metadata: Dict[str, Any] = None) -> bool:
        """Log a performance metric"""
        data = {
            'id': hashlib.md5(f"{operation_type}-{datetime.now().isoformat()}".encode()).hexdigest(),
            'operation_type': operation_type,
            'response_time': response_time,
            'success': success,
            'api_provider': api_provider,
            'cache_hit': cache_hit,
            'metadata': json.dumps(metadata) if metadata else None
        }
        try:
            self.insert_record('performance_metrics', data)
            return True
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
            return False
    
    def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance analytics for the specified period"""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Overall statistics
        overall_query = """
            SELECT 
                COUNT(*) as total_operations,
                AVG(response_time) as avg_response_time,
                MIN(response_time) as min_response_time,
                MAX(response_time) as max_response_time,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
            FROM performance_metrics 
            WHERE timestamp >= ?
        """
        
        overall_stats = self.execute_query(overall_query, (start_date,))[0]
        
        # Operation breakdown
        breakdown_query = """
            SELECT 
                operation_type,
                COUNT(*) as count,
                AVG(response_time) as avg_time,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
            FROM performance_metrics 
            WHERE timestamp >= ?
            GROUP BY operation_type
            ORDER BY count DESC
        """
        
        breakdown = self.execute_query(breakdown_query, (start_date,))
        
        return {
            'period_days': days,
            'overall': overall_stats,
            'by_operation': breakdown,
            'generated_at': datetime.now().isoformat()
        }
    
    # Backup and maintenance methods
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/langgraph_101_backup_{timestamp}.db"
        
        # Ensure backup directory exists
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with self.get_connection() as conn:
                backup = sqlite3.connect(backup_path)
                conn.backup(backup)
                backup.close()
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise DatabaseError(f"Backup failed: {e}")
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        if not Path(backup_path).exists():
            raise DatabaseError(f"Backup file not found: {backup_path}")
        
        try:
            # Create a backup of current database before restore
            current_backup = self.backup_database()
            logger.info(f"Created safety backup: {current_backup}")
            
            # Restore from backup
            backup_conn = sqlite3.connect(backup_path)
            with self.get_connection() as conn:
                backup_conn.backup(conn)
            backup_conn.close()
            
            logger.info(f"Database restored from: {backup_path}")
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise DatabaseError(f"Restore failed: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to keep database size manageable"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cleanup_queries = [
            ("performance_metrics", "timestamp"),
            ("system_logs", "timestamp"),
            ("system_metrics", "timestamp"),
            ("api_usage", "timestamp")
        ]
        
        total_deleted = 0
        for table, date_column in cleanup_queries:
            query = f"DELETE FROM {table} WHERE {date_column} < ?"
            with self.get_connection() as conn:
                cursor = conn.execute(query, (cutoff_date,))
                deleted = cursor.rowcount
                conn.commit()
                total_deleted += deleted
                logger.info(f"Cleaned up {deleted} old records from {table}")
        
        logger.info(f"Total records cleaned up: {total_deleted}")
        return total_deleted
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {}
        
        # Get table sizes
        with self.get_connection() as conn:
            # Get all table names
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            tables = [row[0] for row in conn.execute(tables_query).fetchall()]
            
            for table in tables:
                count_query = f"SELECT COUNT(*) FROM {table}"
                count = conn.execute(count_query).fetchone()[0]
                stats[f"{table}_count"] = count
            
            # Get database file size
            stats['file_size_bytes'] = Path(self.db_path).stat().st_size
            stats['file_size_mb'] = stats['file_size_bytes'] / (1024 * 1024)
              # Get database info
            stats['sqlite_version'] = conn.execute("SELECT sqlite_version()").fetchone()[0]
            stats['page_size'] = conn.execute("PRAGMA page_size").fetchone()[0]
            stats['page_count'] = conn.execute("PRAGMA page_count").fetchone()[0]
        
        return stats
    
    def vacuum_database(self):
        """Optimize database by running VACUUM"""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuum completed successfully")
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            raise DatabaseError(f"Vacuum failed: {e}")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                # Test basic query
                conn.execute("SELECT 1").fetchone()
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self._executor:
            self._executor.shutdown(wait=True)


# Global database manager instance
_db_manager = None


def get_database_manager(database_url: str = None) -> UnifiedDatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        from core.config import get_config
        config = get_config()
        database_url = database_url or config.get_database_url()
        _db_manager = UnifiedDatabaseManager(database_url)
    return _db_manager


# Convenience functions for common database operations

def save_conversation(conversation_id: str, persona_name: str, title: str = None, metadata: Dict[str, Any] = None) -> bool:
    """Save a conversation"""
    db = get_database_manager()
    return db.save_conversation(conversation_id, persona_name, title, metadata)


def save_message(message_id: str, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
    """Save a message"""
    db = get_database_manager()
    return db.save_message(message_id, conversation_id, role, content, metadata)


def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a conversation"""
    db = get_database_manager()
    return db.get_conversation_messages(conversation_id)


def log_performance_metric(operation_type: str, response_time: float, success: bool = True, 
                         api_provider: str = None, cache_hit: bool = False) -> bool:
    """Log a performance metric"""
    db = get_database_manager()
    return db.log_performance_metric(operation_type, response_time, success, api_provider, cache_hit)


if __name__ == "__main__":
    # Test database manager
    try:
        db = get_database_manager()
        print("✅ Database manager initialized successfully")
        
        # Test basic operations
        stats = db.get_database_stats()
        print(f"📊 Database stats: {stats}")
        
        # Test performance logging
        success = log_performance_metric("test_operation", 0.5, True, "test_provider")
        print(f"📈 Performance metric logged: {success}")
        
    except Exception as e:
        print(f"❌ Database manager error: {e}")
