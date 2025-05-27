"""
Database module for the Don Corleone AI project.

This module provides database functionality for storing conversations,
memories, and agent task results persistently.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """SQLite database for persistent storage."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database.

        Args:
            db_path: Path to the database file. If None, a default path will be used.
        """
        if db_path is None:
            # Use default path in the project directory
            db_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "don_corleone.db")

        self.db_path = db_path
        self.conn = None

        # Initialize database
        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to the SQLite database."""
        try:
            # Quick fix for thread safety in Streamlit: allow connection across threads
            # TODO: For production, use SQLAlchemy or a connection pool for robust thread management
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Use Row factory for better row access
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()

            # Create conversation table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE,
                    persona_name TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')

            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    message_id TEXT UNIQUE,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
            ''')

            # Create memories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE,
                    conversation_id TEXT,
                    category TEXT,
                    content TEXT,
                    source_message TEXT,
                    importance REAL,
                    timestamp TEXT,
                    access_count INTEGER DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
            ''')

            # Create agent tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    agent_id TEXT,
                    agent_type TEXT,
                    task_type TEXT,
                    parameters TEXT,
                    status TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    result TEXT
                )
            ''')

            # Create user preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_key TEXT UNIQUE,
                    preference_value TEXT,
                    updated_at TEXT
                )
            ''')

            # Add agents table to manage multiple agents
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT UNIQUE,
                    agent_type TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')

            # Add agent_interactions table to log interactions between agents
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT UNIQUE,
                    agent_id TEXT,
                    target_agent_id TEXT,
                    interaction_type TEXT,
                    content TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
                )
            ''')

            # Add tools table to manage additional tools
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT UNIQUE,
                    tool_name TEXT,
                    configuration TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')

            # Add indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages (conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON agent_tasks (agent_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tool_id ON tools (tool_id)')

            # Add collaboration_sessions table for multi-agent collaboration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collaboration_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    agent_ids TEXT, -- Comma-separated list of agent IDs
                    session_type TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')

            # Add tool_usage_logs table to log tool usage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT,
                    agent_id TEXT,
                    usage_details TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (tool_id) REFERENCES tools(tool_id) ON DELETE CASCADE,
                    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
                )
            ''')

            self.conn.commit()
            logger.info("Database tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    # Conversation methods

    def create_conversation(self, conversation_id: str, persona_name: str) -> bool:
        """Create a new conversation.

        Args:
            conversation_id: Unique ID for the conversation.
            persona_name: Name of the persona for this conversation.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute('''
                INSERT INTO conversations (conversation_id, persona_name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (conversation_id, persona_name, now, now))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error creating conversation: {e}")
            self.conn.rollback()
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation to retrieve.

        Returns:
            Dictionary with conversation data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM conversations WHERE conversation_id = ?
            ''', (conversation_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting conversation: {e}")
            return None

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations.

        Returns:
            List of conversation dictionaries.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM conversations ORDER BY updated_at DESC
            ''')

            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting all conversations: {e}")
            return []

    def add_message(self, conversation_id: str, message_id: str, role: str, content: str) -> bool:
        """Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation.
            message_id: Unique ID for the message.
            role: Role of the message sender (user, assistant, system).
            content: Content of the message.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()

            # Insert the message
            cursor.execute('''
                INSERT INTO messages (conversation_id, message_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (conversation_id, message_id, role, content, now))

            # Update conversation updated_at
            cursor.execute('''
                UPDATE conversations SET updated_at = ? WHERE conversation_id = ?
            ''', (now, conversation_id))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding message: {e}")
            self.conn.rollback()
            return False

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            List of message dictionaries.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp
            ''', (conversation_id,))

            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting messages: {e}")
            return []

    # Memory methods

    def add_memory(self, memory_id: str, category: str, content: str, importance: int) -> bool:
        """Add a memory.

        Args:
            memory_id: Unique ID for the memory.
            category: Category of the memory.
            content: Content of the memory.
            importance: Importance score of the memory.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute('''
                INSERT INTO memories (memory_id, category, content, importance, timestamp, access_count)
                VALUES (?, ?, ?, ?, ?, 0)
            ''', (memory_id, category, content, importance, now))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding memory: {e}")
            self.conn.rollback()
            return False

    def increment_memory_access(self, memory_id: str) -> bool:
        """Increment the access count for a memory.

        Args:
            memory_id: ID of the memory.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                UPDATE memories
                SET access_count = access_count + 1
                WHERE memory_id = ?
            ''', (memory_id,))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error incrementing memory access: {e}")
            self.conn.rollback()
            return False

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories.

        Returns:
            List of memory dictionaries.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM memories
                ORDER BY importance DESC, timestamp DESC
            ''')

            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting all memories: {e}")
            return []

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID.

        Args:
            memory_id: ID of the memory to retrieve.

        Returns:
            Memory dictionary, or None if not found.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM memories
                WHERE memory_id = ?
            ''', (memory_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting memory: {e}")
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                DELETE FROM memories
                WHERE memory_id = ?
            ''', (memory_id,))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error deleting memory: {e}")
            self.conn.rollback()
            return False

    def clear_memories(self) -> bool:
        """Clear all memories.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                DELETE FROM memories
            ''')

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error clearing memories: {e}")
            self.conn.rollback()
            return False

    def create_memory(self, conversation_id: str, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Create a new memory associated with a conversation.

        Args:
            conversation_id: ID of the conversation this memory belongs to.
            memory_id: Unique ID for the memory.
            memory_data: Dictionary with memory data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()

            # Add conversation_id to memory data
            memory_data['conversation_id'] = conversation_id

            cursor.execute('''
                INSERT INTO memories (
                    memory_id,
                    conversation_id,
                    category,
                    content,
                    source_message,
                    importance,
                    timestamp,
                    access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_id,
                conversation_id,
                memory_data.get('category', 'general'),
                memory_data.get('content', ''),
                memory_data.get('source_message', ''),
                memory_data.get('importance', 1.0),
                memory_data.get('timestamp', now),
                memory_data.get('access_count', 0)
            ))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error creating memory: {e}")
            self.conn.rollback()
            return False

    def update_memory(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Update an existing memory.

        Args:
            memory_id: ID of the memory to update.
            memory_data: Dictionary with updated memory data.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()

            # Update specified fields
            cursor.execute('''
                UPDATE memories SET
                    category = ?,
                    content = ?,
                    source_message = ?,
                    importance = ?,
                    access_count = ?
                WHERE memory_id = ?
            ''', (
                memory_data.get('category', 'general'),
                memory_data.get('content', ''),
                memory_data.get('source_message', ''),
                memory_data.get('importance', 1.0),
                memory_data.get('access_count', 0),
                memory_id
            ))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating memory: {e}")
            self.conn.rollback()
            return False

    def get_memories(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a specific conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            List of memory dictionaries.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM memories
                WHERE conversation_id = ?
                ORDER BY importance DESC, timestamp DESC
            ''', (conversation_id,))

            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting memories for conversation: {e}")
            return []

    def delete_conversation_memories(self, conversation_id: str) -> bool:
        """Delete all memories associated with a conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                DELETE FROM memories
                WHERE conversation_id = ?
            ''', (conversation_id,))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error deleting conversation memories: {e}")
            self.conn.rollback()
            return False

    # Agent task methods

    def save_agent_task(self, task_id: str, agent_id: str, agent_type: str,
                       task_type: str, parameters: Dict[str, Any], status: str,
                       created_at: str, completed_at: Optional[str] = None,
                       result: Optional[Dict[str, Any]] = None) -> bool:
        """Save an agent task.

        Args:
            task_id: Unique ID for the task.
            agent_id: ID of the agent that executed the task.
            agent_type: Type of the agent.
            task_type: Type of the task.
            parameters: Parameters for the task.
            status: Status of the task.
            created_at: Creation timestamp.
            completed_at: Completion timestamp (optional).
            result: Task result (optional).

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()

            # Convert dictionaries to JSON
            parameters_json = json.dumps(parameters)
            result_json = json.dumps(result) if result else None

            cursor.execute('''
                INSERT OR REPLACE INTO agent_tasks
                (task_id, agent_id, agent_type, task_type, parameters, status, created_at, completed_at, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (task_id, agent_id, agent_type, task_type, parameters_json,
                  status, created_at, completed_at, result_json))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving agent task: {e}")
            self.conn.rollback()
            return False

    def get_agent_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent task by ID.

        Args:
            task_id: ID of the task to retrieve.

        Returns:
            Task dictionary, or None if not found.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT * FROM agent_tasks
                WHERE task_id = ?
            ''', (task_id,))

            row = cursor.fetchone()
            if row:
                task = dict(row)

                # Parse JSON strings back to dictionaries
                if task["parameters"]:
                    task["parameters"] = json.loads(task["parameters"])
                if task["result"]:
                    task["result"] = json.loads(task["result"])

                return task
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting agent task: {e}")
            return None

    def get_agent_tasks(self, agent_id: Optional[str] = None,
                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get agent tasks, optionally filtered by agent ID and/or status.

        Args:
            agent_id: Optional ID of the agent to filter by.
            status: Optional status to filter by.

        Returns:
            List of task dictionaries.
        """
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM agent_tasks"
            params = []

            if agent_id or status:
                query += " WHERE"

                if agent_id:
                    query += " agent_id = ?"
                    params.append(agent_id)

                    if status:
                        query += " AND"

                if status:
                    query += " status = ?"
                    params.append(status)

            query += " ORDER BY created_at DESC"

            cursor.execute(query, params)

            tasks = []
            for row in cursor.fetchall():
                task = dict(row)

                # Parse JSON strings back to dictionaries
                if task["parameters"]:
                    task["parameters"] = json.loads(task["parameters"])
                if task["result"]:
                    task["result"] = json.loads(task["result"])

                tasks.append(task)

            return tasks
        except sqlite3.Error as e:
            logger.error(f"Error getting agent tasks: {e}")
            return []

    def log_agent_interaction(self, from_agent_id: str, to_agent_id: str, task_id: str, action: str, details: Optional[str] = None):
        """Log an agent-to-agent interaction in the agent_interactions table."""
        query = """
        INSERT INTO agent_interactions (from_agent_id, to_agent_id, task_id, action, details, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        timestamp = datetime.now().isoformat()
        with self.conn:
            self.conn.execute(query, (from_agent_id, to_agent_id, task_id, action, details, timestamp))
        logger.info(f"Logged agent interaction: {from_agent_id} -> {to_agent_id}, Task: {task_id}, Action: {action}")

    # User preference methods

    def set_preference(self, key: str, value: Any) -> bool:
        """Set a user preference.

        Args:
            key: Preference key.
            value: Preference value.

        Returns:
            True if successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()

            # Convert value to JSON if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)

            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences
                (preference_key, preference_value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, now))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error setting preference: {e}")
            self.conn.rollback()
            return False

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference.

        Args:
            key: Preference key.
            default: Default value if preference is not found.

        Returns:
            Preference value, or default if not found.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT preference_value FROM user_preferences
                WHERE preference_key = ?
            ''', (key,))

            row = cursor.fetchone()
            if row:
                value = row["preference_value"]

                # Try to parse as JSON, return as string if it fails
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            return default
        except sqlite3.Error as e:
            logger.error(f"Error getting preference: {e}")
            return default

    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all user preferences.

        Returns:
            Dictionary of preference key-value pairs.
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                SELECT preference_key, preference_value FROM user_preferences
            ''')

            preferences = {}
            for row in cursor.fetchall():
                key = row["preference_key"]
                value = row["preference_value"]

                # Try to parse as JSON, leave as string if it fails
                try:
                    preferences[key] = json.loads(value)
                except json.JSONDecodeError:
                    preferences[key] = value

            return preferences
        except sqlite3.Error as e:
            logger.error(f"Error getting all preferences: {e}")
            return {}

    # Backup and restore methods

    def backup(self, backup_path: Optional[str] = None) -> bool:
        """Backup the database.

        Args:
            backup_path: Path for the backup file. If None, a default path will be used.

        Returns:
            True if successful, False otherwise.
        """
        if backup_path is None:
            # Create a default backup path
            backup_dir = os.path.join(os.path.dirname(__file__), "backups")
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"don_corleone_backup_{timestamp}.db")

        try:
            # Create a new connection for the backup
            backup_conn = sqlite3.connect(backup_path)

            # Back up the database
            with backup_conn:
                self.conn.backup(backup_conn)

            backup_conn.close()
            logger.info(f"Database backed up to {backup_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error backing up database: {e}")
            return False

    def restore(self, backup_path: str) -> bool:
        """Restore the database from a backup.

        Args:
            backup_path: Path to the backup file.

        Returns:
            True if successful, False otherwise.
        """
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            # Close the current connection
            self.close()

            # Create a backup of the current database
            current_backup = f"{self.db_path}.bak"
            if os.path.exists(self.db_path):
                os.rename(self.db_path, current_backup)

            # Copy the backup to the current database path
            backup_conn = sqlite3.connect(backup_path)
            db_conn = sqlite3.connect(self.db_path)

            with db_conn:
                backup_conn.backup(db_conn)

            backup_conn.close()
            db_conn.close()

            # Reconnect to the database
            self._connect()

            logger.info(f"Database restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error restoring database: {e}")

            # Try to restore the original database
            if os.path.exists(current_backup):
                try:
                    os.replace(current_backup, self.db_path)
                    self._connect()
                except Exception as restore_error:
                    logger.error(f"Error restoring original database: {restore_error}")

            return False


# Create a singleton instance
_db_instance = None

def get_database() -> Database:
    """Get the database instance.

    Returns:
        Database instance.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
