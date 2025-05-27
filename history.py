"""
History module for LangGraph 101 project.

This module handles the conversation history management.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os
from datetime import datetime
import uuid
import logging
from database import get_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationHistory:
    """Class to manage conversation history.

    This class tracks and manages the conversation history between the user
    and the AI agent, allowing for context-aware responses. The history is
    stored persistently in a database.
    """

    def __init__(self, max_history: int = 20, conversation_id: Optional[str] = None,
                 persona_name: Optional[str] = None, history_file: Optional[str] = None):
        """Initialize a new conversation history.

        Args:
            max_history: Maximum number of messages to store in history.
            conversation_id: Optional ID for the conversation. If not provided, a UUID will be generated.
            persona_name: Name of the persona for this conversation.
            history_file: Optional path to a file for saving history (for backwards compatibility).
        """
        self.messages = []
        self.max_history = max_history
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.db = get_database()
        self.history_file = history_file

        # Ensure conversation exists in the database
        if not self.db.get_conversation(self.conversation_id):
            # Use provided persona_name or a default if not available
            current_persona_name = persona_name if persona_name else "DefaultPersona"
            if self.db.create_conversation(self.conversation_id, current_persona_name):
                logger.info(f"Created new conversation {self.conversation_id} with persona {current_persona_name}")
            else:
                # This is a critical failure, should probably raise an exception or handle more gracefully
                logger.error(f"Failed to create conversation {self.conversation_id} with persona {current_persona_name} in database. History may be inconsistent.")
                # For now, it will proceed, but errors are likely.

        # Load messages from database
        self._load_from_db()

        # Try loading from file if provided (backwards compatibility)
        if history_file and os.path.exists(history_file):
            self.load()

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history.

        Args:
            content: Content of the user message.
        """
        message_id = str(uuid.uuid4())
        message = {"role": "user", "content": content, "id": message_id}
        self.messages.append(message)

        # Save to database
        self.db.add_message(self.conversation_id, message_id, "user", content)

        self._truncate_history()

    def add_ai_message(self, content: str) -> None:
        """Add an AI message to the history.

        Args:
            content: Content of the AI message.
        """
        message_id = str(uuid.uuid4())
        message = {"role": "assistant", "content": content, "id": message_id}
        self.messages.append(message)

        # Save to database
        self.db.add_message(self.conversation_id, message_id, "assistant", content)

        self._truncate_history()

    def add_system_message(self, content: str) -> None:
        """Add a system message to the history.

        Args:
            content: Content of the system message.
        """
        message_id = str(uuid.uuid4())
        message = {"role": "system", "content": content, "id": message_id}
        self.messages.append(message)

        # Save to database
        self.db.add_message(self.conversation_id, message_id, "system", content)

        self._truncate_history()

    def add_agent_message(self, content: str) -> None:
        """Add an agent message to the history.

        Args:
            content: Content of the agent message.
        """
        message_id = str(uuid.uuid4())
        message = {"role": "agent", "content": content, "id": message_id}
        self.messages.append(message)

        # Save to database
        self.db.add_message(self.conversation_id, message_id, "agent", content)

        self._truncate_history()

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of message dictionaries with role and content keys.
        """
        # Return messages without the internal ID
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self) -> None:
        """Clear the conversation history in memory.
        Note: This does not delete messages from the database.
        """
        self.messages = []

    def _load_from_db(self) -> None:
        """Load messages from the database."""
        try:
            db_messages = self.db.get_messages(self.conversation_id)

            if db_messages:
                self.messages = [
                    {"role": m["role"], "content": m["content"], "id": m["message_id"]}
                    for m in db_messages
                ]
                logger.info(f"Loaded {len(self.messages)} messages from database for conversation {self.conversation_id}")
            else:
                logger.info(f"No messages found in database for conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Error loading messages from database: {e}")

    def _truncate_history(self) -> None:
        """Truncate the history to the maximum length."""
        if len(self.messages) > self.max_history:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_history:]

    def save(self) -> bool:
        """Save the conversation history to a file.

        Returns:
            True if successful, False otherwise.
        """
        if not self.history_file:
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

            # Convert database-style messages to file-style format for backward compatibility
            file_messages = []
            for msg in self.messages:
                file_msg = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": datetime.now().isoformat()  # Add timestamp for backward compatibility
                }
                file_messages.append(file_msg)

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "messages": file_messages,
                    "timestamp": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving conversation history to file: {e}")
            return False

    def load(self) -> bool:
        """Load the conversation history from a file.

        Returns:
            True if successful, False otherwise.
        """
        if not self.history_file or not os.path.exists(self.history_file):
            return False

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_messages = data.get("messages", [])

                # Convert file-style messages to database-style format
                self.messages = []
                for msg in file_messages:
                    db_msg = {
                        "role": "assistant" if msg["role"] == "ai" else msg["role"],  # Convert "ai" to "assistant"
                        "content": msg["content"],
                        "id": str(uuid.uuid4())  # Generate new IDs for loaded messages
                    }
                    self.messages.append(db_msg)
            return True
        except Exception as e:
            logger.error(f"Error loading conversation history from file: {e}")
            return False


def get_conversation_history(conversation_id: Optional[str] = None, persona_name: Optional[str] = None, max_history: int = 20) -> ConversationHistory:
    """Get a conversation history instance.

    Args:
        conversation_id: Optional ID for the conversation.
        persona_name: Optional name of the persona for this conversation.
        max_history: Maximum number of messages to store in history.

    Returns:
        A ConversationHistory instance.
    """
    # For backwards compatibility with tests, set up history file if persona is specified
    history_file = None
    if persona_name:
        # Create filename based on current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"

        # Create history directory
        history_dir = os.path.join(os.path.dirname(__file__), "history")
        os.makedirs(history_dir, exist_ok=True)

        # Set history file path
        history_file = os.path.join(history_dir, filename)

    return ConversationHistory(
        max_history=max_history,
        conversation_id=conversation_id,
        persona_name=persona_name,
        history_file=history_file
    )


def get_history_manager(conversation_id: Optional[str] = None, persona_name: Optional[str] = None, max_history: int = 20) -> ConversationHistory:
    """Alias for get_conversation_history for backwards compatibility.

    Args:
        conversation_id: Optional ID for the conversation.
        persona_name: Optional name of the persona for this conversation.
        max_history: Maximum number of messages to store in history.

    Returns:
        A ConversationHistory instance.
    """
    return get_conversation_history(conversation_id=conversation_id, persona_name=persona_name, max_history=max_history)
