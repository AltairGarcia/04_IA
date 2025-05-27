"""
Agent Commands for the Don Corleone AI project.

This module handles natural language commands to autonomous agents,
allowing users to interact with agents directly through the chat interface.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import create_agent, AgentStatus, TaskConfig, Task, ExecutionResult
from agents import ResearchAgent, DocumentAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommandProcessor:
    """Process natural language commands to agents."""

    # Command patterns for different agent actions
    COMMAND_PATTERNS = {
        # Research agent commands
        "research": r"(?:research|search|find|look up|investigate)\s+(.*?)(?:\s+for\s+me|\s+please|\s*$)",
        "fact_check": r"(?:fact[- ]check|verify|confirm|is it true that)\s+(.*?)(?:\s+for\s+me|\s+please|\s*$)",
        "summarize_topic": r"(?:summarize|give me a summary of|tell me about)\s+(.*?)(?:\s+for\s+me|\s+please|\s*$)",

        # Document agent commands
        "summarize_document": r"(?:summarize|give me a summary of|summarize document|summarize file)\s+(?:the\s+)?(?:document|file|text)\s+(?:at\s+)?([\w\\/:.]+)(?:\s+for\s+me|\s+please|\s*$)",
        "extract_from_document": r"(?:extract|pull|find)\s+(.*?)\s+from\s+(?:the\s+)?(?:document|file|text)\s+(?:at\s+)?([\w\\/:.]+)(?:\s+for\s+me|\s+please|\s*$)",
        "answer_about_document": r"(?:answer|tell me|can you tell me|explain|what|how|who|when|where|why)\s+(.*?)\s+(?:in|from|about)\s+(?:the\s+)?(?:document|file|text)\s+(?:at\s+)?([\w\\/:.]+)(?:\s+for\s+me|\s+please|\s*$)",
    }

    def __init__(self):
        """Initialize the command processor."""
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            cmd: re.compile(pattern, re.IGNORECASE)
            for cmd, pattern in self.COMMAND_PATTERNS.items()
        }

    def detect_command(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Detect if the text contains a command for an agent.

        Args:
            text: The text to analyze for commands.

        Returns:
            A tuple containing the command type (or None if no command detected)
            and a dictionary of extracted parameters.
        """
        for cmd, pattern in self.compiled_patterns.items():
            match = pattern.match(text.strip())
            if match:
                if cmd == "research":
                    return cmd, {"query": match.group(1).strip()}
                elif cmd == "fact_check":
                    return cmd, {"statement": match.group(1).strip()}
                elif cmd == "summarize_topic":
                    topic = match.group(1).strip()
                    # Try to extract aspects if specified
                    aspects = []
                    aspects_match = re.search(r"(.*?)\s+(?:focusing on|with focus on|about|regarding|on|covering)\s+(.*?)$", topic)
                    if aspects_match:
                        topic = aspects_match.group(1).strip()
                        aspects_text = aspects_match.group(2).strip()
                        aspects = [a.strip() for a in aspects_text.split(",")]

                    return cmd, {"topic": topic, "aspects": aspects}
                elif cmd == "summarize_document":
                    return cmd, {"file_path": match.group(1).strip()}
                elif cmd == "extract_from_document":
                    return cmd, {
                        "pattern": match.group(1).strip(),
                        "file_path": match.group(2).strip()
                    }
                elif cmd == "answer_about_document":
                    return cmd, {
                        "question": match.group(1).strip(),
                        "file_path": match.group(2).strip()
                    }

        return None, {}

    def process_command(self, text: str, agents_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[ExecutionResult], Optional[str]]:
        """Process a command and execute the appropriate agent task.

        Args:
            text: The command text.
            agents_dict: Dictionary of available agents.

        Returns:
            A tuple containing:
            - Task ID (or None if no command was recognized)
            - Execution result (or None if task is pending or failed)
            - Error message (or None if successful)
        """
        command_type, params = self.detect_command(text)

        if not command_type:
            return None, None, None

        try:
            # Select appropriate agent and task type based on command
            if command_type in ["research", "fact_check", "summarize_topic"]:
                # Find research agent
                research_agents = [
                    agent_data for agent_id, agent_data in agents_dict.items()
                    if agent_data["type"] == "research"
                ]

                if not research_agents:
                    return None, None, "No research agent available. Please create one in the Agents tab."

                # Use the first available research agent
                agent_data = research_agents[0]
                agent_id = agent_data["id"]
                agent = agent_data["instance"]

                # Map command to task type
                task_type_map = {
                    "research": "web_research",
                    "fact_check": "fact_check",
                    "summarize_topic": "topic_summary"
                }

                task_type = task_type_map[command_type]

                # Create task config
                if task_type == "web_research":
                    task_config = TaskConfig(task_type=task_type, parameters={"query": params["query"], "num_sources": 3})
                elif task_type == "fact_check":
                    task_config = TaskConfig(task_type=task_type, parameters={"statement": params["statement"]})
                elif task_type == "topic_summary":
                    task_config = TaskConfig(task_type=task_type, parameters={"topic": params["topic"], "aspects": params["aspects"]})
                else:
                    return None, None, f"Unsupported task type: {task_type}"

            elif command_type in ["summarize_document", "extract_from_document", "answer_about_document"]:
                # Find document agent
                document_agents = [
                    agent_data for agent_id, agent_data in agents_dict.items()
                    if agent_data["type"] == "document"
                ]

                if not document_agents:
                    return None, None, "No document agent available. Please create one in the Agents tab."

                # Use the first available document agent
                agent_data = document_agents[0]
                agent_id = agent_data["id"]
                agent = agent_data["instance"]

                # Map command to task type
                task_type_map = {
                    "summarize_document": "document_summary",
                    "extract_from_document": "text_extraction",
                    "answer_about_document": "document_qa"
                }

                task_type = task_type_map[command_type]

                # Create task config
                if task_type == "document_summary":
                    task_config = TaskConfig(
                        task_type=task_type,
                        parameters={
                            "file_path": params["file_path"],
                            "max_length": 1000,
                            "format": "markdown",
                            "include_metadata": True
                        }
                    )
                elif task_type == "text_extraction":
                    task_config = TaskConfig(
                        task_type=task_type,
                        parameters={
                            "file_path": params["file_path"],
                            "type": "keyword",
                            "keywords": [params["pattern"]],
                            "context_size": 100
                        }
                    )
                elif task_type == "document_qa":
                    task_config = TaskConfig(
                        task_type=task_type,
                        parameters={
                            "file_path": params["file_path"],
                            "question": params["question"]
                        }
                    )
                else:
                    return None, None, f"Unsupported task type: {task_type}"
            else:
                return None, None, f"Unsupported command type: {command_type}"

            # Check if agent can handle task
            if not agent.can_handle_task(task_config):
                return None, None, f"Agent {agent_data['name']} cannot handle task type {task_type}"

            # Create the task
            task = agent.submit_task(task_config, f"Command: {text}")
            task_id = task.task_id

            # Execute the task
            result = agent.run_task(task_id)

            # Update task in agent data
            agent_data["tasks"][task_id] = {
                "task_id": task_id,
                "task_type": task_type,
                "parameters": task_config.parameters,
                "status": AgentStatus.COMPLETED if result.success else AgentStatus.FAILED,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at
            }

            # Return task ID and result
            if result.success:
                return task_id, result, None
            else:
                return task_id, None, result.error

        except Exception as e:
            logger.exception(f"Error processing command: {e}")
            return None, None, f"Error processing command: {str(e)}"


# Create a singleton instance
command_processor = CommandProcessor()


def process_agent_command(text: str, agents_dict: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[ExecutionResult]]:
    """Process a potential agent command in text.

    Args:
        text: The text to check for commands.
        agents_dict: Dictionary of available agents.

    Returns:
        Tuple containing:
        - Boolean indicating if text was a command
        - Response text (result or error message)
        - Execution result if successful
    """
    task_id, result, error = command_processor.process_command(text, agents_dict)

    if task_id is None:
        # No command detected
        return False, None, None

    if error:
        return True, f"I couldn't complete that task: {error}", None

    return True, result.output, result


def is_agent_command(text: str) -> bool:
    """Check if text contains an agent command.

    Args:
        text: The text to check.

    Returns:
        True if text contains an agent command, False otherwise.
    """
    command_type, _ = command_processor.detect_command(text)
    return command_type is not None
