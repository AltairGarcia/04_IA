"""
Base Agent module for the Don Corleone AI project.

This module defines the base Agent class that all specialized agents will inherit from.
It provides the core functionality for autonomous task execution, state management,
and interaction with external tools.
"""

import os
import json
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    """Status of an agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    IDLE = "idle"  # Added default status


class ExecutionResult:
    """Represents the result of an agent execution."""

    def __init__(
        self,
        success: bool,
        output: Any,
        error: Optional[str] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize a new execution result.

        Args:
            success: Whether the execution was successful.
            output: The output of the execution.
            error: Optional error message if the execution failed.
            artifacts: Optional list of artifacts generated during execution.
        """
        self.success = success
        self.output = output
        self.error = error
        self.artifacts = artifacts or []
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution result to a dictionary.

        Returns:
            Dictionary representation of the execution result.
        """
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create an execution result from a dictionary.

        Args:
            data: Dictionary containing execution result data.

        Returns:
            An ExecutionResult instance.
        """
        return cls(
            success=data["success"],
            output=data["output"],
            error=data.get("error"),
            artifacts=data.get("artifacts", [])
        )


class TaskConfig:
    """Configuration for an agent task."""

    def __init__(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        max_retries: int = 3,
        timeout_seconds: int = 300,
        requires_confirmation: bool = True
    ):
        """Initialize a new task configuration.

        Args:
            task_type: Type of the task.
            parameters: Parameters for the task.
            max_retries: Maximum number of retries if the task fails.
            timeout_seconds: Timeout for the task in seconds.
            requires_confirmation: Whether the task requires user confirmation before execution.
        """
        self.task_type = task_type
        self.parameters = parameters
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.requires_confirmation = requires_confirmation

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task configuration to a dictionary.

        Returns:
            Dictionary representation of the task configuration.
        """
        return {
            "task_type": self.task_type,
            "parameters": self.parameters,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "requires_confirmation": self.requires_confirmation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """Create a task configuration from a dictionary.

        Args:
            data: Dictionary containing task configuration data.

        Returns:
            A TaskConfig instance.
        """
        return cls(
            task_type=data["task_type"],
            parameters=data["parameters"],
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300),
            requires_confirmation=data.get("requires_confirmation", True)
        )


class Task:
    """Represents a task to be executed by an agent."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        config: Optional[TaskConfig] = None,
        description: Optional[str] = None
    ):
        """Initialize a new task.

        Args:
            task_id: Optional ID for the task. If not provided, a UUID will be generated.
            config: Configuration for the task.
            description: Human-readable description of the task.
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.config = config
        self.description = description
        self.status = AgentStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary.

        Returns:
            Dictionary representation of the task.
        """
        return {
            "task_id": self.task_id,
            "config": self.config.to_dict() if self.config else None,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result.to_dict() if self.result else None,
            "attempts": self.attempts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a task from a dictionary.

        Args:
            data: Dictionary containing task data.

        Returns:
            A Task instance.
        """
        task = cls(
            task_id=data["task_id"],
            config=TaskConfig.from_dict(data["config"]) if data.get("config") else None,
            description=data.get("description")
        )
        task.status = data.get("status", AgentStatus.PENDING)
        task.created_at = data.get("created_at", datetime.now().isoformat())
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        if data.get("result"):
            task.result = ExecutionResult.from_dict(data["result"])
        task.attempts = data.get("attempts", 0)

        return task


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        """Initialize a new agent.

        Args:
            agent_id: Optional ID for the agent. If not provided, a UUID will be generated.
            name: Optional name for the agent.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.tasks: Dict[str, Task] = {}
        self.logger = logging.getLogger(f"agent.{self.name}")

    @abstractmethod
    def can_handle_task(self, task_config: TaskConfig) -> bool:
        """Check if this agent can handle a task.

        Args:
            task_config: Configuration for the task.

        Returns:
            True if the agent can handle the task, False otherwise.
        """
        pass

    @abstractmethod
    def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a task.

        Args:
            task: The task to execute.

        Returns:
            The result of the execution.
        """
        pass

    def submit_task(self, config: TaskConfig, description: Optional[str] = None) -> Task:
        """Submit a task for execution.

        Args:
            config: Configuration for the task.
            description: Human-readable description of the task.

        Returns:
            The created task.

        Raises:
            ValueError: If the agent cannot handle the task.
        """
        if not self.can_handle_task(config):
            raise ValueError(f"Agent {self.name} cannot handle task type {config.task_type}")

        task = Task(config=config, description=description)
        self.tasks[task.task_id] = task

        # Log task submission
        self.logger.info(f"Task {task.task_id} submitted: {description or config.task_type}")

        return task

    def run_task(self, task_id: str) -> ExecutionResult:
        """Run a submitted task.

        Args:
            task_id: ID of the task to run.

        Returns:
            The result of the execution.

        Raises:
            ValueError: If the task does not exist.
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} does not exist")

        task = self.tasks[task_id]

        # Update task status
        task.status = AgentStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        task.attempts += 1

        # Log task start
        self.logger.info(f"Starting task {task_id}: {task.description or task.config.task_type}")

        try:
            # Execute the task
            result = self.execute_task(task)

            # Update task status based on result
            task.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
            task.completed_at = datetime.now().isoformat()
            task.result = result

            # Log task completion
            log_method = self.logger.info if result.success else self.logger.error
            log_method(f"Task {task_id} {'completed' if result.success else 'failed'}: {result.output or result.error}")

            return result

        except Exception as e:
            # Handle unexpected exceptions
            error_message = f"Error executing task {task_id}: {str(e)}"
            self.logger.exception(error_message)

            # Update task status
            task.status = AgentStatus.FAILED
            task.completed_at = datetime.now().isoformat()
            task.result = ExecutionResult(success=False, output=None, error=error_message)

            return task.result

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: ID of the task to get.

        Returns:
            The task if found, None otherwise.
        """
        return self.tasks.get(task_id)

    def get_tasks(self, status: Optional[AgentStatus] = None) -> List[Task]:
        """Get all tasks, optionally filtered by status.

        Args:
            status: Optional status to filter by.

        Returns:
            List of tasks.
        """
        if status is None:
            return list(self.tasks.values())

        return [task for task in self.tasks.values() if task.status == status]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: ID of the task to cancel.

        Returns:
            True if the task was cancelled, False otherwise.
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Only pending or running tasks can be cancelled
        if task.status not in [AgentStatus.PENDING, AgentStatus.RUNNING]:
            return False

        # Update task status
        task.status = AgentStatus.CANCELLED
        task.completed_at = datetime.now().isoformat()

        # Log task cancellation
        self.logger.info(f"Task {task_id} cancelled")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent to a dictionary.

        Returns:
            Dictionary representation of the agent.
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create an agent from a dictionary.

        This is a factory method that should be implemented by subclasses.

        Args:
            data: Dictionary containing agent data.

        Returns:
            An Agent instance.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Agent.from_dict must be implemented by subclasses")


class AgentRegistry:
    """Registry for all available agent types."""

    _agents: Dict[str, type] = {}

    @classmethod
    def register(cls, agent_class: type) -> type:
        """Register an agent class.

        Can be used as a decorator:

        @AgentRegistry.register
        class MyAgent(Agent):
            ...

        Args:
            agent_class: Agent class to register.

        Returns:
            The registered agent class.
        """
        cls._agents[agent_class.__name__] = agent_class
        return agent_class

    @classmethod
    def get_agent_class(cls, agent_type: str) -> Optional[type]:
        """Get an agent class by type.

        Args:
            agent_type: Type of the agent to get.

        Returns:
            The agent class if found, None otherwise.
        """
        return cls._agents.get(agent_type)

    @classmethod
    def create_agent(cls, agent_type: str, **kwargs) -> Optional[Agent]:
        """Create an agent instance by type.

        Args:
            agent_type: Type of the agent to create.
            **kwargs: Additional arguments to pass to the agent constructor.

        Returns:
            The created agent if the type exists, None otherwise.
        """
        agent_class = cls.get_agent_class(agent_type)
        if agent_class is None:
            return None

        return agent_class(**kwargs)

    @classmethod
    def available_agents(cls) -> List[str]:
        """Get a list of all available agent types.

        Returns:
            List of agent types.
        """
        return list(cls._agents.keys())
