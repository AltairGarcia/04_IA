"""
Agents package for the Don Corleone AI project.

This package contains various autonomous agents that can perform tasks
without constant user supervision.
"""

from .base_agent import (
    Agent, TaskConfig, Task, ExecutionResult, AgentStatus, AgentRegistry
)
from .research_agent import ResearchAgent
from .document_agent import DocumentAgent

# Export all available agent types
__all__ = [
    'Agent', 'TaskConfig', 'Task', 'ExecutionResult', 'AgentStatus', 'AgentRegistry',
    'ResearchAgent', 'DocumentAgent'
]

# Create a registry of available agents
AVAILABLE_AGENTS = {
    "research": ResearchAgent,
    "document": DocumentAgent
}

def get_agent_types() -> dict:
    """Get a dictionary of available agent types.

    Returns:
        Dictionary mapping agent types to their descriptions.
    """
    return {
        "research": "Conducts web research and compiles information",
        "document": "Analyzes and processes documents"
    }

def create_agent(agent_type: str, **kwargs) -> Agent:
    """Create an agent of the specified type.

    Args:
        agent_type: Type of agent to create.
        **kwargs: Additional arguments for the agent constructor.

    Returns:
        An Agent instance.

    Raises:
        ValueError: If the agent type is not supported.
    """
    if agent_type not in AVAILABLE_AGENTS:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    return AVAILABLE_AGENTS[agent_type](**kwargs)
