"""
Agent Orchestrator module for multi-agent collaboration and task routing.

This module provides the Orchestrator class, which manages agent registration, task assignment,
and agent-to-agent communication for collaborative workflows.
"""

from typing import Dict, Any, Optional, List
from agents.base_agent import Agent, TaskConfig, Task, ExecutionResult
import logging
from database import get_database
import threading
import time

class Orchestrator:
    """Manages multiple agents and orchestrates collaborative tasks."""
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.logger = logging.getLogger("orchestrator")

    def register_agent(self, agent: Agent):
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def assign_task(self, agent_id: str, task_config: TaskConfig, description: Optional[str] = None) -> Optional[Task]:
        agent = self.get_agent(agent_id)
        if not agent:
            self.logger.error(f"Agent {agent_id} not found.")
            return None
        return agent.submit_task(task_config, description)

    def run_task(self, agent_id: str, task_id: str) -> Optional[ExecutionResult]:
        agent = self.get_agent(agent_id)
        if not agent:
            self.logger.error(f"Agent {agent_id} not found.")
            return None
        return agent.run_task(task_id)

    def route_task_between_agents(self, from_agent_id: str, to_agent_id: str, task_config: TaskConfig, description: Optional[str] = None) -> Optional[Task]:
        """Route a task or result from one agent to another for collaboration."""
        from_agent = self.get_agent(from_agent_id)
        to_agent = self.get_agent(to_agent_id)
        if not from_agent or not to_agent:
            self.logger.error(f"Invalid agent(s) for routing: {from_agent_id}, {to_agent_id}")
            return None
        # Optionally, log the interaction here
        return to_agent.submit_task(task_config, description)

    def log_interaction(self, from_agent_id: str, to_agent_id: str, task_id: str, action: str, details: Optional[str] = None):
        """Log an agent-to-agent interaction (for DB or audit trail)."""
        self.logger.info(f"Interaction: {from_agent_id} -> {to_agent_id}, Task: {task_id}, Action: {action}, Details: {details}")
        db = get_database()
        db.log_agent_interaction(from_agent_id, to_agent_id, task_id, action, details)

    def list_agents(self) -> List[Agent]:
        return list(self.agents.values())

    def collaborative_workflow(self, agent_sequence: List[str], initial_task_config: TaskConfig, descriptions: Optional[List[str]] = None) -> List[ExecutionResult]:
        """Run a collaborative workflow where multiple agents work in sequence, passing results along."""
        results = []
        current_task_config = initial_task_config
        for idx, agent_id in enumerate(agent_sequence):
            agent = self.get_agent(agent_id)
            if not agent:
                self.logger.error(f"Agent {agent_id} not found in workflow.")
                break
            desc = descriptions[idx] if descriptions and idx < len(descriptions) else None
            task = agent.submit_task(current_task_config, desc)
            result = agent.run_task(task.task_id)
            results.append(result)
            # Prepare next task config if needed (e.g., pass result as input)
            if idx + 1 < len(agent_sequence):
                current_task_config = TaskConfig(
                    task_type=current_task_config.task_type,
                    parameters={**current_task_config.parameters, "input_from_previous": result.output}
                )
            # Log the collaboration
            if idx > 0:
                self.log_interaction(agent_sequence[idx-1], agent_id, task.task_id, "collaboration", f"Step {idx}")
        return results

def schedule_workflow(orchestrator, agent_sequence, initial_task_config, interval_seconds=3600, descriptions=None, stop_event=None):
    """Run a collaborative workflow on a schedule (in a background thread)."""
    def run():
        while not (stop_event and stop_event.is_set()):
            orchestrator.collaborative_workflow(agent_sequence, initial_task_config, descriptions)
            time.sleep(interval_seconds)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t
