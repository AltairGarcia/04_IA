"""
Research Agent for the Don Corleone AI project.

This module implements a specialized agent for conducting web research
and compiling information autonomously.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_agent import Agent, TaskConfig, Task, ExecutionResult, AgentStatus, AgentRegistry
from tools import search_web


@AgentRegistry.register
class ResearchAgent(Agent):
    """Agent that conducts web research and compiles information."""

    SUPPORTED_TASK_TYPES = ["web_research", "fact_check", "topic_summary"]

    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        """Initialize a new ResearchAgent.

        Args:
            agent_id: Optional ID for the agent. If not provided, a UUID will be generated.
            name: Optional name for the agent.
        """
        super().__init__(agent_id=agent_id, name=name or "Research Agent")
        self.logger = logging.getLogger("agent.research")

    def can_handle_task(self, task_config: TaskConfig) -> bool:
        """Check if this agent can handle a task.

        Args:
            task_config: Configuration for the task.

        Returns:
            True if the agent can handle the task, False otherwise.
        """
        return task_config.task_type in self.SUPPORTED_TASK_TYPES

    def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a research task.

        Args:
            task: The task to execute.

        Returns:
            The result of the execution.
        """
        if not task.config:
            return ExecutionResult(
                success=False,
                output=None,
                error="Task configuration is missing"
            )

        task_type = task.config.task_type
        parameters = task.config.parameters

        if task_type == "web_research":
            return self._execute_web_research(parameters)
        elif task_type == "fact_check":
            return self._execute_fact_check(parameters)
        elif task_type == "topic_summary":
            return self._execute_topic_summary(parameters)
        else:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Unsupported task type: {task_type}"
            )

    def _execute_web_research(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a web research task.

        Args:
            parameters: Parameters for the task.

        Returns:
            The result of the execution.
        """
        query = parameters.get("query")
        if not query:
            return ExecutionResult(
                success=False,
                output=None,
                error="Query parameter is missing"
            )

        # Number of sources to gather information from
        num_sources = parameters.get("num_sources", 3)

        try:
            self.logger.info(f"Performing web research for query: {query}")

            # Initial search results
            search_results = search_web(query)

            # Process the results
            if "Erro ao buscar na web:" in search_results:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Search error: {search_results}"
                )

            compiled_info = self._compile_research_results(search_results, num_sources)

            return ExecutionResult(
                success=True,
                output=compiled_info,
                artifacts=[
                    {
                        "type": "search_results",
                        "content": search_results,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            )

        except Exception as e:
            self.logger.exception(f"Error during web research: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Error during web research: {str(e)}"
            )

    def _execute_fact_check(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a fact checking task.

        Args:
            parameters: Parameters for the task.

        Returns:
            The result of the execution.
        """
        statement = parameters.get("statement")
        if not statement:
            return ExecutionResult(
                success=False,
                output=None,
                error="Statement parameter is missing"
            )

        try:
            self.logger.info(f"Fact checking statement: {statement}")

            # Perform a search to fact check the statement
            search_results = search_web(f"fact check {statement}")

            # Process the results
            if "Erro ao buscar na web:" in search_results:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Search error: {search_results}"
                )

            # Analyze search results to determine if statement is true
            fact_check_result = self._analyze_fact_check_results(statement, search_results)

            return ExecutionResult(
                success=True,
                output=fact_check_result,
                artifacts=[
                    {
                        "type": "fact_check_results",
                        "content": search_results,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            )

        except Exception as e:
            self.logger.exception(f"Error during fact checking: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Error during fact checking: {str(e)}"
            )

    def _execute_topic_summary(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a topic summary task.

        Args:
            parameters: Parameters for the task.

        Returns:
            The result of the execution.
        """
        topic = parameters.get("topic")
        if not topic:
            return ExecutionResult(
                success=False,
                output=None,
                error="Topic parameter is missing"
            )

        # Aspects of the topic to research
        aspects = parameters.get("aspects", [])

        try:
            self.logger.info(f"Creating summary for topic: {topic}")

            # Basic topic overview
            search_results = search_web(f"{topic} overview")

            # Process the results
            if "Erro ao buscar na web:" in search_results:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Search error: {search_results}"
                )

            # Search for additional aspects if specified
            all_results = [
                {
                    "aspect": "overview",
                    "results": search_results
                }
            ]

            for aspect in aspects:
                aspect_results = search_web(f"{topic} {aspect}")
                if "Erro ao buscar na web:" not in aspect_results:
                    all_results.append({
                        "aspect": aspect,
                        "results": aspect_results
                    })

                    # Be nice to search APIs - don't flood with requests
                    time.sleep(1)

            # Compile a comprehensive summary
            topic_summary = self._compile_topic_summary(topic, all_results)

            return ExecutionResult(
                success=True,
                output=topic_summary,
                artifacts=[
                    {
                        "type": "topic_results",
                        "aspect": result["aspect"],
                        "content": result["results"],
                        "timestamp": datetime.now().isoformat()
                    }
                    for result in all_results
                ]
            )

        except Exception as e:
            self.logger.exception(f"Error during topic summary: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Error during topic summary: {str(e)}"
            )

    def _compile_research_results(self, search_results: str, num_sources: int) -> str:
        """Compile research results into a structured format.

        Args:
            search_results: Raw search results as a string.
            num_sources: Number of sources to include.

        Returns:
            Compiled research information.
        """
        # Extract and structure the search results
        sources = []
        current_source = None

        for line in search_results.split('\n'):
            if line.startswith('- ['):
                # New source title
                source_title = line.split(']: ')[0][3:]
                current_source = {
                    "title": source_title,
                    "url": line.split(']: ')[1] if ']:' in line else "",
                    "content": ""
                }
                sources.append(current_source)
            elif line.startswith('\tSummary: ') and current_source:
                # Source content
                current_source["content"] = line.replace('\tSummary: ', '')

        # Limit to the specified number of sources
        sources = sources[:num_sources]

        # Compile the information
        compiled_info = "## Research Results\n\n"

        for i, source in enumerate(sources, 1):
            compiled_info += f"### Source {i}: {source['title']}\n"
            compiled_info += f"**URL:** {source['url']}\n\n"
            compiled_info += f"{source['content']}\n\n"

        # Add a conclusion section
        compiled_info += "## Summary\n\n"
        compiled_info += "Based on the research results, here are the key findings:\n\n"

        # Extract key points from each source
        for i, source in enumerate(sources, 1):
            # Extract a short key point from the content
            content = source['content']
            if content:
                key_point = content.split('.')[0] + '.' if '.' in content else content
                compiled_info += f"- {key_point}\n"

        return compiled_info

    def _analyze_fact_check_results(self, statement: str, search_results: str) -> str:
        """Analyze fact check results to determine if a statement is true.

        Args:
            statement: The statement to fact check.
            search_results: Raw search results as a string.

        Returns:
            Fact check analysis.
        """
        # Extract and process the search results
        sources = []
        current_source = None

        for line in search_results.split('\n'):
            if line.startswith('- ['):
                # New source title
                source_title = line.split(']: ')[0][3:]
                current_source = {
                    "title": source_title,
                    "url": line.split(']: ')[1] if ']:' in line else "",
                    "content": ""
                }
                sources.append(current_source)
            elif line.startswith('\tSummary: ') and current_source:
                # Source content
                current_source["content"] = line.replace('\tSummary: ', '')

        # Analyze the results
        analysis = f"## Fact Check: {statement}\n\n"
        analysis += "### Sources Consulted\n\n"

        for i, source in enumerate(sources, 1):
            analysis += f"{i}. **{source['title']}** - {source['url']}\n"

        analysis += "\n### Analysis\n\n"

        # Look for confirmation or refutation indicators in the content
        confirmation_indicators = ["true", "confirmed", "accurate", "correct", "verify", "factual"]
        refutation_indicators = ["false", "debunked", "inaccurate", "incorrect", "misleading", "fake"]

        confirmation_count = 0
        refutation_count = 0

        for source in sources:
            content = source['content'].lower()

            for indicator in confirmation_indicators:
                if indicator in content:
                    confirmation_count += 1

            for indicator in refutation_indicators:
                if indicator in content:
                    refutation_count += 1

        # Determine the verdict based on the indicators
        if confirmation_count > refutation_count:
            verdict = "**Likely True**: Most sources confirm this statement."
        elif refutation_count > confirmation_count:
            verdict = "**Likely False**: Most sources refute this statement."
        else:
            verdict = "**Inconclusive**: The evidence is mixed or insufficient."

        analysis += f"{verdict}\n\n"
        analysis += "### Evidence\n\n"

        for i, source in enumerate(sources, 1):
            analysis += f"**Source {i}**: {source['content']}\n\n"

        return analysis

    def _compile_topic_summary(self, topic: str, results: List[Dict[str, Any]]) -> str:
        """Compile a comprehensive summary of a topic.

        Args:
            topic: The topic to summarize.
            results: List of search results for different aspects of the topic.

        Returns:
            Comprehensive topic summary.
        """
        summary = f"# {topic.title()}: Comprehensive Summary\n\n"

        # Process each aspect
        for result in results:
            aspect = result["aspect"]
            search_results = result["results"]

            summary += f"## {aspect.title()}\n\n"

            # Extract and process the search results
            sources = []
            current_source = None

            for line in search_results.split('\n'):
                if line.startswith('- ['):
                    # New source title
                    source_title = line.split(']: ')[0][3:]
                    current_source = {
                        "title": source_title,
                        "url": line.split(']: ')[1] if ']:' in line else "",
                        "content": ""
                    }
                    sources.append(current_source)
                elif line.startswith('\tSummary: ') and current_source:
                    # Source content
                    current_source["content"] = line.replace('\tSummary: ', '')

            # Compile information from all sources for this aspect
            all_content = ""
            for source in sources[:2]:  # Limit to top 2 sources per aspect
                all_content += source["content"] + " "

            # Add the compiled information to the summary
            summary += all_content + "\n\n"

            # Add source attribution
            summary += "**Sources:**\n"
            for i, source in enumerate(sources[:2], 1):
                summary += f"{i}. [{source['title']}]({source['url']})\n"

            summary += "\n"

        # Add a conclusion section
        summary += "## Key Takeaways\n\n"
        summary += f"Based on the research about {topic}, here are the main points:\n\n"

        # Placeholder for key takeaways - in a real implementation,
        # we would extract these programmatically from the content
        summary += "1. This is a comprehensive summary compiled from multiple sources.\n"
        summary += "2. Information has been organized by different aspects of the topic.\n"
        summary += "3. All sources have been properly attributed.\n"

        return summary

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchAgent':
        """Create a ResearchAgent from a dictionary.

        Args:
            data: Dictionary containing agent data.

        Returns:
            A ResearchAgent instance.
        """
        agent = cls(agent_id=data["agent_id"], name=data["name"])

        # Restore tasks
        for task_id, task_data in data.get("tasks", {}).items():
            agent.tasks[task_id] = Task.from_dict(task_data)

        return agent
