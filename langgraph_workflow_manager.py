#!/usr/bin/env python3
"""
LangGraph 101 - Workflow Manager
==============================

Advanced workflow management system for orchestrating multi-agent collaborations,
complex task processing, and async workflow execution with state management.

Features:
- Multi-agent workflow orchestration
- Complex task routing and processing
- State management and persistence
- Workflow templates and reusability
- Real-time monitoring and analytics
- Error handling and recovery

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import json
import uuid
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import traceback
from contextlib import asynccontextmanager

# Import project components
from langgraph_streaming_agent import (
    get_streaming_agent,
    get_multi_agent_orchestrator,
    StreamingAgent,
    StreamingMode
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Workflow step status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    agent_id: str
    step_type: str
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    condition: Optional[str] = None  # Conditional execution


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    user_id: str
    status: WorkflowStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    current_step: int
    step_results: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTemplate:
    """Workflow template definition"""
    template_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    global_config: Dict[str, Any]
    required_agents: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    created_at: datetime
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """Core workflow execution engine"""
    
    def __init__(self):
        self.streaming_agent = get_streaming_agent()
        self.orchestrator = get_multi_agent_orchestrator()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_locks: Dict[str, asyncio.Lock] = {}
        self.step_handlers: Dict[str, Callable] = {}
        
        # Register default step handlers
        self._register_default_handlers()
        
        logger.info("WorkflowEngine initialized")
    
    def _register_default_handlers(self):
        """Register default step handlers"""
        
        @self.register_step_handler("chat")
        async def handle_chat_step(
            step: WorkflowStep,
            execution: WorkflowExecution,
            context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Handle chat interaction step"""
            
            message = step.config.get("message", "")
            persona = step.config.get("persona", "default")
            
            # Create or get session
            session_id = context.get("session_id")
            if not session_id:
                session_id = await self.streaming_agent.create_streaming_session(
                    user_id=execution.user_id,
                    persona_name=persona,
                    metadata={"workflow_id": execution.workflow_id, "step_id": step.step_id}
                )
                context["session_id"] = session_id
            
            # Collect response chunks
            response_chunks = []
            async for chunk in self.streaming_agent.stream_response(session_id, message):
                response_chunks.append(chunk.content)
            
            response = "".join(response_chunks)
            
            return {
                "response": response,
                "session_id": session_id,
                "persona": persona,
                "message": message
            }
        
        @self.register_step_handler("content_creation")
        async def handle_content_creation_step(
            step: WorkflowStep,
            execution: WorkflowExecution,
            context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Handle content creation step"""
            
            topic = step.config.get("topic", "")
            content_type = step.config.get("content_type", "blog")
            options = step.config.get("options", {})
            
            # Use content creation agent
            session_id = await self.streaming_agent.create_streaming_session(
                user_id=execution.user_id,
                persona_name="content_creator",
                metadata={"workflow_id": execution.workflow_id, "step_id": step.step_id}
            )
            
            message = f"Create {content_type} content about: {topic}"
            if options:
                message += f" with options: {json.dumps(options)}"
            
            response_chunks = []
            async for chunk in self.streaming_agent.stream_response(session_id, message):
                response_chunks.append(chunk.content)
            
            content = "".join(response_chunks)
            
            return {
                "content": content,
                "topic": topic,
                "content_type": content_type,
                "session_id": session_id
            }
        
        @self.register_step_handler("data_analysis")
        async def handle_data_analysis_step(
            step: WorkflowStep,
            execution: WorkflowExecution,
            context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Handle data analysis step"""
            
            data_source = step.config.get("data_source", "")
            analysis_type = step.config.get("analysis_type", "summary")
            
            # Use data analysis agent
            session_id = await self.streaming_agent.create_streaming_session(
                user_id=execution.user_id,
                persona_name="data_analyst",
                metadata={"workflow_id": execution.workflow_id, "step_id": step.step_id}
            )
            
            message = f"Perform {analysis_type} analysis on: {data_source}"
            
            response_chunks = []
            async for chunk in self.streaming_agent.stream_response(session_id, message):
                response_chunks.append(chunk.content)
            
            analysis = "".join(response_chunks)
            
            return {
                "analysis": analysis,
                "data_source": data_source,
                "analysis_type": analysis_type,
                "session_id": session_id
            }
        
        @self.register_step_handler("conditional")
        async def handle_conditional_step(
            step: WorkflowStep,
            execution: WorkflowExecution,
            context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Handle conditional logic step"""
            
            condition = step.config.get("condition", "true")
            true_action = step.config.get("true_action", {})
            false_action = step.config.get("false_action", {})
            
            # Evaluate condition (simple string evaluation for demo)
            # In production, use a safe expression evaluator
            try:
                result = eval(condition, {"__builtins__": {}}, context)
                action = true_action if result else false_action
                
                return {
                    "condition": condition,
                    "condition_result": result,
                    "action_taken": action,
                    "context": context
                }
            except Exception as e:
                logger.error(f"Error evaluating condition '{condition}': {e}")
                return {
                    "condition": condition,
                    "condition_result": False,
                    "error": str(e),
                    "action_taken": false_action
                }
    
    def register_step_handler(self, step_type: str):
        """Decorator to register step handlers"""
        def decorator(func: Callable):
            self.step_handlers[step_type] = func
            return func
        return decorator
    
    async def execute_workflow(
        self,
        workflow: WorkflowTemplate,
        input_data: Dict[str, Any],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a workflow"""
        
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.template_id,
            user_id=user_id,
            status=WorkflowStatus.PENDING,
            input_data=input_data,
            output_data={},
            current_step=0,
            step_results={},
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store execution
        self.active_executions[execution_id] = execution
        self.execution_locks[execution_id] = asyncio.Lock()
        
        try:
            async with self.execution_locks[execution_id]:
                # Start execution
                execution.status = WorkflowStatus.RUNNING
                execution.started_at = datetime.utcnow()
                
                # Yield initial status
                yield {
                    "execution_id": execution_id,
                    "status": "started",
                    "workflow_id": workflow.template_id,
                    "timestamp": execution.started_at.isoformat()
                }
                
                # Execute steps
                context = {**input_data, "execution_id": execution_id}
                
                for i, step in enumerate(workflow.steps):
                    execution.current_step = i
                    
                    # Check dependencies
                    if not await self._check_dependencies(step, execution):
                        logger.warning(f"Dependencies not met for step {step.step_id}")
                        continue
                    
                    # Check condition
                    if step.condition and not await self._evaluate_condition(step.condition, context):
                        logger.info(f"Condition not met for step {step.step_id}")
                        execution.step_results[step.step_id] = {
                            "status": StepStatus.SKIPPED.value,
                            "reason": "condition_not_met"
                        }
                        continue
                    
                    # Execute step
                    yield {
                        "execution_id": execution_id,
                        "status": "step_started",
                        "step_id": step.step_id,
                        "step_index": i,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    step_result = await self._execute_step(step, execution, context)
                    execution.step_results[step.step_id] = step_result
                    
                    # Update context with step result
                    context.update(step_result)
                    
                    yield {
                        "execution_id": execution_id,
                        "status": "step_completed",
                        "step_id": step.step_id,
                        "step_index": i,
                        "result": step_result,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Complete execution
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.output_data = context
                
                yield {
                    "execution_id": execution_id,
                    "status": "completed",
                    "workflow_id": workflow.template_id,
                    "output_data": execution.output_data,
                    "step_results": execution.step_results,
                    "duration": (execution.completed_at - execution.started_at).total_seconds(),
                    "timestamp": execution.completed_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error executing workflow {workflow.template_id}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            
            yield {
                "execution_id": execution_id,
                "status": "failed",
                "workflow_id": workflow.template_id,
                "error": str(e),
                "step_results": execution.step_results,
                "timestamp": execution.completed_at.isoformat()
            }
        
        finally:
            # Clean up
            if execution_id in self.execution_locks:
                del self.execution_locks[execution_id]
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        start_time = datetime.utcnow()
        
        try:
            if step.step_type not in self.step_handlers:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            handler = self.step_handlers[step.step_type]
            
            # Execute with timeout
            result = await asyncio.wait_for(
                handler(step, execution, context),
                timeout=step.timeout_seconds
            )
            
            end_time = datetime.utcnow()
            
            return {
                "status": StepStatus.COMPLETED.value,
                "result": result,
                "started_at": start_time.isoformat(),
                "completed_at": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds()
            }
            
        except asyncio.TimeoutError:
            return {
                "status": StepStatus.FAILED.value,
                "error": f"Step timeout after {step.timeout_seconds} seconds",
                "started_at": start_time.isoformat(),
                "failed_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": StepStatus.FAILED.value,
                "error": str(e),
                "started_at": start_time.isoformat(),
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def _check_dependencies(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> bool:
        """Check if step dependencies are satisfied"""
        
        for dep_step_id in step.dependencies:
            if dep_step_id not in execution.step_results:
                return False
            
            dep_result = execution.step_results[dep_step_id]
            if dep_result.get("status") != StepStatus.COMPLETED.value:
                return False
        
        return True
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate step condition"""
        try:
            # Simple evaluation - in production, use a safe expression evaluator
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        if execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        return asdict(execution)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            return False
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        return True


class WorkflowManager:
    """High-level workflow management system"""
    
    def __init__(self):
        self.engine = WorkflowEngine()
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Load default templates
        self._load_default_templates()
        
        logger.info("WorkflowManager initialized")
    
    def _load_default_templates(self):
        """Load default workflow templates"""
        
        # Content creation workflow
        content_workflow = WorkflowTemplate(
            template_id="content_creation_workflow",
            name="Content Creation Workflow",
            description="Multi-step content creation with research and optimization",
            version="1.0",
            steps=[
                WorkflowStep(
                    step_id="research",
                    agent_id="research_agent",
                    step_type="chat",
                    config={
                        "message": "Research the topic: {topic}",
                        "persona": "researcher"
                    }
                ),
                WorkflowStep(
                    step_id="content_creation",
                    agent_id="content_agent",
                    step_type="content_creation",
                    config={
                        "topic": "{topic}",
                        "content_type": "{content_type}",
                        "options": {"include_research": True}
                    },
                    dependencies=["research"]
                ),
                WorkflowStep(
                    step_id="review",
                    agent_id="review_agent",
                    step_type="chat",
                    config={
                        "message": "Review and improve this content: {content}",
                        "persona": "editor"
                    },
                    dependencies=["content_creation"]
                )
            ],
            global_config={},
            required_agents=["research_agent", "content_agent", "review_agent"],
            input_schema={"topic": "string", "content_type": "string"},
            output_schema={"final_content": "string"},
            created_at=datetime.utcnow(),
            tags=["content", "creation", "multi-agent"]
        )
        
        self.register_template(content_workflow)
        
        # Data analysis workflow
        analysis_workflow = WorkflowTemplate(
            template_id="data_analysis_workflow",
            name="Data Analysis Workflow",
            description="Comprehensive data analysis and reporting",
            version="1.0",
            steps=[
                WorkflowStep(
                    step_id="data_collection",
                    agent_id="data_agent",
                    step_type="data_analysis",
                    config={
                        "data_source": "{data_source}",
                        "analysis_type": "collection"
                    }
                ),
                WorkflowStep(
                    step_id="analysis",
                    agent_id="analysis_agent",
                    step_type="data_analysis",
                    config={
                        "data_source": "{data_source}",
                        "analysis_type": "statistical"
                    },
                    dependencies=["data_collection"]
                ),
                WorkflowStep(
                    step_id="visualization",
                    agent_id="viz_agent",
                    step_type="chat",
                    config={
                        "message": "Create visualizations for this analysis: {analysis}",
                        "persona": "data_visualizer"
                    },
                    dependencies=["analysis"]
                ),
                WorkflowStep(
                    step_id="report",
                    agent_id="report_agent",
                    step_type="content_creation",
                    config={
                        "topic": "Data Analysis Report",
                        "content_type": "report",
                        "options": {"include_analysis": True, "include_visualizations": True}
                    },
                    dependencies=["analysis", "visualization"]
                )
            ],
            global_config={},
            required_agents=["data_agent", "analysis_agent", "viz_agent", "report_agent"],
            input_schema={"data_source": "string"},
            output_schema={"report": "string", "analysis": "object"},
            created_at=datetime.utcnow(),
            tags=["data", "analysis", "reporting"]
        )
        
        self.register_template(analysis_workflow)
    
    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template"""
        self.templates[template.template_id] = template
        logger.info(f"Registered workflow template: {template.template_id}")
    
    async def execute_template(
        self,
        template_id: str,
        input_data: Dict[str, Any],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a workflow template"""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        
        async for result in self.engine.execute_workflow(template, input_data, user_id, metadata):
            yield result
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get all available templates"""
        return [
            {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "version": template.version,
                "required_agents": template.required_agents,
                "input_schema": template.input_schema,
                "output_schema": template.output_schema,
                "tags": template.tags
            }
            for template in self.templates.values()
        ]
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific template"""
        return self.templates.get(template_id)
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        return await self.engine.get_execution_status(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution"""
        return await self.engine.cancel_execution(execution_id)


# Global manager instance
_workflow_manager = None


def get_workflow_manager() -> WorkflowManager:
    """Get or create the global workflow manager instance"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager


# Export main classes and functions
__all__ = [
    "WorkflowManager",
    "WorkflowEngine",
    "WorkflowTemplate",
    "WorkflowExecution",
    "WorkflowStep",
    "WorkflowStatus",
    "StepStatus",
    "get_workflow_manager"
]


if __name__ == "__main__":
    # Example usage
    async def demo():
        manager = get_workflow_manager()
        
        # List available templates
        templates = manager.get_templates()
        print("Available templates:")
        for template in templates:
            print(f"- {template['name']}: {template['description']}")
        
        # Execute content creation workflow
        input_data = {
            "topic": "Artificial Intelligence in Healthcare",
            "content_type": "blog"
        }
        
        print(f"\nExecuting content creation workflow...")
        async for result in manager.execute_template(
            "content_creation_workflow",
            input_data,
            "demo_user"
        ):
            print(f"Status: {result.get('status')} - {result.get('timestamp')}")
            if result.get('status') == 'completed':
                print(f"Final output: {result.get('output_data', {}).keys()}")
    
    # Run demo
    asyncio.run(demo())
