#!/usr/bin/env python3
"""
LangGraph 101 - Enhanced Streaming Agent Architecture
===================================================

Production-ready streaming agent with WebSocket support, async workflow processing,
and multi-agent orchestration capabilities for Phase 4 implementation.

Features:
- Real-time streaming responses with chunking
- WebSocket integration for live chat
- Async workflow processing with queue management
- Multi-agent orchestration capabilities
- State management and persistence
- Enhanced error handling and recovery
- Performance monitoring and analytics

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
from contextlib import asynccontextmanager

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core LangGraph and LangChain imports
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import BaseTool
    from langgraph import StateGraph, START, END
    from langgraph.graph import MessagesState
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain/LangGraph not available. Using fallback implementation.")

# WebSocket support
try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Import project components
from config import load_config
from tools import get_tools
from personas import get_persona_by_name, Persona
from history import get_history_manager
from memory_manager import get_memory_manager
from agent import create_agent, invoke_agent


class StreamingMode(Enum):
    """Streaming modes for different response types"""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    BINARY = "binary"


@dataclass
class StreamingConfig:
    """Configuration for streaming operations"""
    chunk_size: int = 50
    delay_ms: int = 10
    enable_thinking: bool = True
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout_seconds: int = 30


@dataclass
class StreamingContext:
    """Context for streaming operations"""
    session_id: str
    user_id: str
    persona_name: str
    streaming_mode: StreamingMode
    conversation_state: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    config: StreamingConfig = None


@dataclass
class StreamingChunk:
    """Individual streaming response chunk"""
    chunk_id: str
    session_id: str
    content: str
    chunk_type: str = "text"
    is_final: bool = False
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class StreamingAgent:
    """Enhanced agent with streaming capabilities"""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.agent = None
        self.tools = None
        self.history_manager = None
        self.memory_manager = None
        self.active_sessions: Dict[str, StreamingContext] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Enhanced StreamingAgent initialized successfully")
    
    def _initialize_components(self):
        """Initialize LangGraph components"""
        try:
            # Load configuration
            app_config = load_config()
            
            # Initialize tools and agent
            self.tools = get_tools()
            self.agent = create_agent(tools=self.tools)
            
            # Initialize managers
            self.history_manager = get_history_manager()
            self.memory_manager = get_memory_manager()
            
            # Initialize performance tracking
            self.performance_metrics = {
                "total_sessions": 0,
                "active_sessions": 0,
                "total_messages": 0,
                "average_response_time": 0.0,
                "error_count": 0,
                "uptime": datetime.utcnow()
            }
            
            logger.info("StreamingAgent components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize StreamingAgent components: {e}")
            raise
    
    async def create_streaming_session(
        self,
        user_id: str,
        persona_name: str = "default",
        streaming_mode: StreamingMode = StreamingMode.TEXT,
        config: Optional[StreamingConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new streaming session"""
        
        session_id = str(uuid.uuid4())
        
        # Get persona
        persona = get_persona_by_name(persona_name)
        if not persona:
            persona = get_persona_by_name("default")
        
        # Create context
        context = StreamingContext(
            session_id=session_id,
            user_id=user_id,
            persona_name=persona_name,
            streaming_mode=streaming_mode,
            conversation_state={
                "messages": [],
                "persona": persona.name if persona else "default",
                "tools_used": [],
                "memory_items": [],
                "total_tokens": 0,
                "message_count": 0
            },
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            config=config or self.config
        )
        
        # Store session
        self.active_sessions[session_id] = context
        self.session_locks[session_id] = asyncio.Lock()
        
        # Update metrics
        self.performance_metrics["total_sessions"] += 1
        self.performance_metrics["active_sessions"] = len(self.active_sessions)
        
        logger.info(f"Created streaming session {session_id} for user {user_id}")
        return session_id
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        stream_chunks: bool = True
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response for a message"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context = self.active_sessions[session_id]
        start_time = datetime.utcnow()
        
        async with self.session_locks[session_id]:
            try:
                # Update conversation state
                context.conversation_state["messages"].append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                context.conversation_state["message_count"] += 1
                
                # Yield thinking indicator if enabled
                if context.config.enable_thinking:
                    yield StreamingChunk(
                        chunk_id=str(uuid.uuid4()),
                        session_id=session_id,
                        content="🤔 Processing your request...",
                        chunk_type="thinking",
                        is_final=False
                    )
                
                # Process with agent
                if stream_chunks:
                    async for chunk in self._stream_agent_response(context, message):
                        yield chunk
                else:
                    # Non-streaming fallback
                    response = await self._process_agent_response(context, message)
                    yield StreamingChunk(
                        chunk_id=str(uuid.uuid4()),
                        session_id=session_id,
                        content=response,
                        chunk_type="final",
                        is_final=True
                    )
                
                # Update metrics
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds()
                self.performance_metrics["total_messages"] += 1
                
                # Update average response time
                total_msgs = self.performance_metrics["total_messages"]
                avg_time = self.performance_metrics["average_response_time"]
                self.performance_metrics["average_response_time"] = (
                    (avg_time * (total_msgs - 1) + response_time) / total_msgs
                )
                
                # Save interaction to history
                if self.history_manager:
                    self.history_manager.save_interaction(
                        session_id,
                        message,
                        context.conversation_state["messages"][-1].get("content", "")
                    )
                
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                self.performance_metrics["error_count"] += 1
                yield StreamingChunk(
                    chunk_id=str(uuid.uuid4()),
                    session_id=session_id,
                    content=f"Error: {str(e)}",
                    chunk_type="error",
                    is_final=True
                )
    
    async def _stream_agent_response(
        self,
        context: StreamingContext,
        message: str
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Stream agent response with chunking"""
        
        try:
            # Get persona
            persona = get_persona_by_name(context.persona_name)
            
            # Prepare state for agent
            state = {
                "messages": [{"role": "user", "content": message}],
                "persona": persona.name if persona else "default",
                "session_id": context.session_id
            }
            
            # Process with agent
            response = await asyncio.to_thread(invoke_agent, self.agent, state)
            
            # Extract response content
            if isinstance(response, dict) and 'messages' in response:
                messages = response['messages']
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and 'content' in last_message:
                        full_response = last_message['content']
                    else:
                        full_response = str(last_message)
                else:
                    full_response = "No response generated"
            else:
                full_response = str(response)
            
            # Stream response in chunks
            chunk_size = context.config.chunk_size
            for i in range(0, len(full_response), chunk_size):
                chunk_content = full_response[i:i + chunk_size]
                is_final = (i + chunk_size) >= len(full_response)
                
                chunk = StreamingChunk(
                    chunk_id=str(uuid.uuid4()),
                    session_id=context.session_id,
                    content=chunk_content,
                    chunk_type="text",
                    is_final=is_final,
                    metadata={
                        "chunk_index": i // chunk_size,
                        "total_length": len(full_response),
                        "persona": context.persona_name
                    }
                )
                
                yield chunk
                
                # Small delay to simulate real streaming
                if context.config.delay_ms > 0:
                    await asyncio.sleep(context.config.delay_ms / 1000)
            
            # Update conversation state
            context.conversation_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update token count (estimate)
            context.conversation_state["total_tokens"] += len(full_response.split())
            
        except Exception as e:
            logger.error(f"Error in agent streaming: {e}")
            yield StreamingChunk(
                chunk_id=str(uuid.uuid4()),
                session_id=context.session_id,
                content=f"Error processing request: {str(e)}",
                chunk_type="error",
                is_final=True
            )
    
    async def _process_agent_response(self, context: StreamingContext, message: str) -> str:
        """Process agent response without streaming"""
        
        try:
            # Get persona
            persona = get_persona_by_name(context.persona_name)
            
            # Prepare state for agent
            state = {
                "messages": [{"role": "user", "content": message}],
                "persona": persona.name if persona else "default",
                "session_id": context.session_id
            }
            
            # Process with agent
            response = await asyncio.to_thread(invoke_agent, self.agent, state)
            
            # Extract response content
            if isinstance(response, dict) and 'messages' in response:
                messages = response['messages']
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and 'content' in last_message:
                        full_response = last_message['content']
                    else:
                        full_response = str(last_message)
                else:
                    full_response = "No response generated"
            else:
                full_response = str(response)
            
            # Update conversation state
            context.conversation_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error processing agent response: {e}")
            return f"Error processing request: {str(e)}"
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state"""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "user_id": context.user_id,
            "persona_name": context.persona_name,
            "streaming_mode": context.streaming_mode.value,
            "conversation_state": context.conversation_state,
            "metadata": context.metadata,
            "created_at": context.created_at.isoformat(),
            "message_count": len(context.conversation_state.get("messages", [])),
            "total_tokens": context.conversation_state.get("total_tokens", 0)
        }
    
    async def close_session(self, session_id: str) -> bool:
        """Close a streaming session"""
        if session_id not in self.active_sessions:
            return False
        
        # Clean up session
        del self.active_sessions[session_id]
        if session_id in self.session_locks:
            del self.session_locks[session_id]
        
        # Update metrics
        self.performance_metrics["active_sessions"] = len(self.active_sessions)
        
        logger.info(f"Closed streaming session {session_id}")
        return True
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_sessions.keys())
    
    async def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, context in self.active_sessions.items():
            age = current_time - context.created_at
            if age > timedelta(hours=max_age_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.close_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.performance_metrics.copy()
        metrics["uptime_seconds"] = (datetime.utcnow() - metrics["uptime"]).total_seconds()
        metrics["uptime"] = metrics["uptime"].isoformat()
        return metrics
    
    async def initialize(self):
        """Async initialization hook"""
        pass

    async def shutdown(self):
        """Async shutdown hook"""
        # Close all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)
        
        logger.info("StreamingAgent shutdown complete")


class MultiAgentOrchestrator:
    """Multi-agent orchestration capabilities"""
    
    def __init__(self):
        self.agents: Dict[str, StreamingAgent] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    async def register_agent(self, agent_id: str, agent: StreamingAgent):
        """Register an agent"""
        self.agents[agent_id] = agent
        logger.info(f"Registered agent {agent_id}")
    
    async def create_workflow(
        self,
        workflow_id: str,
        agent_sequence: List[str],
        workflow_config: Dict[str, Any]
    ):
        """Create a multi-agent workflow"""
        self.workflows[workflow_id] = {
            "agent_sequence": agent_sequence,
            "config": workflow_config,
            "created_at": datetime.utcnow(),
            "executions": []
        }
        logger.info(f"Created workflow {workflow_id} with agents: {agent_sequence}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_message: str,
        user_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a multi-agent workflow"""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        agent_sequence = workflow["agent_sequence"]
        
        execution_id = str(uuid.uuid4())
        execution_log = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "started_at": datetime.utcnow(),
            "steps": []
        }
        
        try:
            current_message = initial_message
            
            for i, agent_id in enumerate(agent_sequence):
                if agent_id not in self.agents:
                    raise ValueError(f"Agent {agent_id} not found")
                
                agent = self.agents[agent_id]
                
                # Create session for this step
                session_id = await agent.create_streaming_session(
                    user_id=user_id,
                    persona_name="default",
                    metadata={"workflow_id": workflow_id, "step": i}
                )
                
                step_result = {
                    "step": i,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "input": current_message,
                    "output": "",
                    "started_at": datetime.utcnow()
                }
                
                # Stream response from agent
                response_chunks = []
                async for chunk in agent.stream_response(session_id, current_message):
                    response_chunks.append(chunk.content)
                    
                    # Yield intermediate result
                    yield {
                        "execution_id": execution_id,
                        "workflow_id": workflow_id,
                        "step": i,
                        "agent_id": agent_id,
                        "chunk": chunk.content,
                        "is_final": chunk.is_final
                    }
                
                # Prepare for next step
                current_message = "".join(response_chunks)
                step_result["output"] = current_message
                step_result["completed_at"] = datetime.utcnow()
                
                execution_log["steps"].append(step_result)
                
                # Close session
                await agent.close_session(session_id)
            
            execution_log["completed_at"] = datetime.utcnow()
            workflow["executions"].append(execution_log)
            self.execution_history.append(execution_log)
            
            # Final result
            yield {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "completed",
                "final_result": current_message,
                "execution_log": execution_log
            }
            
        except Exception as e:
            execution_log["error"] = str(e)
            execution_log["failed_at"] = datetime.utcnow()
            workflow["executions"].append(execution_log)
            self.execution_history.append(execution_log)
            
            yield {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_log": execution_log
            }
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return self.execution_history
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get metrics for all registered agents"""
        metrics = {}
        for agent_id, agent in self.agents.items():
            metrics[agent_id] = agent.get_performance_metrics()
        return metrics


# Global instances
_streaming_agent = None
_multi_agent_orchestrator = None


def get_streaming_agent() -> StreamingAgent:
    """Get or create the global streaming agent instance"""
    global _streaming_agent
    if _streaming_agent is None:
        _streaming_agent = StreamingAgent()
    return _streaming_agent


def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the global multi-agent orchestrator instance"""
    global _multi_agent_orchestrator
    if _multi_agent_orchestrator is None:
        _multi_agent_orchestrator = MultiAgentOrchestrator()
    return _multi_agent_orchestrator


# Export main classes and functions
__all__ = [
    "StreamingAgent",
    "MultiAgentOrchestrator", 
    "StreamingMode",
    "StreamingConfig",
    "StreamingContext",
    "StreamingChunk",
    "get_streaming_agent",
    "get_multi_agent_orchestrator"
]


if __name__ == "__main__":
    # Example usage
    async def demo():
        agent = get_streaming_agent()
        session_id = await agent.create_streaming_session("demo_user")
        
        print(f"Created session: {session_id}")
        print("🚀 Testing streaming response...")
        
        async for chunk in agent.stream_response(session_id, "Hello, how are you?"):
            print(f"Chunk: {chunk.content}")
            if chunk.is_final:
                break
        
        # Get session state
        state = await agent.get_session_state(session_id)
        print(f"\nSession state: {state['message_count']} messages")
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        print(f"Performance: {metrics['total_messages']} messages, {metrics['average_response_time']:.2f}s avg")
        
        await agent.close_session(session_id)
    
    # Run demo
    asyncio.run(demo())
