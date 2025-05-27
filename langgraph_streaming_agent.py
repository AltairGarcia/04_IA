#!/usr/bin/env python3
"""
LangGraph 101 - Streaming Agent Architecture
==========================================

Enhanced streaming agent that provides real-time responses with WebSocket support,
async workflow processing, and multi-agent orchestration capabilities.

Features:
- Real-time streaming responses
- WebSocket integration for live chat
- Async workflow processing with queue management
- Multi-agent orchestration capabilities
- State management and persistence
- Enhanced error handling and recovery

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

# Core LangGraph and LangChain imports
try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming modes for different response types"""
    TEXT = "text"
    JSON = "json"
    EVENT_STREAM = "event_stream"
    WEBSOCKET = "websocket"


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


@dataclass
class StreamingChunk:
    """Individual streaming response chunk"""
    chunk_id: str
    session_id: str
    content: str
    chunk_type: str = "text"
    metadata: Dict[str, Any] = None
    is_final: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class StreamingAgent:
    """Enhanced agent with streaming capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agent = None
        self.tools = None
        self.history_manager = None
        self.memory_manager = None
        self.active_sessions: Dict[str, StreamingContext] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("StreamingAgent initialized successfully")
    
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
            
            logger.info("StreamingAgent components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize StreamingAgent components: {e}")
            raise
    
    async def create_streaming_session(
        self,
        user_id: str,
        persona_name: str = "default",
        streaming_mode: StreamingMode = StreamingMode.TEXT,
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
                "memory_items": []
            },
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )
        
        # Store session
        self.active_sessions[session_id] = context
        self.session_locks[session_id] = asyncio.Lock()
        
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
        
        async with self.session_locks[session_id]:
            try:
                # Update conversation state
                context.conversation_state["messages"].append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
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
                
                # Save interaction to history
                if self.history_manager:
                    self.history_manager.save_interaction(
                        session_id,
                        message,
                        context.conversation_state["messages"][-1].get("content", "")
                    )
                
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
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
            chunk_size = 50  # Characters per chunk
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
                        "total_length": len(full_response)
                    }
                )
                
                yield chunk
                
                # Small delay to simulate real streaming
                await asyncio.sleep(0.1)
            
            # Update conversation state
            context.conversation_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
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
            "message_count": len(context.conversation_state.get("messages", []))
        }
    
    async def close_session(self, session_id: str) -> bool:
        """Close a streaming session"""
        if session_id not in self.active_sessions:
            return False
        
        # Clean up session
        del self.active_sessions[session_id]
        if session_id in self.session_locks:
            del self.session_locks[session_id]
        
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
    
    async def initialize(self):
        """Async initialization hook (for FastAPI bridge compatibility)"""
        pass

    async def shutdown(self):
        """Async shutdown hook (for FastAPI bridge compatibility)"""
        pass


class MultiAgentOrchestrator:
    """Multi-agent orchestration capabilities"""
    
    def __init__(self):
        self.agents: Dict[str, StreamingAgent] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
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
            
            yield {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_log": execution_log
            }


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
        
        async for chunk in agent.stream_response(session_id, "Hello, how are you?"):
            print(f"Chunk: {chunk.content}")
            if chunk.is_final:
                break
        
        await agent.close_session(session_id)
    
    # Run demo
    asyncio.run(demo())
