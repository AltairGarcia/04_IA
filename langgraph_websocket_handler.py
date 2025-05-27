#!/usr/bin/env python3
"""
LangGraph 101 - WebSocket Handler
================================

WebSocket handler for real-time streaming chat experiences with LangGraph agents.
Provides bidirectional communication, connection management, and error handling.

Features:
- Real-time bidirectional communication
- Connection lifecycle management
- Message routing and processing
- Error handling and recovery
- Session management integration
- Broadcasting capabilities

Author: GitHub Copilot
Date: 2024
"""

import os
import sys
import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
import traceback

# WebSocket imports
try:
    from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
    from fastapi.websockets import WebSocketState
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Import streaming agent
from langgraph_streaming_agent import (
    get_streaming_agent,
    get_multi_agent_orchestrator,
    StreamingMode,
    StreamingChunk
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket connection information"""
    connection_id: str
    websocket: WebSocket
    user_id: str
    session_id: Optional[str]
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    message_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    connection_id: str


class WebSocketConnectionManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.streaming_agent = get_streaming_agent()
        
        # Register default message handlers
        self._register_default_handlers()
        
        logger.info("WebSocketConnectionManager initialized")
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        
        @self.register_handler("ping")
        async def handle_ping(connection: WebSocketConnection, message: WebSocketMessage):
            """Handle ping messages"""
            await self.send_to_connection(
                connection.connection_id,
                "pong",
                {"timestamp": datetime.utcnow().isoformat()}
            )
        
        @self.register_handler("chat_message")
        async def handle_chat_message(connection: WebSocketConnection, message: WebSocketMessage):
            """Handle chat messages"""
            try:
                payload = message.payload
                text = payload.get("text", "")
                persona = payload.get("persona", "default")
                
                # Create session if needed
                if not connection.session_id:
                    connection.session_id = await self.streaming_agent.create_streaming_session(
                        user_id=connection.user_id,
                        persona_name=persona,
                        streaming_mode=StreamingMode.WEBSOCKET
                    )
                    
                    # Update session connections
                    if connection.session_id not in self.session_connections:
                        self.session_connections[connection.session_id] = set()
                    self.session_connections[connection.session_id].add(connection.connection_id)
                
                # Send typing indicator
                await self.send_to_connection(
                    connection.connection_id,
                    "typing_start",
                    {"session_id": connection.session_id}
                )
                
                # Stream response
                async for chunk in self.streaming_agent.stream_response(connection.session_id, text):
                    await self.send_to_connection(
                        connection.connection_id,
                        "chat_chunk",
                        {
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "chunk_type": chunk.chunk_type,
                            "is_final": chunk.is_final,
                            "session_id": connection.session_id,
                            "metadata": chunk.metadata
                        }
                    )
                
                # Send typing stop
                await self.send_to_connection(
                    connection.connection_id,
                    "typing_stop",
                    {"session_id": connection.session_id}
                )
                
            except Exception as e:
                logger.error(f"Error handling chat message: {e}")
                await self.send_to_connection(
                    connection.connection_id,
                    "error",
                    {"error": str(e), "message_id": message.message_id}
                )
        
        @self.register_handler("get_session_state")
        async def handle_get_session_state(connection: WebSocketConnection, message: WebSocketMessage):
            """Handle session state requests"""
            if connection.session_id:
                state = await self.streaming_agent.get_session_state(connection.session_id)
                await self.send_to_connection(
                    connection.connection_id,
                    "session_state",
                    {"state": state, "request_id": message.payload.get("request_id")}
                )
            else:
                await self.send_to_connection(
                    connection.connection_id,
                    "error",
                    {"error": "No active session", "message_id": message.message_id}
                )
        
        @self.register_handler("workflow_execute")
        async def handle_workflow_execute(connection: WebSocketConnection, message: WebSocketMessage):
            """Handle workflow execution requests"""
            try:
                payload = message.payload
                workflow_id = payload.get("workflow_id")
                initial_message = payload.get("message", "")
                
                orchestrator = get_multi_agent_orchestrator()
                
                async for result in orchestrator.execute_workflow(
                    workflow_id, initial_message, connection.user_id
                ):
                    await self.send_to_connection(
                        connection.connection_id,
                        "workflow_result",
                        result
                    )
                    
            except Exception as e:
                logger.error(f"Error executing workflow: {e}")
                await self.send_to_connection(
                    connection.connection_id,
                    "error",
                    {"error": str(e), "message_id": message.message_id}
                )
    
    def register_handler(self, message_type: str):
        """Decorator to register message handlers"""
        def decorator(func: Callable):
            self.message_handlers[message_type] = func
            return func
        return decorator
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Accept and register a new WebSocket connection"""
        
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            session_id=None,
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store connection
        self.active_connections[connection_id] = connection
        
        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
        
        # Send welcome message
        await self.send_to_connection(
            connection_id,
            "connected",
            {
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat(),
                "server_info": "LangGraph 101 WebSocket Server"
            }
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Disconnect and clean up a WebSocket connection"""
        
        if connection_id not in self.active_connections:
            return
        
        connection = self.active_connections[connection_id]
        
        # Close streaming session if exists
        if connection.session_id:
            await self.streaming_agent.close_session(connection.session_id)
            
            # Clean up session connections
            if connection.session_id in self.session_connections:
                self.session_connections[connection.session_id].discard(connection_id)
                if not self.session_connections[connection.session_id]:
                    del self.session_connections[connection.session_id]
        
        # Clean up user connections
        if connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Remove connection
        del self.active_connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_connection(
        self,
        connection_id: str,
        message_type: str,
        payload: Dict[str, Any]
    ):
        """Send message to a specific connection"""
        
        if connection_id not in self.active_connections:
            logger.warning(f"Connection {connection_id} not found")
            return
        
        connection = self.active_connections[connection_id]
        
        message = {
            "message_id": str(uuid.uuid4()),
            "message_type": message_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            connection.last_activity = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            # Connection might be broken, schedule for cleanup
            await self.disconnect(connection_id)
    
    async def send_to_user(
        self,
        user_id: str,
        message_type: str,
        payload: Dict[str, Any]
    ):
        """Send message to all connections of a user"""
        
        if user_id not in self.user_connections:
            return
        
        connection_ids = list(self.user_connections[user_id])
        for connection_id in connection_ids:
            await self.send_to_connection(connection_id, message_type, payload)
    
    async def send_to_session(
        self,
        session_id: str,
        message_type: str,
        payload: Dict[str, Any]
    ):
        """Send message to all connections in a session"""
        
        if session_id not in self.session_connections:
            return
        
        connection_ids = list(self.session_connections[session_id])
        for connection_id in connection_ids:
            await self.send_to_connection(connection_id, message_type, payload)
    
    async def broadcast(
        self,
        message_type: str,
        payload: Dict[str, Any],
        exclude_connections: Optional[Set[str]] = None
    ):
        """Broadcast message to all active connections"""
        
        exclude_connections = exclude_connections or set()
        
        for connection_id in list(self.active_connections.keys()):
            if connection_id not in exclude_connections:
                await self.send_to_connection(connection_id, message_type, payload)
    
    async def handle_message(self, connection_id: str, message_data: str):
        """Handle incoming WebSocket message"""
        
        if connection_id not in self.active_connections:
            logger.warning(f"Received message from unknown connection: {connection_id}")
            return
        
        connection = self.active_connections[connection_id]
        connection.last_activity = datetime.utcnow()
        
        try:
            # Parse message
            data = json.loads(message_data)
            message = WebSocketMessage(
                message_id=data.get("message_id", str(uuid.uuid4())),
                message_type=data.get("message_type", "unknown"),
                payload=data.get("payload", {}),
                timestamp=datetime.utcnow(),
                connection_id=connection_id
            )
            
            # Handle message
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                await handler(connection, message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                await self.send_to_connection(
                    connection_id,
                    "error",
                    {"error": f"Unknown message type: {message.message_type}"}
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {connection_id}: {e}")
            await self.send_to_connection(
                connection_id,
                "error",
                {"error": "Invalid JSON format"}
            )
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self.send_to_connection(
                connection_id,
                "error",
                {"error": str(e)}
            )
    
    async def cleanup_inactive_connections(self, max_idle_minutes: int = 30):
        """Clean up inactive connections"""
        
        current_time = datetime.utcnow()
        inactive_connections = []
        
        for connection_id, connection in self.active_connections.items():
            idle_time = current_time - connection.last_activity
            if idle_time.total_seconds() > (max_idle_minutes * 60):
                inactive_connections.append(connection_id)
        
        for connection_id in inactive_connections:
            logger.info(f"Cleaning up inactive connection: {connection_id}")
            await self.disconnect(connection_id)
        
        if inactive_connections:
            logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "unique_users": len(self.user_connections),
            "active_sessions": len(self.session_connections),
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }


class WebSocketHandler:
    """FastAPI WebSocket handler for LangGraph streaming"""
    
    def __init__(self):
        self.connection_manager = WebSocketConnectionManager()
        self.cleanup_task = None
        
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await self.connection_manager.cleanup_inactive_connections()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def handle_websocket(
        self,
        websocket: WebSocket,
        user_id: str = "anonymous",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Main WebSocket handler"""
        
        connection_id = None
        
        try:
            # Connect
            connection_id = await self.connection_manager.connect(
                websocket, user_id, metadata
            )
            
            # Start cleanup task if not running
            await self.start_cleanup_task()
            
            # Message loop
            while True:
                try:
                    # Receive message
                    message_data = await websocket.receive_text()
                    
                    # Handle message
                    await self.connection_manager.handle_message(connection_id, message_data)
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected: {connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket message loop: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
        
        finally:
            # Clean up connection
            if connection_id:
                await self.connection_manager.disconnect(connection_id)


# Global handler instance
_websocket_handler = None


def get_websocket_handler() -> WebSocketHandler:
    """Get or create the global WebSocket handler instance"""
    global _websocket_handler
    if _websocket_handler is None:
        _websocket_handler = WebSocketHandler()
    return _websocket_handler


# Export main classes and functions
__all__ = [
    "WebSocketHandler",
    "WebSocketConnectionManager",
    "WebSocketConnection",
    "WebSocketMessage",
    "get_websocket_handler"
]


if __name__ == "__main__":
    # Example usage with FastAPI
    from fastapi import FastAPI
    
    app = FastAPI()
    handler = get_websocket_handler()
    
    @app.websocket("/ws/{user_id}")
    async def websocket_endpoint(websocket: WebSocket, user_id: str):
        await handler.handle_websocket(websocket, user_id)
    
    @app.get("/ws/stats")
    async def get_websocket_stats():
        return handler.connection_manager.get_connection_stats()
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
