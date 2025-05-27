"""
Real-time Analytics Dashboard and Streaming Capabilities

Provides real-time analytics dashboard components and streaming data capabilities
for monitoring AI platform usage and performance in real-time.
"""

import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

from .analytics_logger import AnalyticsEvent, get_analytics_logger

@dataclass
class RealTimeMetrics:
    """Real-time metrics snapshot"""
    timestamp: str
    active_users: int
    requests_per_minute: float
    avg_response_time: float
    error_rate: float
    top_models: List[Dict[str, Any]]
    system_health: Dict[str, Any]

class RealTimeAnalytics:
    """Real-time analytics processor and dashboard data provider"""
    
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.window_size = window_minutes * 60  # Convert to seconds
        
        # Real-time data stores
        self.events_window = deque(maxlen=1000)  # Last 1000 events
        self.metrics_history = deque(maxlen=288)  # 24 hours of 5-min intervals
        self.active_sessions = {}
        self.model_stats = defaultdict(lambda: {
            'requests': 0,
            'total_response_time': 0,
            'errors': 0,
            'tokens': 0,
            'cost': 0
        })
        
        # Subscribers for real-time updates
        self.subscribers: List[Callable[[RealTimeMetrics], None]] = []
        
        # Background processing
        self._stop_event = threading.Event()
        self._processor_thread = None
        
        # Register with analytics logger
        analytics_logger = get_analytics_logger()
        analytics_logger.add_real_time_listener(self._process_event)
        
        self.start_background_processor()
    
    def start_background_processor(self):
        """Start background thread for real-time metrics calculation"""
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(
                target=self._processor_loop, 
                daemon=True
            )
            self._processor_thread.start()
    
    def _processor_loop(self):
        """Background loop for calculating and broadcasting metrics"""
        while not self._stop_event.is_set():
            try:
                # Calculate current metrics
                metrics = self._calculate_current_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Notify subscribers
                for subscriber in self.subscribers:
                    try:
                        subscriber(metrics)
                    except Exception as e:
                        print(f"Error notifying real-time subscriber: {e}")
                
                # Wait for next interval
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Error in real-time processor loop: {e}")
                time.sleep(5)
    
    def _process_event(self, event: AnalyticsEvent):
        """Process incoming analytics event for real-time tracking"""
        current_time = time.time()
        
        # Add to sliding window
        self.events_window.append({
            'event': event,
            'timestamp': current_time
        })
        
        # Clean old events from window
        cutoff_time = current_time - self.window_size
        while self.events_window and self.events_window[0]['timestamp'] < cutoff_time:
            self.events_window.popleft()
        
        # Update session tracking
        if event.session_id:
            self.active_sessions[event.session_id] = {
                'user_id': event.user_id,
                'last_activity': current_time,
                'model_id': event.model_id
            }
        
        # Update model statistics
        if event.model_id:
            stats = self.model_stats[event.model_id]
            stats['requests'] += 1
            
            if event.response_time_ms:
                stats['total_response_time'] += event.response_time_ms
            
            if event.error_code:
                stats['errors'] += 1
                
            if event.tokens_used:
                stats['tokens'] += event.tokens_used
                
            if event.cost_estimate:
                stats['cost'] += event.cost_estimate
        
        # Clean up inactive sessions (older than 30 minutes)
        inactive_cutoff = current_time - (30 * 60)
        inactive_sessions = [
            sid for sid, session in self.active_sessions.items()
            if session['last_activity'] < inactive_cutoff
        ]
        for sid in inactive_sessions:
            del self.active_sessions[sid]
    
    def _calculate_current_metrics(self) -> RealTimeMetrics:
        """Calculate current real-time metrics"""
        current_time = time.time()
        
        # Filter events in current window
        window_events = [
            item['event'] for item in self.events_window
            if current_time - item['timestamp'] <= self.window_size
        ]
        
        # Calculate basic metrics
        total_requests = len(window_events)
        requests_per_minute = total_requests / self.window_minutes if self.window_minutes > 0 else 0
        
        # Response time metrics
        response_times = [
            event.response_time_ms for event in window_events
            if event.response_time_ms is not None
        ]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Error rate
        error_count = sum(1 for event in window_events if event.error_code)
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        
        # Active users (unique sessions in last 5 minutes)
        active_users = len(self.active_sessions)
        
        # Top models by usage
        model_usage = defaultdict(int)
        for event in window_events:
            if event.model_id:
                model_usage[event.model_id] += 1
        
        top_models = [
            {
                'model_id': model_id,
                'requests': count,
                'avg_response_time': (
                    self.model_stats[model_id]['total_response_time'] / 
                    self.model_stats[model_id]['requests']
                ) if self.model_stats[model_id]['requests'] > 0 else 0
            }
            for model_id, count in sorted(
                model_usage.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        ]
        
        # System health indicators
        system_health = {
            'status': 'healthy' if error_rate < 5 else 'warning' if error_rate < 15 else 'critical',
            'avg_response_time_status': (
                'good' if avg_response_time < 1000 else 
                'warning' if avg_response_time < 3000 else 'poor'
            ),
            'load_status': (
                'low' if requests_per_minute < 10 else
                'medium' if requests_per_minute < 50 else 'high'
            )
        }
        
        return RealTimeMetrics(
            timestamp=datetime.now().isoformat(),
            active_users=active_users,
            requests_per_minute=requests_per_minute,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            top_models=top_models,
            system_health=system_health
        )
    
    def get_current_metrics(self) -> RealTimeMetrics:
        """Get current real-time metrics snapshot"""
        return self._calculate_current_metrics()
    
    def get_metrics_history(self, hours: int = 1) -> List[RealTimeMetrics]:
        """Get historical metrics for the specified number of hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics.timestamp) >= cutoff_time
        ]
    
    def subscribe_to_updates(self, callback: Callable[[RealTimeMetrics], None]):
        """Subscribe to real-time metrics updates"""
        self.subscribers.append(callback)
    
    def unsubscribe_from_updates(self, callback: Callable[[RealTimeMetrics], None]):
        """Unsubscribe from real-time metrics updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for UI display"""
        current_metrics = self.get_current_metrics()
        recent_history = self.get_metrics_history(hours=6)
        
        # Calculate trends
        if len(recent_history) >= 2:
            recent_requests = recent_history[-1].requests_per_minute
            previous_requests = recent_history[-2].requests_per_minute
            request_trend = ((recent_requests - previous_requests) / previous_requests * 100) if previous_requests > 0 else 0
            
            recent_response_time = recent_history[-1].avg_response_time
            previous_response_time = recent_history[-2].avg_response_time
            response_time_trend = ((recent_response_time - previous_response_time) / previous_response_time * 100) if previous_response_time > 0 else 0
        else:
            request_trend = 0
            response_time_trend = 0
        
        return {
            'current_metrics': asdict(current_metrics),
            'trends': {
                'requests_per_minute_change': request_trend,
                'response_time_change': response_time_trend
            },
            'history': [asdict(m) for m in recent_history],
            'active_sessions': len(self.active_sessions),
            'model_performance': {
                model_id: {
                    'total_requests': stats['requests'],
                    'avg_response_time': (
                        stats['total_response_time'] / stats['requests']
                    ) if stats['requests'] > 0 else 0,
                    'error_rate': (
                        stats['errors'] / stats['requests'] * 100
                    ) if stats['requests'] > 0 else 0,
                    'total_tokens': stats['tokens'],
                    'total_cost': stats['cost']
                }
                for model_id, stats in self.model_stats.items()
            }
        }
    
    async def stream_metrics(self, websocket_send: Callable):
        """Stream real-time metrics via WebSocket or similar"""
        def send_update(metrics: RealTimeMetrics):
            asyncio.create_task(websocket_send(json.dumps(asdict(metrics))))
        
        self.subscribe_to_updates(send_update)
        
        try:
            # Send initial data
            initial_metrics = self.get_current_metrics()
            await websocket_send(json.dumps(asdict(initial_metrics)))
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                
        except Exception as e:
            print(f"WebSocket streaming error: {e}")
        finally:
            self.unsubscribe_from_updates(send_update)
    
    def shutdown(self):
        """Shutdown real-time analytics processor"""
        self._stop_event.set()
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)
        
        print("Real-time analytics processor shutdown complete")

# Global instance
_real_time_analytics = None

def get_real_time_analytics() -> RealTimeAnalytics:
    """Get the global real-time analytics instance"""
    global _real_time_analytics
    if _real_time_analytics is None:
        _real_time_analytics = RealTimeAnalytics()
    return _real_time_analytics
