"""
Enhanced Analytics Logger with comprehensive metrics collection and real-time capabilities
"""

import sqlite3
import json
import datetime
import os
import uuid
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from queue import Queue
import asyncio

DATABASE_DIR = "analytics_data"
DATABASE_NAME = os.path.join(DATABASE_DIR, "analytics.db")

@dataclass
class AnalyticsEvent:
    """Structured analytics event data"""
    event_type: str
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model_id: Optional[str] = None
    prompt_template_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    response_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    
    # Error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Advanced metrics
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    feature_flags: Optional[List[str]] = None

class AnalyticsLogger:
    """Enhanced analytics logger with real-time capabilities and comprehensive metrics"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.event_queue = Queue()
        self.real_time_listeners: List[Callable[[AnalyticsEvent], None]] = []
        self.batch_size = 100
        self.flush_interval = 5  # seconds
        self._worker_thread = None
        self._stop_event = threading.Event()
        
        self.initialize_database()
        self.start_background_worker()
        self._initialized = True
    
    def initialize_database(self):
        """Enhanced database initialization with comprehensive schema"""
        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR)
            
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # Enhanced events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                model_id TEXT,
                prompt_template_id TEXT,
                details_json TEXT,
                response_time_ms REAL,
                tokens_used INTEGER,
                cost_estimate REAL,
                error_code TEXT,
                error_message TEXT,
                user_agent TEXT,
                ip_address TEXT,
                feature_flags TEXT
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TEXT,
                end_time TEXT,
                total_requests INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                device_info TEXT,
                location TEXT
            )
        ''')
        
        # Model performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                date TEXT NOT NULL,
                avg_response_time REAL,
                total_requests INTEGER,
                success_rate REAL,
                avg_tokens_per_request REAL,
                total_cost REAL,
                error_count INTEGER
            )
        ''')
        
        # User behavior patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behavior (
                user_id TEXT,
                date TEXT,
                feature_usage TEXT,  -- JSON of feature usage counts
                session_duration REAL,
                requests_per_session REAL,
                preferred_models TEXT,  -- JSON array
                PRIMARY KEY (user_id, date)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_model_id ON events(model_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)')
        
        conn.commit()
        conn.close()
        print(f"Enhanced database initialized at {DATABASE_NAME}")
    
    def start_background_worker(self):
        """Start background worker for batch processing events"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
    
    def _worker_loop(self):
        """Background worker loop for processing events"""
        events_batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Collect events for batching
                while len(events_batch) < self.batch_size and not self._stop_event.is_set():
                    try:
                        event = self.event_queue.get(timeout=1.0)
                        events_batch.append(event)
                        self.event_queue.task_done()
                    except:
                        break
                
                # Flush if batch is full or enough time has passed
                current_time = time.time()
                if events_batch and (len(events_batch) >= self.batch_size or 
                                   current_time - last_flush >= self.flush_interval):
                    self._flush_events_batch(events_batch)
                    events_batch.clear()
                    last_flush = current_time
                    
            except Exception as e:
                print(f"Error in analytics worker loop: {e}")
                time.sleep(1)
        
        # Flush remaining events on shutdown
        if events_batch:
            self._flush_events_batch(events_batch)
    
    def _flush_events_batch(self, events: List[AnalyticsEvent]):
        """Flush a batch of events to the database"""
        if not events:
            return
            
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        try:
            # Prepare batch insert data
            event_data = []
            for event in events:
                event_data.append((
                    event.event_type,
                    event.timestamp,
                    event.user_id,
                    event.session_id,
                    event.model_id,
                    event.prompt_template_id,
                    json.dumps(event.details) if event.details else None,
                    event.response_time_ms,
                    event.tokens_used,
                    event.cost_estimate,
                    event.error_code,
                    event.error_message,
                    event.user_agent,
                    event.ip_address,
                    json.dumps(event.feature_flags) if event.feature_flags else None
                ))
            
            # Batch insert
            cursor.executemany('''
                INSERT INTO events (
                    event_type, timestamp, user_id, session_id, model_id, 
                    prompt_template_id, details_json, response_time_ms, 
                    tokens_used, cost_estimate, error_code, error_message,
                    user_agent, ip_address, feature_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', event_data)
            
            conn.commit()
            print(f"Flushed {len(events)} analytics events to database")
            
        except sqlite3.Error as e:
            print(f"Error flushing events batch: {e}")
        finally:
            conn.close()
    
    def log_event(self, event_type: str, **kwargs) -> str:
        """
        Log an analytics event with enhanced capabilities
        
        Returns:
            str: Event ID for tracking
        """
        event_id = str(uuid.uuid4())
        
        event = AnalyticsEvent(
            event_type=event_type,
            timestamp=datetime.datetime.now().isoformat(),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            model_id=kwargs.get('model_id'),
            prompt_template_id=kwargs.get('prompt_template_id'),
            details=kwargs.get('details'),
            response_time_ms=kwargs.get('response_time_ms'),
            tokens_used=kwargs.get('tokens_used'),
            cost_estimate=kwargs.get('cost_estimate'),
            error_code=kwargs.get('error_code'),
            error_message=kwargs.get('error_message'),
            user_agent=kwargs.get('user_agent'),
            ip_address=kwargs.get('ip_address'),
            feature_flags=kwargs.get('feature_flags')
        )
        
        # Add to queue for batch processing
        self.event_queue.put(event)
        
        # Notify real-time listeners
        for listener in self.real_time_listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"Error in real-time listener: {e}")
        
        return event_id
    
    def add_real_time_listener(self, listener: Callable[[AnalyticsEvent], None]):
        """Add a real-time event listener"""
        self.real_time_listeners.append(listener)
    
    def remove_real_time_listener(self, listener: Callable[[AnalyticsEvent], None]):
        """Remove a real-time event listener"""
        if listener in self.real_time_listeners:
            self.real_time_listeners.remove(listener)
    
    def get_events(self, 
                   event_type: Optional[str] = None,
                   user_id: Optional[str] = None,
                   model_id: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve events with filtering capabilities"""
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
            
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor.execute(query, params)
            events = [dict(row) for row in cursor.fetchall()]
            return events
        except sqlite3.Error as e:
            print(f"Error retrieving events: {e}")
            return []
        finally:
            conn.close()
    
    def get_performance_metrics(self, 
                              model_id: Optional[str] = None,
                              days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for analysis"""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        conn = sqlite3.connect(DATABASE_NAME)
        # conn.row_factory = sqlite3.Row # Ensure this is set for dict results
        cursor = conn.cursor()
        
        try:
            # Base query conditions
            where_clause = "WHERE timestamp >= ?"
            params: List[Any] = [start_date.isoformat()]
            
            if model_id:
                where_clause += " AND model_id = ?"
                params.append(model_id)
            
            # Get basic metrics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(tokens_used) as total_tokens,
                    SUM(cost_estimate) as total_cost,
                    COUNT(CASE WHEN error_code IS NOT NULL THEN 1 END) as error_count
                FROM events {where_clause}
            """, params)
            
            # metrics = dict(cursor.fetchone()) if cursor.fetchone() else {} # Corrected fetch
            row = cursor.fetchone()
            metrics = dict(zip([col[0] for col in cursor.description], row)) if row else {}
            
            # Get hourly breakdown
            cursor.execute(f"""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    COUNT(*) as requests,
                    AVG(response_time_ms) as avg_response_time
                FROM events {where_clause}
                GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
                ORDER BY hour
            """, params)
            
            # hourly_data = [dict(row) for row in cursor.fetchall()] # Corrected fetch
            hourly_data = [dict(zip([col[0] for col in cursor.description], rows)) for rows in cursor.fetchall()]
            metrics['hourly_breakdown'] = hourly_data
            
            return metrics
            
        except sqlite3.Error as e:
            print(f"Error getting performance metrics: {e}")
            return {}
        finally:
            conn.close()
    
    def get_api_calls_in_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Retrieve API call events within a specified date range."""
        return self.get_events(event_type='api_call', start_time=start_date, end_time=end_date, limit=10000)

    def get_user_interactions_in_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Retrieve user interaction events within a specified date range."""
        return self.get_events(event_type='user_interaction', start_time=start_date, end_time=end_date, limit=10000)

    def get_performance_metrics_in_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Retrieve performance metric events within a specified date range."""
        # This might need a more specific implementation if "performance_metric" is a distinct event_type
        # or if it needs to aggregate data like get_performance_metrics.
        # For now, assuming it's fetching events tagged as performance related.
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM events WHERE event_type = 'performance_metric' AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?"
        params = [start_date, end_date, 10000]
        
        try:
            cursor.execute(query, params)
            events = [dict(row) for row in cursor.fetchall()]
            return events
        except sqlite3.Error as e:
            print(f"Error retrieving performance_metric events: {e}")
            return []
        finally:
            conn.close()

    def get_errors_in_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Retrieve error events within a specified date range."""
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM events WHERE error_code IS NOT NULL AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?"
        params = [start_date, end_date, 10000]
        
        try:
            cursor.execute(query, params)
            events = [dict(row) for row in cursor.fetchall()]
            return events
        except sqlite3.Error as e:
            print(f"Error retrieving error events: {e}")
            return []
        finally:
            conn.close()
            
    def shutdown(self):
        """Gracefully shutdown the analytics logger"""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        
        # Flush any remaining events
        remaining_events = []
        while not self.event_queue.empty():
            try:
                remaining_events.append(self.event_queue.get_nowait())
            except:
                break
        
        if remaining_events:
            self._flush_events_batch(remaining_events)
        
        print("Analytics logger shutdown complete")

# Global instance
_analytics_logger = None

def get_analytics_logger() -> AnalyticsLogger:
    """Get the global analytics logger instance"""
    global _analytics_logger
    if _analytics_logger is None:
        _analytics_logger = AnalyticsLogger()
    return _analytics_logger

# Convenience functions for backward compatibility
def log_event(event_type: str, **kwargs) -> str:
    """Log an event using the global analytics logger"""
    return get_analytics_logger().log_event(event_type, **kwargs)

def initialize_database():
    """Initialize the database using the global analytics logger"""
    get_analytics_logger().initialize_database()
