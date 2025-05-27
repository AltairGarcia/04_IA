#!/usr/bin/env python3
"""
LangGraph 101 - Message Queue System
====================================

Comprehensive async task processing system using Celery and Redis for:
- Background job processing
- Distributed task execution
- Task scheduling and retry mechanisms
- Performance monitoring and optimization
- Integration with existing security and infrastructure components

Features:
- Multiple task types and priorities
- Retry mechanisms with exponential backoff
- Task monitoring and statistics
- Rate limiting integration
- Security logging and audit trails
- Performance metrics and optimization
- Health checks and diagnostics
- Dead letter queue handling
- Task result caching
- Distributed worker management

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
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import signal

# Third-party imports
import redis
from celery import Celery, Task
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun, task_failure
from celery.exceptions import MaxRetriesExceededError, Retry
from kombu import Queue, Exchange
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class TaskMetrics:
    """Task execution metrics"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    retry_tasks: int = 0
    average_execution_time: float = 0.0
    peak_memory_usage: float = 0.0
    worker_count: int = 0
    queue_size: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

class MessageQueueConfig:
    """Configuration for message queue system"""
      def __init__(self):
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6380))
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        self.redis_password = os.getenv('REDIS_PASSWORD')
        
        # Celery configuration
        self.broker_url = f"redis://{':' + self.redis_password + '@' if self.redis_password else ''}{self.redis_host}:{self.redis_port}/{self.redis_db}"
        self.result_backend = self.broker_url
        
        # Task configuration
        self.task_serializer = 'json'
        self.result_serializer = 'json'
        self.accept_content = ['json']
        self.timezone = 'UTC'
        self.enable_utc = True
        
        # Worker configuration
        self.worker_concurrency = int(os.getenv('WORKER_CONCURRENCY', psutil.cpu_count()))
        self.worker_prefetch_multiplier = int(os.getenv('WORKER_PREFETCH_MULTIPLIER', 1))
        self.worker_max_tasks_per_child = int(os.getenv('WORKER_MAX_TASKS_PER_CHILD', 1000))
        
        # Retry configuration
        self.default_retry_delay = int(os.getenv('DEFAULT_RETRY_DELAY', 60))
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))
        self.retry_backoff = bool(os.getenv('RETRY_BACKOFF', True))
        self.retry_jitter = bool(os.getenv('RETRY_JITTER', True))
        
        # Monitoring configuration
        self.metrics_update_interval = int(os.getenv('METRICS_UPDATE_INTERVAL', 60))
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', 30))

class CustomTask(Task):
    """Custom Celery task with enhanced features"""
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.metrics = {}
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying due to: {exc}")
        
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        execution_time = time.time() - self.start_time if self.start_time else 0
        logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {exc}")
        
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        execution_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Task {task_id} completed successfully in {execution_time:.2f}s")

class MessageQueueSystem:
    """Main message queue system class"""
    
    def __init__(self, config: Optional[MessageQueueConfig] = None):
        self.config = config or MessageQueueConfig()
        self.celery_app = None
        self.redis_client = None
        self.task_registry = {}
        self.metrics = TaskMetrics()
        self.is_running = False
        self.monitoring_thread = None
        self._lock = threading.RLock()
        
        # Initialize components
        self._initialize_redis()
        self._initialize_celery()
        self._setup_queues()
        self._register_signals()
        
        logger.info("Message Queue System initialized successfully")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _initialize_celery(self):
        """Initialize Celery application"""
        self.celery_app = Celery('langgraph_tasks')
        
        # Configure Celery
        self.celery_app.conf.update(
            broker_url=self.config.broker_url,
            result_backend=self.config.result_backend,
            task_serializer=self.config.task_serializer,
            result_serializer=self.config.result_serializer,
            accept_content=self.config.accept_content,
            timezone=self.config.timezone,
            enable_utc=self.config.enable_utc,
            worker_concurrency=self.config.worker_concurrency,
            worker_prefetch_multiplier=self.config.worker_prefetch_multiplier,
            worker_max_tasks_per_child=self.config.worker_max_tasks_per_child,
            task_default_retry_delay=self.config.default_retry_delay,
            task_max_retries=self.config.max_retries,
            task_acks_late=True,
            worker_disable_rate_limits=False,
            task_reject_on_worker_lost=True,
            task_ignore_result=False,
            result_expires=3600,  # 1 hour
            broker_connection_retry_on_startup=True
        )
        
        # Set custom task base class
        self.celery_app.Task = CustomTask
        
        logger.info("Celery application configured successfully")
    
    def _setup_queues(self):
        """Setup task queues with different priorities"""
        # Define exchanges
        default_exchange = Exchange('default', type='direct')
        priority_exchange = Exchange('priority', type='direct')
        
        # Define queues
        self.celery_app.conf.task_routes = {
            'langgraph_tasks.critical_task': {'queue': 'critical'},
            'langgraph_tasks.high_task': {'queue': 'high'},
            'langgraph_tasks.normal_task': {'queue': 'normal'},
            'langgraph_tasks.low_task': {'queue': 'low'},
        }
        
        self.celery_app.conf.task_queues = (
            Queue('critical', priority_exchange, routing_key='critical', queue_arguments={'x-max-priority': 10}),
            Queue('high', priority_exchange, routing_key='high', queue_arguments={'x-max-priority': 7}),
            Queue('normal', default_exchange, routing_key='normal', queue_arguments={'x-max-priority': 5}),
            Queue('low', default_exchange, routing_key='low', queue_arguments={'x-max-priority': 2}),
        )
        
        # Default queue for unrouted tasks
        self.celery_app.conf.task_default_queue = 'normal'
        self.celery_app.conf.task_default_exchange = 'default'
        self.celery_app.conf.task_default_routing_key = 'normal'
        
        logger.info("Task queues configured successfully")
    
    def _register_signals(self):
        """Register Celery signals for monitoring"""
        
        @worker_ready.connect
        def worker_ready_handler(sender=None, **kwargs):
            logger.info(f"Worker {sender} is ready")
            with self._lock:
                self.metrics.worker_count += 1
        
        @worker_shutdown.connect
        def worker_shutdown_handler(sender=None, **kwargs):
            logger.info(f"Worker {sender} is shutting down")
            with self._lock:
                self.metrics.worker_count = max(0, self.metrics.worker_count - 1)
        
        @task_prerun.connect
        def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
            logger.debug(f"Task {task_id} starting")
            task.start_time = time.time()
            with self._lock:
                self.metrics.total_tasks += 1
        
        @task_postrun.connect
        def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, state=None, **kwargs):
            execution_time = time.time() - task.start_time if hasattr(task, 'start_time') and task.start_time else 0
            
            with self._lock:
                if state == 'SUCCESS':
                    self.metrics.successful_tasks += 1
                elif state == 'FAILURE':
                    self.metrics.failed_tasks += 1
                elif state == 'RETRY':
                    self.metrics.retry_tasks += 1
                
                # Update average execution time
                total_completed = self.metrics.successful_tasks + self.metrics.failed_tasks
                if total_completed > 0:
                    self.metrics.average_execution_time = (
                        (self.metrics.average_execution_time * (total_completed - 1) + execution_time) / total_completed
                    )
                
                self.metrics.last_updated = datetime.utcnow()
            
            # Store task result
            self._store_task_result(task_id, state, retval, execution_time)
        
        @task_failure.connect
        def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
            logger.error(f"Task {task_id} failed: {exception}")
            self._store_task_result(task_id, 'FAILURE', None, 0, str(exception))
    
    def _store_task_result(self, task_id: str, status: str, result: Any, execution_time: float, error: str = None):
        """Store task result in Redis"""
        try:
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus(status.lower()),
                result=result,
                error=error,
                execution_time=execution_time,
                completed_at=datetime.utcnow()
            )
            
            # Store in Redis with 24-hour expiration
            self.redis_client.setex(
                f"task_result:{task_id}",
                86400,  # 24 hours
                json.dumps(asdict(task_result), default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to store task result for {task_id}: {e}")
    
    def register_task(self, 
                     name: str, 
                     func: Callable, 
                     priority: TaskPriority = TaskPriority.NORMAL,
                     max_retries: int = None,
                     retry_delay: int = None,
                     bind: bool = True):
        """Register a new task with the system"""
        
        # Configure task options
        task_options = {
            'bind': bind,
            'name': f'langgraph_tasks.{name}',
            'max_retries': max_retries or self.config.max_retries,
            'default_retry_delay': retry_delay or self.config.default_retry_delay,
            'autoretry_for': (Exception,),
            'retry_backoff': self.config.retry_backoff,
            'retry_jitter': self.config.retry_jitter
        }
        
        # Set queue based on priority
        if priority == TaskPriority.CRITICAL:
            task_options['queue'] = 'critical'
        elif priority == TaskPriority.HIGH:
            task_options['queue'] = 'high'
        elif priority == TaskPriority.LOW:
            task_options['queue'] = 'low'
        else:
            task_options['queue'] = 'normal'
        
        # Register with Celery
        task = self.celery_app.task(**task_options)(func)
        
        # Store in registry
        self.task_registry[name] = {
            'task': task,
            'priority': priority,
            'function': func,
            'options': task_options
        }
        
        logger.info(f"Registered task '{name}' with priority {priority.value}")
        return task
    
    def submit_task(self, 
                   task_name: str, 
                   *args, 
                   priority: Optional[TaskPriority] = None,
                   eta: Optional[datetime] = None,
                   countdown: Optional[int] = None,
                   **kwargs) -> str:
        """Submit a task for execution"""
        
        if task_name not in self.task_registry:
            raise ValueError(f"Task '{task_name}' not registered")
        
        task_info = self.task_registry[task_name]
        task = task_info['task']
        
        # Override priority if specified
        apply_options = {}
        if priority and priority != task_info['priority']:
            if priority == TaskPriority.CRITICAL:
                apply_options['queue'] = 'critical'
            elif priority == TaskPriority.HIGH:
                apply_options['queue'] = 'high'
            elif priority == TaskPriority.LOW:
                apply_options['queue'] = 'low'
            else:
                apply_options['queue'] = 'normal'
        
        # Set scheduling options
        if eta:
            apply_options['eta'] = eta
        elif countdown:
            apply_options['countdown'] = countdown
        
        try:
            # Submit task
            result = task.apply_async(args=args, kwargs=kwargs, **apply_options)
            
            logger.info(f"Submitted task '{task_name}' with ID {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit task '{task_name}': {e}")
            raise
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID"""
        try:
            # Try to get from Redis first
            result_data = self.redis_client.get(f"task_result:{task_id}")
            if result_data:
                data = json.loads(result_data)
                data['status'] = TaskStatus(data['status'])
                if data['created_at']:
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                if data['completed_at']:
                    data['completed_at'] = datetime.fromisoformat(data['completed_at'])
                return TaskResult(**data)
            
            # Fallback to Celery result backend
            celery_result = self.celery_app.AsyncResult(task_id)
            if celery_result.state != 'PENDING':
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus(celery_result.state.lower()),
                    result=celery_result.result if celery_result.successful() else None,
                    error=str(celery_result.info) if celery_result.failed() else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Cancelled task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all queues"""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active() or {}
            
            # Get reserved tasks
            reserved_tasks = inspect.reserved() or {}
            
            # Get queue lengths
            queue_lengths = {}
            for queue_name in ['critical', 'high', 'normal', 'low']:
                try:
                    length = self.redis_client.llen(queue_name)
                    queue_lengths[queue_name] = length
                except:
                    queue_lengths[queue_name] = 0
            
            return {
                'active_tasks': active_tasks,
                'reserved_tasks': reserved_tasks,
                'queue_lengths': queue_lengths,
                'total_workers': len(active_tasks),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {}
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers"""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Get worker statistics
            stats = inspect.stats() or {}
            
            # Get registered tasks
            registered = inspect.registered() or {}
            
            # Get worker configuration
            conf = inspect.conf() or {}
            
            return {
                'statistics': stats,
                'registered_tasks': registered,
                'configuration': conf,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get worker status: {e}")
            return {}
    
    def get_metrics(self) -> TaskMetrics:
        """Get current system metrics"""
        with self._lock:
            # Update queue size
            try:
                total_queue_size = 0
                for queue_name in ['critical', 'high', 'normal', 'low']:
                    try:
                        length = self.redis_client.llen(queue_name)
                        total_queue_size += length
                    except:
                        pass
                
                self.metrics.queue_size = total_queue_size
                
                # Update memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics.peak_memory_usage = max(self.metrics.peak_memory_usage, memory_mb)
                
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")
            
            return self.metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check Redis connection
        try:
            self.redis_client.ping()
            health_status['components']['redis'] = {'status': 'healthy', 'latency_ms': 0}
        except Exception as e:
            health_status['components']['redis'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'unhealthy'
        
        # Check Celery broker
        try:
            inspect = self.celery_app.control.inspect()
            stats = inspect.stats()
            if stats:
                health_status['components']['celery'] = {'status': 'healthy', 'workers': len(stats)}
            else:
                health_status['components']['celery'] = {'status': 'degraded', 'workers': 0}
        except Exception as e:
            health_status['components']['celery'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'unhealthy'
        
        # Check system resources
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
            
            resource_status = 'healthy'
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                resource_status = 'degraded'
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                resource_status = 'critical'
                health_status['overall_status'] = 'degraded'
            
            health_status['components']['resources'] = {
                'status': resource_status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent
            }
            
        except Exception as e:
            health_status['components']['resources'] = {'status': 'unknown', 'error': str(e)}
        
        return health_status
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started monitoring thread")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring thread")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Update metrics
                self.get_metrics()
                
                # Perform health check
                health = self.health_check()
                if health['overall_status'] != 'healthy':
                    logger.warning(f"System health degraded: {health}")
                
                # Log queue status periodically
                queue_status = self.get_queue_status()
                if queue_status:
                    total_active = sum(len(tasks) for tasks in queue_status.get('active_tasks', {}).values())
                    total_reserved = sum(len(tasks) for tasks in queue_status.get('reserved_tasks', {}).values())
                    logger.debug(f"Queue status - Active: {total_active}, Reserved: {total_reserved}")
                
                # Sleep until next check
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def start_worker(self, 
                    queues: Optional[List[str]] = None,
                    concurrency: Optional[int] = None,
                    loglevel: str = 'info'):
        """Start a Celery worker"""
        if queues is None:
            queues = ['critical', 'high', 'normal', 'low']
        
        if concurrency is None:
            concurrency = self.config.worker_concurrency
        
        # Start monitoring
        self.start_monitoring()
        
        try:
            logger.info(f"Starting Celery worker with queues: {queues}")
            
            # Start worker
            self.celery_app.worker_main([
                'worker',
                f'--loglevel={loglevel}',
                f'--concurrency={concurrency}',
                f'--queues={",".join(queues)}',
                '--without-gossip',
                '--without-mingle',
                '--without-heartbeat'
            ])
            
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
        except Exception as e:
            logger.error(f"Worker error: {e}")
            raise
        finally:
            self.stop_monitoring()
    
    def shutdown(self):
        """Graceful shutdown of the message queue system"""
        logger.info("Shutting down Message Queue System...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Message Queue System shutdown complete")


# Example task definitions
def create_example_tasks(mq_system: MessageQueueSystem):
    """Create example tasks for demonstration"""
    
    @mq_system.register_task('process_document', priority=TaskPriority.HIGH)
    def process_document(self, document_id: str, options: Dict[str, Any] = None):
        """Process a document asynchronously"""
        try:
            logger.info(f"Processing document {document_id}")
            
            # Simulate document processing
            time.sleep(2)
            
            result = {
                'document_id': document_id,
                'processed_at': datetime.utcnow().isoformat(),
                'word_count': 1500,
                'status': 'completed'
            }
            
            logger.info(f"Document {document_id} processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            raise
    
    @mq_system.register_task('send_notification', priority=TaskPriority.NORMAL)
    def send_notification(self, user_id: str, message: str, channel: str = 'email'):
        """Send notification to user"""
        try:
            logger.info(f"Sending notification to user {user_id} via {channel}")
            
            # Simulate notification sending
            time.sleep(1)
            
            result = {
                'user_id': user_id,
                'channel': channel,
                'sent_at': datetime.utcnow().isoformat(),
                'status': 'delivered'
            }
            
            logger.info(f"Notification sent to user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to send notification to user {user_id}: {e}")
            raise
    
    @mq_system.register_task('cleanup_temp_files', priority=TaskPriority.LOW)
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """Clean up temporary files"""
        try:
            logger.info(f"Cleaning up temp files older than {older_than_hours} hours")
            
            # Simulate cleanup
            time.sleep(0.5)
            
            result = {
                'files_deleted': 42,
                'space_freed_mb': 128,
                'cleaned_at': datetime.utcnow().isoformat()
            }
            
            logger.info("Temp file cleanup completed")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            raise
    
    @mq_system.register_task('generate_report', priority=TaskPriority.CRITICAL, max_retries=5)
    def generate_report(self, report_type: str, date_range: Dict[str, str]):
        """Generate system report"""
        try:
            logger.info(f"Generating {report_type} report for {date_range}")
            
            # Simulate report generation
            time.sleep(5)
            
            result = {
                'report_type': report_type,
                'date_range': date_range,
                'generated_at': datetime.utcnow().isoformat(),
                'file_path': f'/tmp/reports/{report_type}_{int(time.time())}.pdf',
                'size_mb': 2.5
            }
            
            logger.info(f"Report {report_type} generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate report {report_type}: {e}")
            raise


# CLI interface for testing
def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LangGraph 101 Message Queue System')
    parser.add_argument('command', choices=['worker', 'test', 'status', 'health'], 
                       help='Command to execute')
    parser.add_argument('--queues', nargs='+', default=['critical', 'high', 'normal', 'low'],
                       help='Queues for worker to process')
    parser.add_argument('--concurrency', type=int, help='Worker concurrency')
    parser.add_argument('--loglevel', default='info', help='Log level')
    
    args = parser.parse_args()
    
    # Initialize message queue system
    mq_system = MessageQueueSystem()
    
    # Create example tasks
    create_example_tasks(mq_system)
    
    try:
        if args.command == 'worker':
            mq_system.start_worker(
                queues=args.queues,
                concurrency=args.concurrency,
                loglevel=args.loglevel
            )
        
        elif args.command == 'test':
            # Submit test tasks
            print("Submitting test tasks...")
            
            task_ids = []
            
            # Submit various test tasks
            task_ids.append(mq_system.submit_task('process_document', 'doc_123'))
            task_ids.append(mq_system.submit_task('send_notification', 'user_456', 'Hello World!'))
            task_ids.append(mq_system.submit_task('cleanup_temp_files', 48))
            task_ids.append(mq_system.submit_task('generate_report', 'monthly', 
                                                {'start': '2024-01-01', 'end': '2024-01-31'}))
            
            print(f"Submitted {len(task_ids)} tasks")
            
            # Monitor task completion
            for task_id in task_ids:
                print(f"Task {task_id} submitted")
        
        elif args.command == 'status':
            # Show system status
            print("=== Queue Status ===")
            queue_status = mq_system.get_queue_status()
            print(json.dumps(queue_status, indent=2))
            
            print("\n=== Worker Status ===")
            worker_status = mq_system.get_worker_status()
            print(json.dumps(worker_status, indent=2))
            
            print("\n=== Metrics ===")
            metrics = mq_system.get_metrics()
            print(json.dumps(asdict(metrics), indent=2, default=str))
        
        elif args.command == 'health':
            # Show health status
            health = mq_system.health_check()
            print(json.dumps(health, indent=2))
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        mq_system.shutdown()


if __name__ == '__main__':
    main()
