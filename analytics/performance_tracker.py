"""
Performance Tracker for LangGraph 101 AI Platform

Tracks and analyzes performance metrics across different dimensions including
model performance, system performance, and user experience metrics.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import sqlite3
import json

from .analytics_logger import get_analytics_logger

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    metric_name: str
    value: float
    timestamp: str
    dimensions: Dict[str, str]  # model_id, user_id, etc.
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceSummary:
    """Summary of performance metrics over a time period"""
    metric_name: str
    time_period: str
    avg_value: float
    min_value: float
    max_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    sample_count: int
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_percentage: float

@dataclass
class ModelPerformanceReport:
    """Comprehensive performance report for a model"""
    model_id: str
    time_period: str
    response_time_summary: PerformanceSummary
    token_efficiency_summary: PerformanceSummary
    cost_efficiency_summary: PerformanceSummary
    error_rate: float
    success_rate: float
    total_requests: int
    unique_users: int
    recommendation_score: float
    performance_issues: List[str]

class PerformanceTracker:
    """Comprehensive performance tracking and analysis system"""
    
    def __init__(self):
        self.analytics_logger = get_analytics_logger()
        
        # Real-time performance tracking
        self.real_time_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.performance_thresholds = {
            'response_time_warning': 2000,  # ms
            'response_time_critical': 5000,  # ms
            'error_rate_warning': 5,  # %
            'error_rate_critical': 15,  # %
            'token_efficiency_warning': 0.7,  # tokens per unit value
        }
        
        # Performance baselines (calculated from historical data)
        self.baselines = {}
        self._initialize_baselines()
    
    def _initialize_baselines(self):
        """Initialize performance baselines from historical data"""
        try:
            # Calculate baselines from last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            events = self.analytics_logger.get_events(
                start_time=start_date.isoformat(),
                end_time=end_date.isoformat(),
                limit=10000
            )
            
            if events:
                # Response time baseline
                response_times = [
                    event['response_time_ms'] for event in events
                    if event.get('response_time_ms') is not None
                ]
                if response_times:
                    self.baselines['response_time'] = {
                        'mean': statistics.mean(response_times),
                        'p95': self._percentile(response_times, 95)
                    }
                
                # Error rate baseline
                total_events = len(events)
                error_events = sum(1 for event in events if event.get('error_code'))
                if total_events > 0:
                    self.baselines['error_rate'] = (error_events / total_events) * 100
                
                print(f"Performance baselines initialized: {self.baselines}")
        
        except Exception as e:
            print(f"Error initializing performance baselines: {e}")
            # Set default baselines
            self.baselines = {
                'response_time': {'mean': 1500, 'p95': 3000},
                'error_rate': 2.0
            }
    
    def track_performance_metric(self, 
                                metric_name: str, 
                                value: float, 
                                dimensions: Optional[Dict[str, str]] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Track a performance metric in real-time"""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now().isoformat(),
            dimensions=dimensions or {},
            metadata=metadata
        )
        
        # Add to real-time tracking
        self.real_time_metrics[metric_name].append(metric)
        
        # Check for performance alerts
        self._check_performance_alerts(metric)
    
    def _check_performance_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any performance alerts"""
        alerts = []
        
        if metric.metric_name == 'response_time':
            if metric.value > self.performance_thresholds['response_time_critical']:
                alerts.append(f"CRITICAL: Response time {metric.value}ms exceeds threshold")
            elif metric.value > self.performance_thresholds['response_time_warning']:
                alerts.append(f"WARNING: Response time {metric.value}ms above normal")
        
        elif metric.metric_name == 'error_rate':
            if metric.value > self.performance_thresholds['error_rate_critical']:
                alerts.append(f"CRITICAL: Error rate {metric.value}% is critically high")
            elif metric.value > self.performance_thresholds['error_rate_warning']:
                alerts.append(f"WARNING: Error rate {metric.value}% above normal")
        
        # Log alerts to analytics system
        for alert in alerts:
            self.analytics_logger.log_event(
                'performance_alert',
                details={'alert': alert, 'metric': asdict(metric)}
            )
    
    def get_model_performance_report(self, 
                                   model_id: str, 
                                   days: int = 7) -> Optional[ModelPerformanceReport]:
        """Generate comprehensive performance report for a specific model"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get events for the model
        events = self.analytics_logger.get_events(
            model_id=model_id,
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            limit=10000
        )
        
        if not events:
            return None
        
        # Extract metrics
        response_times = [e['response_time_ms'] for e in events if e.get('response_time_ms')]
        token_counts = [e['tokens_used'] for e in events if e.get('tokens_used')]
        costs = [e['cost_estimate'] for e in events if e.get('cost_estimate')]
        errors = [e for e in events if e.get('error_code')]
        
        # Calculate summaries
        response_time_summary = self._calculate_metric_summary(
            'response_time', response_times, f"{days} days"
        ) if response_times else None
        
        token_efficiency_summary = self._calculate_metric_summary(
            'token_efficiency', token_counts, f"{days} days"
        ) if token_counts else None
        
        cost_efficiency_summary = self._calculate_metric_summary(
            'cost_efficiency', costs, f"{days} days"
        ) if costs else None
        
        # Calculate rates
        total_requests = len(events)
        error_rate = (len(errors) / total_requests * 100) if total_requests > 0 else 0
        success_rate = 100 - error_rate
        
        # Unique users
        unique_users = len(set(e['user_id'] for e in events if e.get('user_id')))
        
        # Calculate recommendation score (0-100)
        recommendation_score = self._calculate_model_recommendation_score(
            response_time_summary, error_rate, total_requests
        )
        
        # Identify performance issues
        performance_issues = self._identify_performance_issues(
            response_time_summary, error_rate, model_id
        )
        
        return ModelPerformanceReport(
            model_id=model_id,
            time_period=f"{days} days",
            response_time_summary=response_time_summary,
            token_efficiency_summary=token_efficiency_summary,
            cost_efficiency_summary=cost_efficiency_summary,
            error_rate=error_rate,
            success_rate=success_rate,
            total_requests=total_requests,
            unique_users=unique_users,
            recommendation_score=recommendation_score,
            performance_issues=performance_issues
        )
    
    def _calculate_metric_summary(self, 
                                metric_name: str, 
                                values: List[float], 
                                time_period: str) -> PerformanceSummary:
        """Calculate summary statistics for a metric"""
        if not values:
            return PerformanceSummary(
                metric_name=metric_name,
                time_period=time_period,
                avg_value=0, min_value=0, max_value=0,
                p50_value=0, p95_value=0, p99_value=0,
                sample_count=0,
                trend_direction='stable',
                trend_percentage=0
            )
        
        sorted_values = sorted(values)
        
        # Calculate percentiles
        p50 = self._percentile(sorted_values, 50)
        p95 = self._percentile(sorted_values, 95)
        p99 = self._percentile(sorted_values, 99)
        
        # Calculate trend (compare first half vs second half)
        mid_point = len(values) // 2
        if mid_point > 0:
            first_half_avg = statistics.mean(values[:mid_point])
            second_half_avg = statistics.mean(values[mid_point:])
            
            if first_half_avg > 0:
                trend_percentage = ((second_half_avg - first_half_avg) / first_half_avg) * 100
                
                if abs(trend_percentage) < 5:
                    trend_direction = 'stable'
                elif trend_percentage > 0:
                    trend_direction = 'degrading' if metric_name in ['response_time', 'error_rate'] else 'improving'
                else:
                    trend_direction = 'improving' if metric_name in ['response_time', 'error_rate'] else 'degrading'
            else:
                trend_direction = 'stable'
                trend_percentage = 0
        else:
            trend_direction = 'stable'
            trend_percentage = 0
        
        return PerformanceSummary(
            metric_name=metric_name,
            time_period=time_period,
            avg_value=statistics.mean(values),
            min_value=min(values),
            max_value=max(values),
            p50_value=p50,
            p95_value=p95,
            p99_value=p99,
            sample_count=len(values),
            trend_direction=trend_direction,
            trend_percentage=trend_percentage
        )
    
    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not sorted_values:
            return 0
        
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f >= len(sorted_values) - 1:
            return sorted_values[-1]
        
        return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
    
    def _calculate_model_recommendation_score(self, 
                                            response_time_summary: Optional[PerformanceSummary],
                                            error_rate: float,
                                            total_requests: int) -> float:
        """Calculate model recommendation score (0-100)"""
        score = 100
        
        # Response time impact (up to -40 points)
        if response_time_summary:
            baseline_response = self.baselines.get('response_time', {}).get('mean', 1500)
            if response_time_summary.avg_value > baseline_response:
                response_penalty = min(40, (response_time_summary.avg_value - baseline_response) / baseline_response * 40)
                score -= response_penalty
        
        # Error rate impact (up to -30 points)
        baseline_error_rate = self.baselines.get('error_rate', 2.0)
        if error_rate > baseline_error_rate:
            error_penalty = min(30, (error_rate - baseline_error_rate) * 3)
            score -= error_penalty
        
        # Volume bonus/penalty (up to ±10 points)
        if total_requests < 10:
            score -= 10  # Penalty for low usage
        elif total_requests > 1000:
            score += 10  # Bonus for high usage
        
        return max(0, min(100, score))
    
    def _identify_performance_issues(self, 
                                   response_time_summary: Optional[PerformanceSummary],
                                   error_rate: float,
                                   model_id: str) -> List[str]:
        """Identify specific performance issues"""
        issues = []
        
        if response_time_summary:
            # High response time
            if response_time_summary.avg_value > self.performance_thresholds['response_time_warning']:
                issues.append(f"High average response time ({response_time_summary.avg_value:.0f}ms)")
            
            # High variance in response time
            if response_time_summary.max_value > response_time_summary.avg_value * 3:
                issues.append("High response time variance detected")
            
            # Degrading performance trend
            if response_time_summary.trend_direction == 'degrading' and abs(response_time_summary.trend_percentage) > 20:
                issues.append(f"Performance degrading ({response_time_summary.trend_percentage:.1f}% increase)")
        
        # High error rate
        if error_rate > self.performance_thresholds['error_rate_warning']:
            issues.append(f"High error rate ({error_rate:.1f}%)")
        
        return issues
    
    def get_system_performance_overview(self, days: int = 1) -> Dict[str, Any]:
        """Get system-wide performance overview"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all events for the period
        events = self.analytics_logger.get_events(
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            limit=50000
        )
        
        if not events:
            return {"error": "No data available for the specified period"}
        
        # Overall metrics
        total_requests = len(events)
        unique_users = len(set(e['user_id'] for e in events if e.get('user_id')))
        
        # Response time analysis
        response_times = [e['response_time_ms'] for e in events if e.get('response_time_ms')]
        response_time_stats = {}
        if response_times:
            response_time_stats = {
                'avg': statistics.mean(response_times),
                'p50': self._percentile(sorted(response_times), 50),
                'p95': self._percentile(sorted(response_times), 95),
                'p99': self._percentile(sorted(response_times), 99),
                'max': max(response_times)
            }
        
        # Error analysis
        errors = [e for e in events if e.get('error_code')]
        error_rate = (len(errors) / total_requests * 100) if total_requests > 0 else 0
        
        # Model performance comparison
        model_stats = defaultdict(lambda: {
            'requests': 0,
            'response_times': [],
            'errors': 0,
            'unique_users': set()
        })
        
        for event in events:
            if event.get('model_id'):
                model_id = event['model_id']
                model_stats[model_id]['requests'] += 1
                
                if event.get('response_time_ms'):
                    model_stats[model_id]['response_times'].append(event['response_time_ms'])
                
                if event.get('error_code'):
                    model_stats[model_id]['errors'] += 1
                
                if event.get('user_id'):
                    model_stats[model_id]['unique_users'].add(event['user_id'])
        
        # Process model statistics
        model_performance = {}
        for model_id, stats in model_stats.items():
            model_performance[model_id] = {
                'requests': stats['requests'],
                'avg_response_time': statistics.mean(stats['response_times']) if stats['response_times'] else 0,
                'error_rate': (stats['errors'] / stats['requests'] * 100) if stats['requests'] > 0 else 0,
                'unique_users': len(stats['unique_users'])
            }
        
        # System health assessment
        health_score = 100
        health_issues = []
        
        if response_time_stats.get('avg', 0) > self.performance_thresholds['response_time_warning']:
            health_score -= 20
            health_issues.append("High average response time")
        
        if error_rate > self.performance_thresholds['error_rate_warning']:
            health_score -= 30
            health_issues.append(f"High error rate ({error_rate:.1f}%)")
        
        if total_requests < 10:
            health_score -= 10
            health_issues.append("Low system usage")
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            'period': f"{days} days",
            'total_requests': total_requests,
            'unique_users': unique_users,
            'response_time_stats': response_time_stats,
            'error_rate': error_rate,
            'error_count': len(errors),
            'model_performance': model_performance,
            'health_score': health_score,
            'health_status': health_status,
            'health_issues': health_issues,
            'requests_per_hour': total_requests / (days * 24) if days > 0 else 0
        }
    
    def get_performance_trends(self, days: int = 7, interval_hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over time with specified interval"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get events
        events = self.analytics_logger.get_events(
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            limit=50000
        )
        
        if not events:
            return {"error": "No data available"}
        
        # Group events by time intervals
        time_buckets = defaultdict(list)
        
        for event in events:
            event_time = datetime.fromisoformat(event['timestamp'])
            # Round down to the nearest interval
            bucket_time = event_time.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(
                hours=(event_time.hour // interval_hours) * interval_hours
            )
            time_buckets[bucket_time].append(event)
        
        # Calculate metrics for each time bucket
        trend_data = []
        for bucket_time in sorted(time_buckets.keys()):
            bucket_events = time_buckets[bucket_time]
            
            # Calculate metrics for this time bucket
            response_times = [e['response_time_ms'] for e in bucket_events if e.get('response_time_ms')]
            errors = [e for e in bucket_events if e.get('error_code')]
            
            trend_point = {
                'timestamp': bucket_time.isoformat(),
                'requests': len(bucket_events),
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'error_rate': (len(errors) / len(bucket_events) * 100) if bucket_events else 0,
                'unique_users': len(set(e['user_id'] for e in bucket_events if e.get('user_id')))
            }
            trend_data.append(trend_point)
        
        return {
            'period': f"{days} days",
            'interval_hours': interval_hours,
            'data_points': len(trend_data),
            'trends': trend_data
        }

# Global instance
_performance_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker
