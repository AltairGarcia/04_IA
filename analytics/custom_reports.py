"""
Custom Reports Generator for LangGraph 101 AI Platform

Provides flexible report generation capabilities with custom filters, 
data aggregations, and export options for comprehensive analytics.
"""

import json
import csv
import io
import statistics
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict
import sqlite3

from .analytics_logger import get_analytics_logger
from .user_behavior_analyzer import get_user_behavior_analyzer
from .performance_tracker import get_performance_tracker

@dataclass
class ReportFilter:
    """Filter configuration for custom reports"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    user_ids: Optional[List[str]] = None
    model_ids: Optional[List[str]] = None
    event_types: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    error_codes: Optional[List[str]] = None
    custom_filters: Optional[Dict[str, Any]] = None

@dataclass
class ReportAggregation:
    """Aggregation configuration for custom reports"""
    group_by: List[str]  # Fields to group by
    metrics: List[str]   # Metrics to calculate
    time_granularity: Optional[str] = None  # 'hour', 'day', 'week', 'month'

@dataclass
class CustomReport:
    """Custom report configuration and results"""
    report_id: str
    title: str
    description: str
    filters: ReportFilter
    aggregations: ReportAggregation
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    generated_at: str
    export_formats: List[str] = None

class CustomReportGenerator:
    """Flexible report generator with custom filtering and aggregation"""
    
    def __init__(self):
        self.analytics_logger = get_analytics_logger()
        self.user_behavior_analyzer = get_user_behavior_analyzer()
        self.performance_tracker = get_performance_tracker()
        
        # Pre-defined report templates
        self.report_templates = {
            'user_engagement': self._create_user_engagement_template(),
            'model_performance': self._create_model_performance_template(),
            'error_analysis': self._create_error_analysis_template(),
            'usage_trends': self._create_usage_trends_template(),
            'cost_analysis': self._create_cost_analysis_template(),
            'feature_adoption': self._create_feature_adoption_template()
        }
    
    def generate_custom_report(self, 
                             title: str,
                             filters: ReportFilter,
                             aggregations: ReportAggregation,
                             description: str = "") -> CustomReport:
        """Generate a custom report with specified filters and aggregations"""
        
        report_id = f"custom_{int(datetime.now().timestamp())}"
        
        # Get filtered data
        filtered_events = self._apply_filters(filters)
        
        # Apply aggregations
        aggregated_data = self._apply_aggregations(filtered_events, aggregations)
        
        # Calculate metadata
        metadata = self._calculate_report_metadata(filtered_events, aggregations)
        
        return CustomReport(
            report_id=report_id,
            title=title,
            description=description,
            filters=filters,
            aggregations=aggregations,
            data=aggregated_data,
            metadata=metadata,
            generated_at=datetime.now().isoformat(),
            export_formats=['json', 'csv', 'html']
        )
    
    def generate_template_report(self, 
                               template_name: str, 
                               **kwargs) -> Optional[CustomReport]:
        """Generate a report from a pre-defined template"""
        
        if template_name not in self.report_templates:
            return None
        
        template = self.report_templates[template_name]
        
        # Apply any custom parameters
        if 'days' in kwargs:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=kwargs['days'])
            template['filters'].start_date = start_date.isoformat()
            template['filters'].end_date = end_date.isoformat()
        
        if 'model_ids' in kwargs:
            template['filters'].model_ids = kwargs['model_ids']
        
        if 'user_ids' in kwargs:
            template['filters'].user_ids = kwargs['user_ids']
        
        return self.generate_custom_report(
            title=template['title'],
            description=template['description'],
            filters=template['filters'],
            aggregations=template['aggregations']
        )
    
    def _apply_filters(self, filters: ReportFilter) -> List[Dict[str, Any]]:
        """Apply filters to get relevant events"""
        
        # Build filter parameters for analytics logger
        filter_params = {}
        
        if filters.start_date:
            filter_params['start_time'] = filters.start_date
        
        if filters.end_date:
            filter_params['end_time'] = filters.end_date
        
        # Get initial dataset
        events = self.analytics_logger.get_events(
            limit=50000,
            **filter_params
        )
        
        # Apply additional filters
        filtered_events = []
        
        for event in events:
            # Check user_ids filter
            if filters.user_ids and event.get('user_id') not in filters.user_ids:
                continue
            
            # Check model_ids filter
            if filters.model_ids and event.get('model_id') not in filters.model_ids:
                continue
            
            # Check event_types filter
            if filters.event_types and event.get('event_type') not in filters.event_types:
                continue
            
            # Check session_ids filter
            if filters.session_ids and event.get('session_id') not in filters.session_ids:
                continue
            
            # Check error_codes filter
            if filters.error_codes:
                if not event.get('error_code') or event.get('error_code') not in filters.error_codes:
                    continue
            
            # Apply custom filters
            if filters.custom_filters:
                skip_event = False
                for field, value in filters.custom_filters.items():
                    if isinstance(value, list):
                        if event.get(field) not in value:
                            skip_event = True
                            break
                    else:
                        if event.get(field) != value:
                            skip_event = True
                            break
                
                if skip_event:
                    continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def _apply_aggregations(self, 
                          events: List[Dict[str, Any]], 
                          aggregations: ReportAggregation) -> List[Dict[str, Any]]:
        """Apply aggregations to the filtered events"""
        
        if not events:
            return []
        
        # Group events by specified fields
        grouped_data = defaultdict(list)
        
        for event in events:
            # Create group key
            group_key = []
            for field in aggregations.group_by:
                if field == 'time' and aggregations.time_granularity:
                    # Time-based grouping
                    event_time = datetime.fromisoformat(event['timestamp'])
                    time_key = self._get_time_group_key(event_time, aggregations.time_granularity)
                    group_key.append(time_key)
                else:
                    group_key.append(str(event.get(field, 'unknown')))
            
            group_key_str = '|'.join(group_key)
            grouped_data[group_key_str].append(event)
        
        # Calculate metrics for each group
        aggregated_results = []
        
        for group_key, group_events in grouped_data.items():
            result = {}
            
            # Parse group key back to individual fields
            group_values = group_key.split('|')
            for i, field in enumerate(aggregations.group_by):
                if i < len(group_values):
                    result[field] = group_values[i]
            
            # Calculate requested metrics
            for metric in aggregations.metrics:
                result.update(self._calculate_metric(metric, group_events))
            
            aggregated_results.append(result)
        
        # Sort results by first group_by field
        if aggregations.group_by:
            sort_field = aggregations.group_by[0]
            aggregated_results.sort(key=lambda x: x.get(sort_field, ''))
        
        return aggregated_results
    
    def _get_time_group_key(self, event_time: datetime, granularity: str) -> str:
        """Get time group key based on granularity"""
        if granularity == 'hour':
            return event_time.strftime('%Y-%m-%d %H:00')
        elif granularity == 'day':
            return event_time.strftime('%Y-%m-%d')
        elif granularity == 'week':
            # Get Monday of the week
            monday = event_time - timedelta(days=event_time.weekday())
            return monday.strftime('%Y-%m-%d (Week)')
        elif granularity == 'month':
            return event_time.strftime('%Y-%m')
        else:
            return event_time.strftime('%Y-%m-%d')
    
    def _calculate_metric(self, metric: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate a specific metric for a group of events"""
        
        result = {}
        
        if metric == 'count':
            result['count'] = len(events)
        
        elif metric == 'unique_users':
            unique_users = set(e.get('user_id') for e in events if e.get('user_id'))
            result['unique_users'] = len(unique_users)
        
        elif metric == 'unique_sessions':
            unique_sessions = set(e.get('session_id') for e in events if e.get('session_id'))
            result['unique_sessions'] = len(unique_sessions)
        
        elif metric == 'avg_response_time':
            response_times = [e.get('response_time_ms') for e in events if e.get('response_time_ms')]
            result['avg_response_time'] = statistics.mean(response_times) if response_times else 0
        
        elif metric == 'total_tokens':
            total_tokens = sum(e.get('tokens_used', 0) for e in events)
            result['total_tokens'] = total_tokens
        
        elif metric == 'total_cost':
            total_cost = sum(e.get('cost_estimate', 0) for e in events)
            result['total_cost'] = total_cost
        
        elif metric == 'error_rate':
            errors = sum(1 for e in events if e.get('error_code'))
            result['error_rate'] = (errors / len(events) * 100) if events else 0
        
        elif metric == 'success_rate':
            errors = sum(1 for e in events if e.get('error_code'))
            result['success_rate'] = ((len(events) - errors) / len(events) * 100) if events else 100
        
        elif metric == 'p95_response_time':
            response_times = [e.get('response_time_ms') for e in events if e.get('response_time_ms')]
            if response_times:
                sorted_times = sorted(response_times)
                p95_index = int(len(sorted_times) * 0.95)
                result['p95_response_time'] = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            else:
                result['p95_response_time'] = 0
        
        elif metric == 'avg_tokens_per_request':
            tokens = [e.get('tokens_used') for e in events if e.get('tokens_used')]
            result['avg_tokens_per_request'] = statistics.mean(tokens) if tokens else 0
        
        elif metric == 'cost_per_request':
            costs = [e.get('cost_estimate') for e in events if e.get('cost_estimate')]
            result['cost_per_request'] = statistics.mean(costs) if costs else 0
        
        return result
    
    def _calculate_report_metadata(self, 
                                 events: List[Dict[str, Any]], 
                                 aggregations: ReportAggregation) -> Dict[str, Any]:
        """Calculate metadata about the report"""
        
        if not events:
            return {
                'total_events': 0,
                'date_range': 'No data',
                'unique_users': 0,
                'unique_models': 0,
                'data_quality': 'No data available'
            }
        
        # Basic statistics
        total_events = len(events)
        
        # Date range
        timestamps = [e['timestamp'] for e in events if e.get('timestamp')]
        if timestamps:
            min_date = min(timestamps)
            max_date = max(timestamps)
            date_range = f"{min_date.split('T')[0]} to {max_date.split('T')[0]}"
        else:
            date_range = 'Unknown'
        
        # Unique counts
        unique_users = len(set(e.get('user_id') for e in events if e.get('user_id')))
        unique_models = len(set(e.get('model_id') for e in events if e.get('model_id')))
        unique_sessions = len(set(e.get('session_id') for e in events if e.get('session_id')))
        
        # Data quality assessment
        events_with_response_time = sum(1 for e in events if e.get('response_time_ms'))
        events_with_tokens = sum(1 for e in events if e.get('tokens_used'))
        events_with_cost = sum(1 for e in events if e.get('cost_estimate'))
        
        data_completeness = {
            'response_time': (events_with_response_time / total_events * 100) if total_events > 0 else 0,
            'tokens': (events_with_tokens / total_events * 100) if total_events > 0 else 0,
            'cost': (events_with_cost / total_events * 100) if total_events > 0 else 0
        }
        
        avg_completeness = statistics.mean(data_completeness.values())
        data_quality = 'Excellent' if avg_completeness >= 90 else 'Good' if avg_completeness >= 70 else 'Fair' if avg_completeness >= 50 else 'Poor'
        
        return {
            'total_events': total_events,
            'date_range': date_range,
            'unique_users': unique_users,
            'unique_models': unique_models,
            'unique_sessions': unique_sessions,
            'data_completeness': data_completeness,
            'data_quality': data_quality,
            'aggregation_groups': len(aggregations.group_by),
            'metrics_calculated': len(aggregations.metrics)
        }
    
    def export_report(self, report: CustomReport, format: str = 'json') -> str:
        """Export report in specified format"""
        
        if format.lower() == 'json':
            return json.dumps(asdict(report), indent=2, default=str)
        
        elif format.lower() == 'csv':
            return self._export_csv(report)
        
        elif format.lower() == 'html':
            return self._export_html(report)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self, report: CustomReport) -> str:
        """Export report data as CSV"""
        if not report.data:
            return "No data to export"
        
        output = io.StringIO()
        
        # Get all unique keys from the data
        fieldnames = set()
        for row in report.data:
            fieldnames.update(row.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in report.data:
            writer.writerow(row)
        
        return output.getvalue()
    
    def _export_html(self, report: CustomReport) -> str:
        """Export report as HTML"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metadata {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #0066cc; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p>{report.description}</p>
                <p><strong>Generated:</strong> {report.generated_at}</p>
            </div>
            
            <div class="metadata">
                <h3>Report Metadata</h3>
                <ul>
        """
        
        for key, value in report.metadata.items():
            html_template += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
        
        html_template += """
                </ul>
            </div>
            
            <h3>Report Data</h3>
            <table>
        """
        
        if report.data:
            # Table headers
            headers = list(report.data[0].keys())
            html_template += "<tr>"
            for header in headers:
                html_template += f"<th>{header.replace('_', ' ').title()}</th>"
            html_template += "</tr>"
            
            # Table rows
            for row in report.data:
                html_template += "<tr>"
                for header in headers:
                    value = row.get(header, '')
                    if isinstance(value, (int, float)) and header in ['count', 'total_tokens', 'total_cost']:
                        html_template += f'<td class="metric">{value}</td>'
                    else:
                        html_template += f"<td>{value}</td>"
                html_template += "</tr>"
        else:
            html_template += "<tr><td colspan='100%'>No data available</td></tr>"
        
        html_template += """
            </table>
        </body>
        </html>
        """
        
        return html_template
    
    # Report templates
    def _create_user_engagement_template(self) -> Dict[str, Any]:
        """Create user engagement report template"""
        return {
            'title': 'User Engagement Analysis',
            'description': 'Analysis of user engagement patterns and behavior',
            'filters': ReportFilter(
                start_date=(datetime.now() - timedelta(days=30)).isoformat(),
                end_date=datetime.now().isoformat()
            ),
            'aggregations': ReportAggregation(
                group_by=['user_id'],
                metrics=['count', 'unique_sessions', 'avg_response_time', 'total_tokens', 'total_cost']
            )
        }
    
    def _create_model_performance_template(self) -> Dict[str, Any]:
        """Create model performance report template"""
        return {
            'title': 'Model Performance Comparison',
            'description': 'Comparative analysis of AI model performance metrics',
            'filters': ReportFilter(
                start_date=(datetime.now() - timedelta(days=7)).isoformat(),
                end_date=datetime.now().isoformat()
            ),
            'aggregations': ReportAggregation(
                group_by=['model_id'],
                metrics=['count', 'unique_users', 'avg_response_time', 'p95_response_time', 'error_rate', 'avg_tokens_per_request', 'cost_per_request']
            )
        }
    
    def _create_error_analysis_template(self) -> Dict[str, Any]:
        """Create error analysis report template"""
        return {
            'title': 'Error Analysis Report',
            'description': 'Analysis of errors and failure patterns',
            'filters': ReportFilter(
                start_date=(datetime.now() - timedelta(days=7)).isoformat(),
                end_date=datetime.now().isoformat(),
                custom_filters={'error_code': None}  # Only events with error codes
            ),
            'aggregations': ReportAggregation(
                group_by=['error_code', 'model_id'],
                metrics=['count', 'unique_users', 'avg_response_time']
            )
        }
    
    def _create_usage_trends_template(self) -> Dict[str, Any]:
        """Create usage trends report template"""
        return {
            'title': 'Usage Trends Over Time',
            'description': 'Time-based analysis of platform usage patterns',
            'filters': ReportFilter(
                start_date=(datetime.now() - timedelta(days=30)).isoformat(),
                end_date=datetime.now().isoformat()
            ),
            'aggregations': ReportAggregation(
                group_by=['time'],
                metrics=['count', 'unique_users', 'avg_response_time', 'total_tokens'],
                time_granularity='day'
            )
        }
    
    def _create_cost_analysis_template(self) -> Dict[str, Any]:
        """Create cost analysis report template"""
        return {
            'title': 'Cost Analysis Report',
            'description': 'Analysis of usage costs and efficiency',
            'filters': ReportFilter(
                start_date=(datetime.now() - timedelta(days=30)).isoformat(),
                end_date=datetime.now().isoformat()
            ),
            'aggregations': ReportAggregation(
                group_by=['model_id'],
                metrics=['count', 'total_cost', 'cost_per_request', 'total_tokens', 'avg_tokens_per_request']
            )
        }
    
    def _create_feature_adoption_template(self) -> Dict[str, Any]:
        """Create feature adoption report template"""
        return {
            'title': 'Feature Adoption Analysis',
            'description': 'Analysis of feature usage and adoption rates',
            'filters': ReportFilter(
                start_date=(datetime.now() - timedelta(days=30)).isoformat(),
                end_date=datetime.now().isoformat()
            ),
            'aggregations': ReportAggregation(
                group_by=['event_type'],
                metrics=['count', 'unique_users', 'unique_sessions']
            )
        }
    
    def list_available_templates(self) -> List[Dict[str, str]]:
        """List all available report templates"""
        return [
            {
                'template_name': name,
                'title': template['title'],
                'description': template['description']
            }
            for name, template in self.report_templates.items()
        ]

# Global instance
_custom_report_generator = None

def get_custom_report_generator() -> CustomReportGenerator:
    """Get the global custom report generator instance"""
    global _custom_report_generator
    if _custom_report_generator is None:
        _custom_report_generator = CustomReportGenerator()
    return _custom_report_generator
