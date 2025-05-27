"""
Custom Reports Generator for LangGraph 101 AI Platform

Generates customizable reports from analytics data with various formats and scheduling.
"""

import json
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from io import BytesIO
import base64

from .analytics_logger import get_analytics_logger

@dataclass
class ReportConfig:
    """Configuration for custom report generation"""
    report_id: str
    name: str
    description: str
    report_type: str  # 'summary', 'detailed', 'comparison', 'trend'
    data_sources: List[str]  # ['api_calls', 'user_behavior', 'performance', 'errors']
    date_range: Dict[str, str]  # {'start': '2024-01-01', 'end': '2024-01-31'}
    filters: Dict[str, Any]
    grouping: Optional[str] = None
    metrics: List[str] = None
    format: str = 'html'  # 'html', 'pdf', 'excel', 'json'
    schedule: Optional[str] = None  # 'daily', 'weekly', 'monthly'
    created_at: str = None
    created_by: str = "system"

@dataclass
class ReportData:
    """Container for report data and metadata"""
    config: ReportConfig
    data: Dict[str, Any]
    charts: List[Dict[str, Any]]
    summary: Dict[str, Any]
    generated_at: str
    file_path: Optional[str] = None

class CustomReportsGenerator:
    """Generates custom reports from analytics data"""
    
    def __init__(self):
        self.analytics_logger = get_analytics_logger()
        self.report_templates = self._load_report_templates()
        
    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined report templates"""
        return {
            'daily_summary': {
                'name': 'Daily Summary Report',
                'description': 'Daily overview of system activity',
                'data_sources': ['api_calls', 'user_behavior', 'performance'],
                'metrics': ['total_requests', 'active_users', 'avg_response_time', 'error_rate'],
                'charts': ['timeline', 'metrics_overview']
            },
            'weekly_performance': {
                'name': 'Weekly Performance Report',
                'description': 'Weekly performance analysis',
                'data_sources': ['performance', 'api_calls'],
                'metrics': ['response_times', 'throughput', 'error_rates', 'resource_usage'],
                'charts': ['trend_analysis', 'performance_comparison']
            },
            'monthly_insights': {
                'name': 'Monthly Insights Report',
                'description': 'Monthly business intelligence report',
                'data_sources': ['user_behavior', 'api_calls', 'performance', 'errors'],
                'metrics': ['user_growth', 'feature_adoption', 'cost_analysis', 'quality_metrics'],
                'charts': ['growth_trends', 'feature_usage', 'cost_breakdown']
            },
            'user_behavior_analysis': {
                'name': 'User Behavior Analysis',
                'description': 'Detailed user behavior and engagement analysis',
                'data_sources': ['user_behavior'],
                'metrics': ['session_duration', 'feature_usage', 'user_journeys', 'engagement_scores'],
                'charts': ['user_segments', 'behavior_patterns', 'engagement_heatmap']
            },
            'api_performance': {
                'name': 'API Performance Report',
                'description': 'API performance and usage analysis',
                'data_sources': ['api_calls', 'performance'],
                'metrics': ['endpoint_usage', 'response_times', 'error_rates', 'rate_limits'],
                'charts': ['endpoint_performance', 'usage_patterns', 'error_distribution']
            }
        }
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available report templates"""
        return [
            {
                'id': template_id,
                'name': template['name'],
                'description': template['description'],
                'data_sources': template['data_sources']
            }
            for template_id, template in self.report_templates.items()
        ]
    
    def create_report_config(self, template_id: str = None, **kwargs) -> ReportConfig:
        """Create a report configuration"""
        if template_id and template_id in self.report_templates:
            template = self.report_templates[template_id]
            config_data = {
                'report_id': f"{template_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'name': template['name'],
                'description': template['description'],
                'report_type': 'summary',
                'data_sources': template['data_sources'],
                'metrics': template.get('metrics', []),
                'date_range': {
                    'start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'end': datetime.now().strftime('%Y-%m-%d')
                },
                'filters': {},
                'format': 'html',
                'created_at': datetime.now().isoformat()
            }
        else:
            config_data = {
                'report_id': f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'name': 'Custom Report',
                'description': 'User-defined custom report',
                'report_type': 'summary',
                'data_sources': ['api_calls'],
                'date_range': {
                    'start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'end': datetime.now().strftime('%Y-%m-%d')
                },
                'filters': {},
                'metrics': [],
                'format': 'html',
                'created_at': datetime.now().isoformat()
            }
        
        # Override with provided kwargs
        config_data.update(kwargs)
        
        return ReportConfig(**config_data)
    
    def generate_report(self, config: ReportConfig) -> ReportData:
        """Generate a report based on configuration"""
        # Collect data from specified sources
        report_data = {}
        
        for source in config.data_sources:
            if source == 'api_calls':
                report_data['api_calls'] = self._get_api_calls_data(config)
            elif source == 'user_behavior':
                report_data['user_behavior'] = self._get_user_behavior_data(config)
            elif source == 'performance':
                report_data['performance'] = self._get_performance_data(config)
            elif source == 'errors':
                report_data['errors'] = self._get_error_data(config)
        
        # Generate charts
        charts = self._generate_charts(report_data, config)
        
        # Create summary
        summary = self._generate_summary(report_data, config)
        
        return ReportData(
            config=config,
            data=report_data,
            charts=charts,
            summary=summary,
            generated_at=datetime.now().isoformat()
        )
    
    def _get_api_calls_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Get API calls data for the report"""
        try:
            # Get data from analytics logger
            start_date = datetime.fromisoformat(config.date_range['start'])
            end_date = datetime.fromisoformat(config.date_range['end'])
            
            api_data = self.analytics_logger.get_api_calls_in_range(start_date, end_date)
            
            # Process and aggregate data
            total_calls = len(api_data)
            success_rate = len([call for call in api_data if call.get('status') == 'success']) / total_calls if total_calls > 0 else 0
            avg_response_time = sum(call.get('response_time_ms', 0) for call in api_data) / total_calls if total_calls > 0 else 0
            
            # Group by provider
            provider_stats = {}
            for call in api_data:
                provider = call.get('provider', 'unknown')
                if provider not in provider_stats:
                    provider_stats[provider] = {'count': 0, 'total_time': 0}
                provider_stats[provider]['count'] += 1
                provider_stats[provider]['total_time'] += call.get('response_time_ms', 0)
            
            return {
                'total_calls': total_calls,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'provider_stats': provider_stats,
                'daily_breakdown': self._group_by_day(api_data),
                'raw_data': api_data[:100]  # Limit raw data
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_user_behavior_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Get user behavior data for the report"""
        try:
            start_date = datetime.fromisoformat(config.date_range['start'])
            end_date = datetime.fromisoformat(config.date_range['end'])
            
            user_data = self.analytics_logger.get_user_interactions_in_range(start_date, end_date)
            
            # Process user behavior metrics
            total_sessions = len(set(interaction.get('session_id') for interaction in user_data if interaction.get('session_id')))
            total_users = len(set(interaction.get('user_id') for interaction in user_data if interaction.get('user_id')))
            
            return {
                'total_sessions': total_sessions,
                'total_users': total_users,
                'avg_session_duration': self._calculate_avg_session_duration(user_data),
                'feature_usage': self._calculate_feature_usage(user_data),
                'user_segments': self._analyze_user_segments(user_data)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_performance_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Get performance data for the report"""
        try:
            start_date = datetime.fromisoformat(config.date_range['start'])
            end_date = datetime.fromisoformat(config.date_range['end'])
            
            perf_data = self.analytics_logger.get_performance_metrics_in_range(start_date, end_date)
            
            return {
                'avg_response_time': sum(p.get('response_time_ms', 0) for p in perf_data) / len(perf_data) if perf_data else 0,
                'max_response_time': max(p.get('response_time_ms', 0) for p in perf_data) if perf_data else 0,
                'min_response_time': min(p.get('response_time_ms', 0) for p in perf_data) if perf_data else 0,
                'throughput': len(perf_data) / ((end_date - start_date).total_seconds() / 3600) if perf_data else 0,  # requests per hour
                'performance_trends': self._analyze_performance_trends(perf_data)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_error_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Get error data for the report"""
        try:
            start_date = datetime.fromisoformat(config.date_range['start'])
            end_date = datetime.fromisoformat(config.date_range['end'])
            
            error_data = self.analytics_logger.get_errors_in_range(start_date, end_date)
            
            # Analyze error patterns
            error_categories = {}
            for error in error_data:
                category = error.get('category', 'unknown')
                error_categories[category] = error_categories.get(category, 0) + 1
            
            return {
                'total_errors': len(error_data),
                'error_categories': error_categories,
                'error_rate_trend': self._calculate_error_rate_trend(error_data),
                'top_errors': self._get_top_errors(error_data)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_charts(self, data: Dict[str, Any], config: ReportConfig) -> List[Dict[str, Any]]:
        """Generate charts for the report"""
        charts = []
        
        # API Calls Timeline Chart
        if 'api_calls' in data and 'daily_breakdown' in data['api_calls']:
            timeline_chart = self._create_timeline_chart(data['api_calls']['daily_breakdown'])
            if timeline_chart:
                charts.append(timeline_chart)
        
        # Performance Metrics Chart
        if 'performance' in data:
            perf_chart = self._create_performance_chart(data['performance'])
            if perf_chart:
                charts.append(perf_chart)
        
        # Error Distribution Chart
        if 'errors' in data and 'error_categories' in data['errors']:
            error_chart = self._create_error_distribution_chart(data['errors']['error_categories'])
            if error_chart:
                charts.append(error_chart)
        
        return charts
    
    def _create_timeline_chart(self, daily_data: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Create timeline chart from daily data"""
        try:
            dates = list(daily_data.keys())
            values = list(daily_data.values())
            
            fig = px.line(x=dates, y=values, title="API Calls Timeline")
            fig.update_layout(xaxis_title="Date", yaxis_title="API Calls")
            
            return {
                'type': 'timeline',
                'title': 'API Calls Timeline',
                'chart_data': fig.to_json()
            }
        except Exception:
            return None
    
    def _create_performance_chart(self, perf_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create performance metrics chart"""
        try:
            metrics = ['avg_response_time', 'max_response_time', 'min_response_time']
            values = [perf_data.get(metric, 0) for metric in metrics]
            
            fig = go.Figure(data=[go.Bar(x=metrics, y=values)])
            fig.update_layout(title="Performance Metrics", yaxis_title="Response Time (ms)")
            
            return {
                'type': 'performance',
                'title': 'Performance Metrics',
                'chart_data': fig.to_json()
            }
        except Exception:
            return None
    
    def _create_error_distribution_chart(self, error_categories: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Create error distribution pie chart"""
        try:
            fig = px.pie(values=list(error_categories.values()), 
                        names=list(error_categories.keys()),
                        title="Error Distribution by Category")
            
            return {
                'type': 'error_distribution',
                'title': 'Error Distribution',
                'chart_data': fig.to_json()
            }
        except Exception:
            return None
    
    def _generate_summary(self, data: Dict[str, Any], config: ReportConfig) -> Dict[str, Any]:
        """Generate executive summary for the report"""
        summary = {
            'report_period': f"{config.date_range['start']} to {config.date_range['end']}",
            'data_sources': config.data_sources,
            'key_metrics': {},
            'insights': [],
            'recommendations': []
        }
        
        # Extract key metrics
        if 'api_calls' in data:
            summary['key_metrics']['total_api_calls'] = data['api_calls'].get('total_calls', 0)
            summary['key_metrics']['api_success_rate'] = f"{data['api_calls'].get('success_rate', 0):.1%}"
        
        if 'performance' in data:
            summary['key_metrics']['avg_response_time'] = f"{data['performance'].get('avg_response_time', 0):.2f}ms"
        
        if 'errors' in data:
            summary['key_metrics']['total_errors'] = data['errors'].get('total_errors', 0)
        
        # Generate insights
        summary['insights'] = self._generate_insights(data)
        
        return summary
    
    def _generate_insights(self, data: Dict[str, Any]) -> List[str]:
        """Generate insights from the data"""
        insights = []
        
        if 'api_calls' in data:
            success_rate = data['api_calls'].get('success_rate', 0)
            if success_rate < 0.95:
                insights.append(f"API success rate is {success_rate:.1%}, below recommended 95%")
            else:
                insights.append(f"API success rate is healthy at {success_rate:.1%}")
        
        if 'performance' in data:
            avg_time = data['performance'].get('avg_response_time', 0)
            if avg_time > 2000:
                insights.append(f"Average response time of {avg_time:.0f}ms is above optimal threshold")
        
        if 'errors' in data:
            total_errors = data['errors'].get('total_errors', 0)
            if total_errors > 0:
                insights.append(f"Detected {total_errors} errors requiring attention")
        
        return insights
    
    def export_report(self, report: ReportData, format: str = None) -> str:
        """Export report to specified format"""
        export_format = format or report.config.format
        
        if export_format == 'json':
            return self._export_json(report)
        elif export_format == 'html':
            return self._export_html(report)
        elif export_format == 'excel':
            return self._export_excel(report)
        else:
            return self._export_html(report)  # Default to HTML
    
    def _export_json(self, report: ReportData) -> str:
        """Export report as JSON"""
        return json.dumps(asdict(report), indent=2, default=str)
    
    def _export_html(self, report: ReportData) -> str:
        """Export report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.config.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e7f3ff; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.config.name}</h1>
                <p>{report.config.description}</p>
                <p>Generated: {report.generated_at}</p>
                <p>Period: {report.summary['report_period']}</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                {self._format_metrics_html(report.summary['key_metrics'])}
            </div>
            
            <div class="section">
                <h2>Insights</h2>
                <ul>
                    {''.join(f'<li>{insight}</li>' for insight in report.summary['insights'])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _export_excel(self, report: ReportData) -> bytes:
        """Export report as Excel file"""
        # This would require openpyxl or xlsxwriter
        # For now, return a simple representation
        return json.dumps(asdict(report), indent=2, default=str).encode()
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML"""
        html = ""
        for key, value in metrics.items():
            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        return html
      # Helper methods
    def _group_by_day(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group data by day"""
        daily_counts = {}
        for item in data:
            date_str = item.get('timestamp', '')[:10]  # Extract date part
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        return daily_counts
    
    def _calculate_avg_session_duration(self, user_data: List[Dict[str, Any]]) -> float:
        """Calculate average session duration in minutes"""
        if not user_data:
            return 0.0
        
        # Group interactions by session
        sessions = {}
        for interaction in user_data:
            session_id = interaction.get('session_id')
            if not session_id:
                continue
                
            if session_id not in sessions:
                sessions[session_id] = {
                    'start_time': None,
                    'end_time': None,
                    'interactions': []
                }
            
            sessions[session_id]['interactions'].append(interaction)
            
            # Track session start and end times
            timestamp = interaction.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if sessions[session_id]['start_time'] is None or dt < sessions[session_id]['start_time']:
                        sessions[session_id]['start_time'] = dt
                    if sessions[session_id]['end_time'] is None or dt > sessions[session_id]['end_time']:
                        sessions[session_id]['end_time'] = dt
                except (ValueError, AttributeError):
                    continue
          # Calculate duration for each session
        total_duration = 0
        valid_sessions = 0
        
        for session_data in sessions.values():
            if session_data['start_time'] and session_data['end_time']:
                duration = (session_data['end_time'] - session_data['start_time']).total_seconds() / 60  # Convert to minutes
                total_duration += duration
                valid_sessions += 1
        
        return total_duration / valid_sessions if valid_sessions > 0 else 0.0
    
    def _calculate_feature_usage(self, user_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate feature usage statistics"""
        if not user_data:
            return {}
        
        feature_counts = {}
        
        for interaction in user_data:
            # Extract feature information from interaction data
            feature = interaction.get('feature')
            action = interaction.get('action')
            event_type = interaction.get('event_type')
            
            # Create feature identifier
            if feature:
                feature_name = feature
            elif action:
                feature_name = f"action_{action}"
            elif event_type:
                feature_name = f"event_{event_type}"
            else:
                # Try to extract from other fields
                endpoint = interaction.get('endpoint', '')
                if endpoint:
                    feature_name = f"endpoint_{endpoint.split('/')[-1]}"
                else:
                    feature_name = "unknown_feature"
            
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
          # Sort by usage count
        sorted_features = dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_features
    
    def _analyze_user_segments(self, user_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze user segments based on behavior patterns"""
        if not user_data:
            return {}
        
        # Group users by their behavior patterns
        user_profiles = {}
        
        for interaction in user_data:
            user_id = interaction.get('user_id')
            if not user_id:
                continue
            
            if user_id not in user_profiles:
                user_profiles[user_id] = {
                    'total_interactions': 0,
                    'unique_features': set(),
                    'session_count': set(),
                    'total_time_spent': 0,
                    'error_count': 0
                }
            
            profile = user_profiles[user_id]
            profile['total_interactions'] += 1
            
            # Track unique features used
            feature = interaction.get('feature') or interaction.get('action') or interaction.get('event_type')
            if feature:
                profile['unique_features'].add(feature)
            
            # Track sessions
            session_id = interaction.get('session_id')
            if session_id:
                profile['session_count'].add(session_id)
            
            # Track errors
            if interaction.get('error') or interaction.get('status') == 'error':
                profile['error_count'] += 1
        
        # Categorize users into segments
        segments = {
            'power_users': 0,      # High interaction count, many features
            'casual_users': 0,     # Low interaction count, few features
            'new_users': 0,        # Very low interaction count
            'engaged_users': 0,    # Medium interaction count, diverse features
            'struggling_users': 0  # High error rate
        }
        
        for user_id, profile in user_profiles.items():
            interaction_count = profile['total_interactions']
            feature_count = len(profile['unique_features'])
            session_count = len(profile['session_count'])
            error_rate = profile['error_count'] / interaction_count if interaction_count > 0 else 0
              # Segment classification logic
            if error_rate > 0.2:  # More than 20% errors
                segments['struggling_users'] += 1
            elif interaction_count >= 50 and feature_count >= 5:
                segments['power_users'] += 1
            elif interaction_count >= 20 and feature_count >= 3:
                segments['engaged_users'] += 1
            elif interaction_count <= 5:
                segments['new_users'] += 1
            else:
                segments['casual_users'] += 1
        
        return segments
    
    def _analyze_performance_trends(self, perf_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if not perf_data:
            return {
                'trend_direction': 'stable',
                'average_improvement': 0,
                'volatility': 0,
                'peak_hours': [],
                'daily_patterns': {}
            }
        
        # Sort data by timestamp
        sorted_data = sorted(perf_data, key=lambda x: x.get('timestamp', ''))
        
        # Extract response times with timestamps
        time_series = []
        daily_aggregates = {}
        hourly_aggregates = {}
        
        for item in sorted_data:
            response_time = item.get('response_time_ms', 0)
            timestamp = item.get('timestamp', '')
            
            if timestamp and response_time:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_series.append((dt, response_time))
                    
                    # Group by day
                    day_key = dt.strftime('%Y-%m-%d')
                    if day_key not in daily_aggregates:
                        daily_aggregates[day_key] = []
                    daily_aggregates[day_key].append(response_time)
                    
                    # Group by hour
                    hour_key = dt.hour
                    if hour_key not in hourly_aggregates:
                        hourly_aggregates[hour_key] = []
                    hourly_aggregates[hour_key].append(response_time)
                    
                except (ValueError, AttributeError):
                    continue
        
        if not time_series:
            return {
                'trend_direction': 'no_data',
                'average_improvement': 0,
                'volatility': 0,
                'peak_hours': [],
                'daily_patterns': {}
            }
        
        # Calculate trend direction
        response_times = [rt for _, rt in time_series]
        if len(response_times) >= 2:
            first_half = response_times[:len(response_times)//2]
            second_half = response_times[len(response_times)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            improvement = (first_avg - second_avg) / first_avg * 100 if first_avg > 0 else 0
            
            if improvement > 5:
                trend_direction = 'improving'
            elif improvement < -5:
                trend_direction = 'degrading'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'
            improvement = 0
        
        # Calculate volatility (coefficient of variation)
        avg_response_time = sum(response_times) / len(response_times)
        variance = sum((rt - avg_response_time) ** 2 for rt in response_times) / len(response_times)
        std_dev = variance ** 0.5
        volatility = (std_dev / avg_response_time * 100) if avg_response_time > 0 else 0
        
        # Find peak hours (hours with highest average response time)
        hour_averages = {}
        for hour, times in hourly_aggregates.items():
            hour_averages[hour] = sum(times) / len(times)
        
        # Get top 3 peak hours
        peak_hours = sorted(hour_averages.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_hours = [f"{hour}:00" for hour, _ in peak_hours]
        
        # Daily patterns
        daily_patterns = {}
        for day, times in daily_aggregates.items():
            daily_patterns[day] = {
                'avg_response_time': sum(times) / len(times),
                'max_response_time': max(times),
                'min_response_time': min(times),
                'request_count': len(times)
            }
        
        return {
            'trend_direction': trend_direction,
            'average_improvement': round(improvement, 2),
            'volatility': round(volatility, 2),
            'peak_hours': peak_hours,            'daily_patterns': daily_patterns
        }
    
    def _calculate_error_rate_trend(self, error_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate error rate trend over time"""
        if not error_data:
            return []
        
        # Group errors by time periods (daily)
        daily_errors = {}
        
        for error in error_data:
            timestamp = error.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    day_key = dt.strftime('%Y-%m-%d')
                    
                    if day_key not in daily_errors:
                        daily_errors[day_key] = {
                            'date': day_key,
                            'error_count': 0,
                            'total_requests': 0,
                            'error_rate': 0,
                            'error_types': {}
                        }
                    
                    daily_errors[day_key]['error_count'] += 1
                    
                    # Track error types
                    error_type = error.get('error_type') or error.get('category') or 'unknown'
                    error_types = daily_errors[day_key]['error_types']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                except (ValueError, AttributeError):
                    continue
        
        # Estimate total requests per day (this would ideally come from request logs)
        # For now, we'll estimate based on error patterns
        for day_data in daily_errors.values():
            # Rough estimation: assume error rate is typically 1-5%
            # So total requests ≈ error_count / estimated_error_rate
            estimated_total = max(day_data['error_count'] * 20, day_data['error_count'])  # Assume 5% error rate
            day_data['total_requests'] = estimated_total
            day_data['error_rate'] = (day_data['error_count'] / estimated_total * 100) if estimated_total > 0 else 0
          # Sort by date and return as list
        trend_data = sorted(daily_errors.values(), key=lambda x: x['date'])
        
        return trend_data
    
    def _get_top_errors(self, error_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get top errors by frequency with detailed analysis"""
        if not error_data:
            return []
        
        # Group errors by message/type
        error_groups = {}
        
        for error in error_data:
            # Create a key to group similar errors
            error_message = error.get('message') or error.get('error_message') or 'Unknown error'
            error_type = error.get('error_type') or error.get('category') or 'general'
            
            # Create a composite key for grouping
            key = f"{error_type}: {error_message[:100]}"  # Limit message length
            
            if key not in error_groups:
                error_groups[key] = {
                    'error_type': error_type,
                    'message': error_message,
                    'count': 0,
                    'first_seen': None,
                    'last_seen': None,
                    'affected_users': set(),
                    'affected_endpoints': set(),
                    'severity_counts': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
                }
            
            group = error_groups[key]
            group['count'] += 1
            
            # Track timing
            timestamp = error.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if group['first_seen'] is None or dt < group['first_seen']:
                        group['first_seen'] = dt
                    if group['last_seen'] is None or dt > group['last_seen']:
                        group['last_seen'] = dt
                except (ValueError, AttributeError):
                    pass
            
            # Track affected users and endpoints
            user_id = error.get('user_id')
            if user_id:
                group['affected_users'].add(user_id)
            
            endpoint = error.get('endpoint') or error.get('url')
            if endpoint:
                group['affected_endpoints'].add(endpoint)
            
            # Track severity
            severity = error.get('severity', 'medium').lower()
            if severity in group['severity_counts']:
                group['severity_counts'][severity] += 1
            else:
                group['severity_counts']['medium'] += 1
        
        # Convert sets to counts and format data
        top_errors = []
        for key, group in error_groups.items():
            error_info = {
                'error_type': group['error_type'],
                'message': group['message'],
                'count': group['count'],
                'affected_users_count': len(group['affected_users']),
                'affected_endpoints_count': len(group['affected_endpoints']),
                'severity_distribution': dict(group['severity_counts']),
                'first_seen': group['first_seen'].isoformat() if group['first_seen'] else None,
                'last_seen': group['last_seen'].isoformat() if group['last_seen'] else None,
                'frequency_score': group['count']  # Can be enhanced with recency weighting
            }
            
            # Calculate impact score (frequency + user impact + severity)
            impact_score = (
                group['count'] * 1.0 +  # Base frequency
                len(group['affected_users']) * 2.0 +  # User impact
                group['severity_counts']['critical'] * 10.0 +  # Critical severity
                group['severity_counts']['high'] * 5.0 +  # High severity
                group['severity_counts']['medium'] * 2.0  # Medium severity
            )
            error_info['impact_score'] = impact_score
            
            top_errors.append(error_info)
        
        # Sort by impact score (descending) and return top 10
        top_errors.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return top_errors[:10]
