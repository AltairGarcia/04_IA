"""
Advanced Analytics Dashboard UI Components for LangGraph 101

This module provides advanced Streamlit UI components for the enhanced analytics system,
including real-time metrics, interactive charts, and comprehensive performance monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from analytics.analytics_logger import AnalyticsLogger
from analytics.real_time_analytics import RealTimeAnalytics
from analytics.user_behavior_analyzer import UserBehaviorAnalyzer
from analytics.performance_tracker import PerformanceTracker
from analytics.custom_reports_generator import CustomReportsGenerator

class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard with real-time capabilities."""
    
    def __init__(self):
        self.analytics_logger = AnalyticsLogger()
        self.real_time_analytics = RealTimeAnalytics()
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.reports_generator = CustomReportsGenerator()
    
    def render_main_dashboard(self):
        """Render the main analytics dashboard."""
        st.title("📊 Advanced Analytics Dashboard")
        st.caption("Real-time insights and comprehensive performance monitoring")
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Real-Time", "🎯 Performance", "👥 User Behavior", 
            "📊 Reports", "🔍 Model Analytics", "⚙️ System Health"
        ])
        
        with tab1:
            self.render_real_time_dashboard()
        
        with tab2:
            self.render_performance_dashboard()
        
        with tab3:
            self.render_user_behavior_dashboard()
        
        with tab4:
            self.render_custom_reports_dashboard()
        
        with tab5:
            self.render_model_analytics_dashboard()
        
        with tab6:
            self.render_system_health_dashboard()
    
    def render_real_time_dashboard(self):
        """Render real-time analytics dashboard."""
        st.header("📈 Real-Time Analytics")
        
        # Auto-refresh controls
        col1, col2 = st.columns([3, 1])
        with col1:
            auto_refresh = st.toggle("🔄 Auto-refresh", value=True)
        with col2:
            refresh_interval = st.selectbox("Interval", [5, 10, 30, 60], index=2)
        
        # Real-time metrics overview
        st.subheader("⚡ Live Metrics")
        
        # Get real-time data
        streaming_data = self.real_time_analytics.get_streaming_data()
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_users = streaming_data.get('active_users', 0)
            st.metric("Active Users", current_users, delta=f"+{streaming_data.get('user_delta', 0)}")
        
        with col2:
            api_calls_rate = streaming_data.get('api_calls_per_minute', 0)
            st.metric("API Calls/min", f"{api_calls_rate:.1f}", 
                     delta=f"{streaming_data.get('api_calls_delta', 0):+.1f}")
        
        with col3:
            avg_response_time = streaming_data.get('avg_response_time_ms', 0)
            response_delta = streaming_data.get('response_time_delta', 0)
            delta_color = "inverse" if response_delta > 0 else "normal"
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms", 
                     delta=f"{response_delta:+.0f}ms", delta_color=delta_color)
        
        with col4:
            error_rate = streaming_data.get('error_rate_percent', 0)
            error_delta = streaming_data.get('error_rate_delta', 0)
            delta_color = "inverse" if error_delta > 0 else "normal"
            st.metric("Error Rate", f"{error_rate:.1f}%", 
                     delta=f"{error_delta:+.1f}%", delta_color=delta_color)
        
        # Real-time charts
        st.subheader("📊 Live Charts")
        
        # API calls timeline
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### API Calls (Last Hour)")
            timeline_data = streaming_data.get('api_calls_timeline', [])
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                fig = px.line(df, x='timestamp', y='count', 
                             title="API Calls Over Time")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timeline data available")
        
        with col2:
            st.markdown("#### Response Times")
            response_data = streaming_data.get('response_times', [])
            if response_data:
                df = pd.DataFrame(response_data)
                fig = px.line(df, x='timestamp', y='response_time_ms',
                             title="Response Times")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No response time data available")
        
        # Provider usage real-time
        st.subheader("🤖 Provider Usage (Real-Time)")
        provider_usage = streaming_data.get('provider_usage', {})
        
        if provider_usage:
            providers_df = pd.DataFrame([
                {'Provider': provider, 'Usage': usage, 'Status': 'Active'}
                for provider, usage in provider_usage.items()
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(providers_df, values='Usage', names='Provider',
                           title="Provider Usage Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(providers_df, x='Provider', y='Usage',
                           title="Provider Call Counts")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No provider usage data available")
        
        # Auto-refresh implementation
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def render_performance_dashboard(self):
        """Render performance analytics dashboard."""
        st.header("🎯 Performance Analytics")
        
        # Time range selector
        col1, col2 = st.columns([2, 1])
        with col1:
            time_range = st.selectbox("Time Range", 
                                    ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])
        with col2:
            st.button("🔄 Refresh Data")
        
        # Get performance data
        hours_map = {"Last Hour": 1, "Last 24 Hours": 24, "Last 7 Days": 168, "Last 30 Days": 720}
        hours = hours_map[time_range]
        
        performance_data = self.performance_tracker.get_performance_report(hours=hours)
        
        # Performance overview
        st.subheader("📊 Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_requests = performance_data.get('total_requests', 0)
            st.metric("Total Requests", f"{total_requests:,}")
        
        with col2:
            avg_response = performance_data.get('avg_response_time_ms', 0)
            st.metric("Avg Response Time", f"{avg_response:.1f}ms")
        
        with col3:
            p95_response = performance_data.get('p95_response_time_ms', 0)
            st.metric("95th Percentile", f"{p95_response:.1f}ms")
        
        with col4:
            success_rate = performance_data.get('success_rate_percent', 0)
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        trends_data = performance_data.get('trends', [])
        if trends_data:
            df = pd.DataFrame(trends_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Response Time', 'Request Volume', 'Error Rate', 'Throughput'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Response time trend
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['response_time_ms'], 
                          name='Response Time', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Request volume
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['request_count'], 
                          name='Requests', line=dict(color='green')),
                row=1, col=2
            )
            
            # Error rate
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['error_rate'], 
                          name='Error Rate', line=dict(color='red')),
                row=2, col=1
            )
            
            # Throughput
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['throughput'], 
                          name='Throughput', line=dict(color='purple')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance trend data available")
        
        # Model performance comparison
        st.subheader("🤖 Model Performance Comparison")
        
        model_performance = performance_data.get('model_performance', {})
        if model_performance:
            models_df = pd.DataFrame([
                {
                    'Model': model,
                    'Avg Response Time (ms)': stats['avg_response_time_ms'],
                    'Success Rate (%)': stats['success_rate_percent'],
                    'Total Requests': stats['total_requests'],
                    'Cost per Request ($)': stats['avg_cost_per_request']
                }
                for model, stats in model_performance.items()
            ])
            
            # Performance comparison chart
            fig = px.scatter(models_df, x='Avg Response Time (ms)', y='Success Rate (%)',
                           size='Total Requests', hover_name='Model',
                           title="Model Performance: Response Time vs Success Rate")
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(models_df, use_container_width=True)
        else:
            st.info("No model performance data available")
    
    def render_user_behavior_dashboard(self):
        """Render user behavior analytics dashboard."""
        st.header("👥 User Behavior Analytics")
        
        # Get behavior insights
        behavior_insights = self.behavior_analyzer.get_behavior_insights()
        
        # User engagement overview
        st.subheader("📈 User Engagement Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = behavior_insights.get('total_users', 0)
            st.metric("Total Users", total_users)
        
        with col2:
            active_users = behavior_insights.get('active_users', 0)
            st.metric("Active Users", active_users)
        
        with col3:
            avg_session_duration = behavior_insights.get('avg_session_duration_minutes', 0)
            st.metric("Avg Session", f"{avg_session_duration:.1f}m")
        
        with col4:
            engagement_score = behavior_insights.get('overall_engagement_score', 0)
            st.metric("Engagement Score", f"{engagement_score:.1f}/100")
        
        # User segmentation
        st.subheader("🎯 User Segmentation")
        
        user_segments = behavior_insights.get('user_segments', {})
        if user_segments:
            segments_df = pd.DataFrame([
                {'Segment': segment, 'Count': data['count'], 'Percentage': data['percentage']}
                for segment, data in user_segments.items()
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(segments_df, values='Count', names='Segment',
                           title="User Segments Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(segments_df, x='Segment', y='Count',
                           title="Users by Segment")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user segmentation data available")
        
        # Feature usage analysis
        st.subheader("🔧 Feature Usage Analysis")
        
        feature_usage = behavior_insights.get('feature_usage', {})
        if feature_usage:
            features_df = pd.DataFrame([
                {'Feature': feature, 'Usage Count': data['usage_count'], 
                 'Unique Users': data['unique_users']}
                for feature, data in feature_usage.items()
            ])
            
            fig = px.bar(features_df, x='Feature', y='Usage Count',
                        title="Feature Usage Statistics")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(features_df, use_container_width=True)
        else:
            st.info("No feature usage data available")
        
        # User activity heatmap
        st.subheader("🔥 Activity Heatmap")
        
        activity_data = behavior_insights.get('activity_heatmap', [])
        if activity_data:
            heatmap_df = pd.DataFrame(activity_data)
            
            # Create pivot table for heatmap
            pivot_df = heatmap_df.pivot(index='hour', columns='day', values='activity_count')
            
            fig = px.imshow(pivot_df, 
                          title="User Activity Heatmap (Hour vs Day of Week)",
                          labels=dict(x="Day of Week", y="Hour of Day", color="Activity Count"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity heatmap data available")
        
        # User journey analysis
        st.subheader("🛣️ User Journey Analysis")
        
        journey_data = behavior_insights.get('user_journeys', [])
        if journey_data:
            st.markdown("**Most Common User Paths:**")
            for i, journey in enumerate(journey_data[:5], 1):
                st.write(f"{i}. {journey['path']} ({journey['frequency']} users)")
        else:
            st.info("No user journey data available")
    
    def render_custom_reports_dashboard(self):
        """Render custom reports dashboard."""
        st.header("📊 Custom Reports")
        
        # Report generator interface
        st.subheader("🛠️ Generate Custom Report")
        
        with st.form("custom_report_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox("Report Type", [
                    "Performance Summary", "User Behavior Report", 
                    "Model Usage Report", "Error Analysis Report",
                    "Cost Analysis Report", "Custom Query Report"
                ])
                
                time_range = st.selectbox("Time Range", [
                    "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"
                ])
                
                if time_range == "Custom Range":
                    start_date = st.date_input("Start Date")
                    end_date = st.date_input("End Date")
            
            with col2:
                export_format = st.selectbox("Export Format", ["HTML", "CSV", "JSON"])
                
                include_charts = st.checkbox("Include Charts", value=True)
                include_raw_data = st.checkbox("Include Raw Data", value=False)
                
                if report_type == "Custom Query Report":
                    custom_filters = st.text_area("Custom Filters (JSON)", 
                                                 placeholder='{"provider": "openai", "status": "success"}')
            
            generate_report = st.form_submit_button("📊 Generate Report")
        
        if generate_report:
            with st.spinner("Generating custom report..."):
                try:
                    # Prepare report parameters
                    params = {
                        "report_type": report_type,
                        "time_range": time_range,
                        "export_format": export_format,
                        "include_charts": include_charts,
                        "include_raw_data": include_raw_data
                    }
                    
                    if time_range == "Custom Range":
                        params["start_date"] = start_date.isoformat()
                        params["end_date"] = end_date.isoformat()
                    
                    if report_type == "Custom Query Report" and custom_filters:
                        params["custom_filters"] = json.loads(custom_filters)
                    
                    # Generate report
                    report_data = self.reports_generator.generate_report(params)
                    
                    if report_data:
                        st.success("✅ Report generated successfully!")
                        
                        # Display report preview
                        st.subheader("📋 Report Preview")
                        
                        # Show summary statistics
                        summary = report_data.get('summary', {})
                        if summary:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Records", summary.get('total_records', 0))
                            with col2:
                                st.metric("Time Period", summary.get('time_period', 'N/A'))
                            with col3:
                                st.metric("Data Sources", summary.get('data_sources', 0))
                            with col4:
                                st.metric("Report Size", f"{summary.get('size_mb', 0):.1f} MB")
                        
                        # Show data preview
                        data_preview = report_data.get('data_preview', [])
                        if data_preview:
                            st.subheader("📊 Data Preview")
                            preview_df = pd.DataFrame(data_preview)
                            st.dataframe(preview_df.head(10), use_container_width=True)
                        
                        # Download button
                        if export_format == "HTML":
                            report_html = report_data.get('html_content', '')
                            st.download_button(
                                label="📥 Download HTML Report",
                                data=report_html,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html"
                            )
                        elif export_format == "CSV":
                            report_csv = report_data.get('csv_content', '')
                            st.download_button(
                                label="📥 Download CSV Report",
                                data=report_csv,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        elif export_format == "JSON":
                            report_json = json.dumps(report_data.get('json_content', {}), indent=2)
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=report_json,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    else:
                        st.error("❌ Failed to generate report")
                        
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
        
        # Predefined reports
        st.subheader("📄 Predefined Reports")
        
        predefined_reports = [
            {"name": "Daily Performance Summary", "description": "Daily overview of system performance and usage"},
            {"name": "Weekly User Behavior Report", "description": "Weekly analysis of user behavior patterns"},
            {"name": "Monthly Cost Analysis", "description": "Monthly breakdown of API usage costs"},
            {"name": "Model Comparison Report", "description": "Performance comparison across different AI models"},
            {"name": "Error Trend Analysis", "description": "Analysis of error patterns and trends"}
        ]
        
        for report in predefined_reports:
            with st.expander(f"📊 {report['name']}"):
                st.write(report['description'])
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"Generate", key=f"gen_{report['name']}"):
                        st.info(f"Generating {report['name']}...")
    
    def render_model_analytics_dashboard(self):
        """Render model-specific analytics dashboard."""
        st.header("🔍 Model Analytics")
        
        # Model performance overview
        st.subheader("🤖 Model Performance Overview")
        
        model_data = self.analytics_logger.get_model_analytics()
        
        if model_data:
            models_df = pd.DataFrame([
                {
                    'Provider': data['provider'],
                    'Model': data['model'],
                    'Total Requests': data['total_requests'],
                    'Avg Response Time (ms)': data['avg_response_time_ms'],
                    'Success Rate (%)': data['success_rate_percent'],
                    'Total Cost ($)': data['total_cost'],
                    'Avg Cost per Request ($)': data['avg_cost_per_request']
                }
                for data in model_data
            ])
            
            # Model performance table
            st.dataframe(models_df, use_container_width=True)
            
            # Model comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(models_df, x='Avg Response Time (ms)', y='Success Rate (%)',
                               color='Provider', size='Total Requests', hover_name='Model',
                               title="Model Performance: Response Time vs Success Rate")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(models_df, x='Model', y='Total Cost ($)',
                           color='Provider', title="Total Cost by Model")
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model analytics data available")
        
        # Model usage trends
        st.subheader("📈 Model Usage Trends")
        
        usage_trends = self.analytics_logger.get_model_usage_trends()
        if usage_trends:
            trends_df = pd.DataFrame(usage_trends)
            
            fig = px.line(trends_df, x='date', y='usage_count', color='model',
                         title="Model Usage Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model usage trend data available")
        
        # Model recommendations
        st.subheader("💡 Model Recommendations")
        
        recommendations = self.performance_tracker.get_model_recommendations()
        if recommendations:
            st.markdown("**Based on recent performance data:**")
            
            for rec in recommendations:
                recommendation_type = rec.get('type', 'info')
                message = rec.get('message', '')
                
                if recommendation_type == 'cost_optimization':
                    st.success(f"💰 Cost Optimization: {message}")
                elif recommendation_type == 'performance_improvement':
                    st.info(f"⚡ Performance: {message}")
                elif recommendation_type == 'reliability_warning':
                    st.warning(f"⚠️ Reliability: {message}")
                else:
                    st.info(f"💡 {message}")
        else:
            st.info("No model recommendations available")
    
    def render_system_health_dashboard(self):
        """Render system health dashboard."""
        st.header("⚙️ System Health")
        
        # System metrics overview
        st.subheader("📊 System Status")
        
        system_metrics = self.real_time_analytics.get_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = system_metrics.get('cpu_usage_percent', 0)
            delta_color = "inverse" if cpu_usage > 80 else "normal"
            st.metric("CPU Usage", f"{cpu_usage:.1f}%", delta_color=delta_color)
        
        with col2:
            memory_usage = system_metrics.get('memory_usage_percent', 0)
            delta_color = "inverse" if memory_usage > 85 else "normal"
            st.metric("Memory Usage", f"{memory_usage:.1f}%", delta_color=delta_color)
        
        with col3:
            disk_usage = system_metrics.get('disk_usage_percent', 0)
            delta_color = "inverse" if disk_usage > 90 else "normal"
            st.metric("Disk Usage", f"{disk_usage:.1f}%", delta_color=delta_color)
        
        with col4:
            active_connections = system_metrics.get('active_connections', 0)
            st.metric("Active Connections", active_connections)
        
        # System alerts
        st.subheader("🚨 System Alerts")
        
        alerts = self.analytics_logger.get_recent_alerts()
        if alerts:
            for alert in alerts[-5:]:  # Show last 5 alerts
                alert_level = alert.get('level', 'info')
                message = alert.get('message', '')
                timestamp = alert.get('timestamp', '')
                
                if alert_level == 'critical':
                    st.error(f"🔴 {timestamp}: {message}")
                elif alert_level == 'warning':
                    st.warning(f"🟡 {timestamp}: {message}")
                else:
                    st.info(f"🔵 {timestamp}: {message}")
        else:
            st.success("✅ No recent alerts")
        
        # Database status
        st.subheader("🗄️ Database Status")
        
        db_status = self.analytics_logger.get_database_status()
        if db_status:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{db_status.get('total_records', 0):,}")
            
            with col2:
                db_size_mb = db_status.get('database_size_mb', 0)
                st.metric("Database Size", f"{db_size_mb:.1f} MB")
            
            with col3:
                last_backup = db_status.get('last_backup', 'Never')
                st.metric("Last Backup", last_backup)
        else:
            st.info("Database status information not available")
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        performance_history = system_metrics.get('performance_history', [])
        if performance_history:
            history_df = pd.DataFrame(performance_history)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time', 'Request Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['cpu_usage'],
                          name='CPU', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['memory_usage'],
                          name='Memory', line=dict(color='green')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['response_time'],
                          name='Response Time', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['request_rate'],
                          name='Request Rate', line=dict(color='purple')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance history data available")

# Utility functions for dashboard components
def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def get_status_color(value: float, thresholds: Dict[str, float]) -> str:
    """Get status color based on value and thresholds."""
    if value >= thresholds.get('critical', 90):
        return 'red'
    elif value >= thresholds.get('warning', 70):
        return 'orange'
    else:
        return 'green'
