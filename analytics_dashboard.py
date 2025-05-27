"""
Analytics Dashboard module for LangGraph 101 project.

This module provides analytics and monitoring features for the LangGraph project,
including usage statistics, error tracking, and performance metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime, timedelta
import time
import sklearn
from sklearn.ensemble import IsolationForest

# Import local modules
from error_handling import ErrorCategory

# Define the path for analytics storage
ANALYTICS_PATH = os.path.join(os.path.dirname(__file__), "analytics_data")
os.makedirs(ANALYTICS_PATH, exist_ok=True)

# Analytics tracking file paths
API_USAGE_FILE = os.path.join(ANALYTICS_PATH, "api_usage.json")
ERROR_TRACKING_FILE = os.path.join(ANALYTICS_PATH, "error_tracking.json")
PERFORMANCE_METRICS_FILE = os.path.join(ANALYTICS_PATH, "performance_metrics.json")
NOTIFICATION_HISTORY_FILE = os.path.join(ANALYTICS_PATH, "notification_history.json")

class AnalyticsTracker:
    """Tracks usage analytics for the application."""

    @staticmethod
    def record_api_call(api_name: str, status: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an API call for analytics.

        Args:
            api_name: Name of the API (e.g., "gemini", "elevenlabs")
            status: Status of the call ("success", "error", "timeout")
            duration_ms: Duration of the call in milliseconds
            metadata: Additional metadata about the call
        """
        # Create data structure if it doesn't exist
        if not os.path.exists(API_USAGE_FILE):
            with open(API_USAGE_FILE, 'w') as f:
                json.dump([], f)

        # Load existing data
        try:
            with open(API_USAGE_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        # Add new entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "api": api_name,
            "status": status,
            "duration_ms": duration_ms
        }

        if metadata:
            entry["metadata"] = metadata

        data.append(entry)

        # Save updated data
        with open(API_USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def record_error(error_message: str, error_category: ErrorCategory, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an error for analytics.

        Args:
            error_message: Error message
            error_category: Category of the error
            source: Source of the error (e.g., function name)
            metadata: Additional metadata about the error
        """
        # Create data structure if it doesn't exist
        if not os.path.exists(ERROR_TRACKING_FILE):
            with open(ERROR_TRACKING_FILE, 'w') as f:
                json.dump([], f)

        # Load existing data
        try:
            with open(ERROR_TRACKING_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        # Add new entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": error_message,
            "category": error_category.value,
            "source": source
        }

        if metadata:
            entry["metadata"] = metadata

        data.append(entry)

        # Save updated data
        with open(ERROR_TRACKING_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def record_performance_metric(component: str, operation: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric.

        Args:
            component: Component being measured (e.g., "script_generation")
            operation: Operation being performed (e.g., "api_call", "processing")
            duration_ms: Duration of the operation in milliseconds
            metadata: Additional metadata about the operation
        """
        # Create data structure if it doesn't exist
        if not os.path.exists(PERFORMANCE_METRICS_FILE):
            with open(PERFORMANCE_METRICS_FILE, 'w') as f:
                json.dump([], f)

        # Load existing data
        try:
            with open(PERFORMANCE_METRICS_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        # Add new entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "duration_ms": duration_ms
        }

        if metadata:
            entry["metadata"] = metadata

        data.append(entry)

        # Save updated data
        with open(PERFORMANCE_METRICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)


class PerformanceMetrics:
    """Track and analyze system performance metrics"""

    def __init__(self):
        self.operation_timings = {}
        self.error_counts = {}
        self.resource_usage = {}
        self.cache_stats = {}

    def record_operation_timing(self, component: str, operation: str, duration_ms: float) -> None:
        key = f"{component}:{operation}"
        if key not in self.operation_timings:
            self.operation_timings[key] = []
        self.operation_timings[key].append(duration_ms)

    def record_error(self, component: str, error_type: str) -> None:
        key = f"{component}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    def record_resource_usage(self, resource: str, usage: float) -> None:
        if resource not in self.resource_usage:
            self.resource_usage[resource] = []
        self.resource_usage[resource].append(usage)

    def record_cache_stats(self, hits: int, misses: int, size: int) -> None:
        self.cache_stats["hits"] = self.cache_stats.get("hits", 0) + hits
        self.cache_stats["misses"] = self.cache_stats.get("misses", 0) + misses
        self.cache_stats["size"] = size

    def get_summary(self) -> dict:
        """Get a summary of all performance metrics"""
        return {
            "operation_timings": {k: {
                "avg": sum(v)/len(v),
                "min": min(v),
                "max": max(v),
                "count": len(v)
            } for k, v in self.operation_timings.items()},
            "error_counts": self.error_counts,
            "resource_usage": {k: {
                "avg": sum(v)/len(v),
                "current": v[-1]
            } for k, v in self.resource_usage.items()},
            "cache_stats": self.cache_stats
        }

# Initialize global metrics tracker
_metrics = PerformanceMetrics()

def get_metrics() -> PerformanceMetrics:
    """Get the global metrics tracker instance"""
    return _metrics


def recommend_next_action(api_usage_df: pd.DataFrame) -> str:
    """Recommend the next action or tool based on recent API usage patterns."""
    if api_usage_df.empty:
        return "Try generating a new script or uploading a document for analysis."
    most_used = api_usage_df['api_name'].value_counts().idxmax()
    if most_used == 'gemini':
        return "You frequently use Gemini. Try ElevenLabs TTS for audio or Stability AI for images."
    elif most_used == 'elevenlabs':
        return "You use TTS often. Try generating a script with Gemini or adding a thumbnail with DALL-E."
    elif most_used == 'stabilityai' or most_used == 'dalle':
        return "You generate images often. Try script or audio generation for a complete workflow."
    else:
        return f"You use {most_used} often. Explore other tools for a full content pipeline."


def render_recommendation():
    st.header("AI Recommendation")
    api_data = load_analytics_data(API_USAGE_FILE)
    if not api_data:
        st.info("No API usage data available yet. Start using the application to generate analytics.")
        return
    df = pd.DataFrame(api_data)
    if df.empty or 'api' not in df:
        st.info("Insufficient API usage data for recommendations.")
        return
    def recommend_next_action(api_usage_df: pd.DataFrame) -> str:
        if api_usage_df.empty:
            return "Try generating a new script or uploading a document for analysis."
        most_used = api_usage_df['api'].value_counts().idxmax()
        if most_used == 'gemini':
            return "You frequently use Gemini. Try ElevenLabs TTS for audio or Stability AI for images."
        elif most_used == 'elevenlabs':
            return "You use TTS often. Try generating a script with Gemini or adding a thumbnail with DALL-E."
        elif most_used == 'stabilityai' or most_used == 'dalle':
            return "You generate images often. Try script or audio generation for a complete workflow."
        else:
            return f"You use {most_used} often. Explore other tools for a full content pipeline."
    rec = recommend_next_action(df)
    st.info(f"Recommended next action: {rec}")


def render_analytics_dashboard():
    """Render the analytics dashboard in Streamlit."""
    st.title("📊 LangGraph Analytics Dashboard")

    # Set up tabs for different analytics views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 API Usage", "❌ Error Tracking", "⚡ Performance Metrics", "📋 System Health", "🔍 Error Anomaly Detection", "🤖 AI Recommendation"
    ])

    # API Usage tab
    with tab1:
        render_api_usage_analytics()

    # Error Tracking tab
    with tab2:
        render_error_tracking_analytics()

    # Performance Metrics tab
    with tab3:
        render_performance_metrics_analytics()

    # System Health tab
    with tab4:
        render_system_health_analytics()

    # Error Anomaly Detection tab
    with tab5:
        render_error_anomaly_detection()

    # AI Recommendation tab
    with tab6:
        render_recommendation()

    # Add error handling for each tab
    try:
        render_api_usage_analytics()
    except Exception as e:
        st.error(f"Failed to load API Usage Analytics: {e}")

    try:
        render_error_tracking_analytics()
    except Exception as e:
        st.error(f"Failed to load Error Tracking Analytics: {e}")

    try:
        render_performance_metrics_analytics()
    except Exception as e:
        st.error(f"Failed to load Performance Metrics Analytics: {e}")

    try:
        render_system_health_analytics()
    except Exception as e:
        st.error(f"Failed to load System Health Analytics: {e}")


def render_api_usage_analytics():
    """Render API usage analytics."""
    st.header("API Usage Analytics")

    # Load API usage data
    api_data = load_analytics_data(API_USAGE_FILE)

    if not api_data:
        st.info("No API usage data available yet. Start using the application to generate analytics.")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(api_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date,
            max_value=datetime.now()
        )

    # Apply filters
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    if filtered_df.empty:
        st.info("No data available for the selected time range.")
        return

    # API call counts by service
    st.subheader("API Usage by Service")
    api_counts = filtered_df['api'].value_counts().reset_index()
    api_counts.columns = ['API', 'Calls']

    fig = px.bar(
        api_counts,
        x='API',
        y='Calls',
        color='API',
        title='API Call Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

    # API status distribution
    st.subheader("API Call Status Distribution")

    status_counts = filtered_df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']

    fig = px.pie(
        status_counts,
        names='Status',
        values='Count',
        title='API Call Outcomes',
        color='Status',
        color_discrete_map={
            'success': 'green',
            'error': 'red',
            'timeout': 'orange'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # API call latency
    st.subheader("API Latency Analysis")

    latency_by_api = filtered_df.groupby('api')['duration_ms'].mean().reset_index()
    latency_by_api.columns = ['API', 'Avg Latency (ms)']

    fig = px.bar(
        latency_by_api,
        x='API',
        y='Avg Latency (ms)',
        color='API',
        title='Average API Latency'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily API usage trend
    st.subheader("Daily API Usage Trend")

    daily_counts = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'api']).size().reset_index()
    daily_counts.columns = ['Date', 'API', 'Calls']

    fig = px.line(
        daily_counts,
        x='Date',
        y='Calls',
        color='API',
        title='Daily API Usage'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data view
    with st.expander("View Raw API Usage Data"):
        st.dataframe(filtered_df)

    # AI Recommendation
    st.subheader("AI Recommendation")
    def recommend_next_action(api_usage_df: pd.DataFrame) -> str:
        if api_usage_df.empty:
            return "Try generating a new script or uploading a document for analysis."
        most_used = api_usage_df['api'].value_counts().idxmax()
        if most_used == 'gemini':
            return "You frequently use Gemini. Try ElevenLabs TTS for audio or Stability AI for images."
        elif most_used == 'elevenlabs':
            return "You use TTS often. Try generating a script with Gemini or adding a thumbnail with DALL-E."
        elif most_used == 'stabilityai' or most_used == 'dalle':
            return "You generate images often. Try script or audio generation for a complete workflow."
        else:
            return f"You use {most_used} often. Explore other tools for a full content pipeline."
    rec = recommend_next_action(filtered_df)
    st.info(f"Recommended next action: {rec}")


def render_error_tracking_analytics():
    """Render error tracking analytics."""
    st.header("Error Tracking Analytics")

    # Load error data
    error_data = load_analytics_data(ERROR_TRACKING_FILE)

    if not error_data:
        st.info("No error data available. That's a good thing!")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(error_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7),
            max_value=datetime.now(),
            key="error_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date,
            max_value=datetime.now(),
            key="error_end_date"
        )

    # Apply filters
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    if filtered_df.empty:
        st.info("No errors recorded in the selected time range. Excellent!")
        return

    # Error category distribution
    st.subheader("Error Category Distribution")

    category_counts = filtered_df['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

    fig = px.pie(
        category_counts,
        names='Category',
        values='Count',
        title='Error Categories',
        color='Category'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error sources
    st.subheader("Error Sources")

    source_counts = filtered_df['source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']

    fig = px.bar(
        source_counts,
        x='Source',
        y='Count',
        color='Source',
        title='Error Sources'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error trend over time
    st.subheader("Error Trend")

    daily_errors = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'category']).size().reset_index()
    daily_errors.columns = ['Date', 'Category', 'Count']

    fig = px.line(
        daily_errors,
        x='Date',
        y='Count',
        color='Category',
        title='Daily Error Trend'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recent errors table
    st.subheader("Recent Errors")
    recent_errors = filtered_df.sort_values('timestamp', ascending=False).head(10)

    # Format the table for better readability
    display_cols = ['timestamp', 'category', 'source', 'message']
    display_df = recent_errors[display_cols].rename(columns={
        'timestamp': 'Time',
        'category': 'Category',
        'source': 'Source',
        'message': 'Error Message'
    })

    st.table(display_df)

    # Raw data view
    with st.expander("View All Error Data"):
        st.dataframe(filtered_df)


def render_performance_metrics_analytics():
    """Render performance metrics analytics."""
    st.header("Performance Metrics Analytics")

    # Load performance data
    perf_data = load_analytics_data(PERFORMANCE_METRICS_FILE)

    if not perf_data:
        st.info("No performance data available yet. Start using the application to generate analytics.")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(perf_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7),
            max_value=datetime.now(),
            key="perf_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=start_date,
            max_value=datetime.now(),
            key="perf_end_date"
        )

    # Apply filters
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]

    if filtered_df.empty:
        st.info("No performance data available for the selected time range.")
        return

    # Component performance comparison
    st.subheader("Component Performance Comparison")

    component_perf = filtered_df.groupby('component')['duration_ms'].mean().reset_index()
    component_perf.columns = ['Component', 'Avg Duration (ms)']
    component_perf = component_perf.sort_values('Avg Duration (ms)', ascending=False)

    fig = px.bar(
        component_perf,
        x='Component',
        y='Avg Duration (ms)',
        color='Component',
        title='Average Component Performance'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Operation performance within components
    st.subheader("Operation Performance")

    # Let user select a component
    components = sorted(filtered_df['component'].unique())
    if components:
        selected_component = st.selectbox("Select Component", components)

        # Filter for selected component
        comp_df = filtered_df[filtered_df['component'] == selected_component]

        op_perf = comp_df.groupby('operation')['duration_ms'].mean().reset_index()
        op_perf.columns = ['Operation', 'Avg Duration (ms)']

        fig = px.bar(
            op_perf,
            x='Operation',
            y='Avg Duration (ms)',
            color='Operation',
            title=f'Operation Performance for {selected_component}'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Performance trend over time
    st.subheader("Performance Trend")

    daily_perf = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'component'])['duration_ms'].mean().reset_index()
    daily_perf.columns = ['Date', 'Component', 'Avg Duration (ms)']

    fig = px.line(
        daily_perf,
        x='Date',
        y='Avg Duration (ms)',
        color='Component',
        title='Daily Performance Trend'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data view
    with st.expander("View Raw Performance Data"):
        st.dataframe(filtered_df)


def render_system_health_analytics():
    """Render system health analytics dashboard."""
    st.header("System Health Analytics")

    # API Health Status
    st.subheader("API Health Status")

    # Load API usage data from the last 24 hours
    api_data = load_analytics_data(API_USAGE_FILE)

    if not api_data:
        st.info("No API usage data available yet.")
    else:
        df = pd.DataFrame(api_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter for last 24 hours
        last_24h = datetime.now() - timedelta(hours=24)
        recent_df = df[df['timestamp'] >= last_24h]

        if recent_df.empty:
            st.info("No API calls in the last 24 hours.")
        else:
            # Calculate health metrics for each API
            api_health = []

            for api_name in recent_df['api'].unique():
                api_df = recent_df[recent_df['api'] == api_name]
                total_calls = len(api_df)
                success_calls = len(api_df[api_df['status'] == 'success'])
                error_calls = total_calls - success_calls
                success_rate = (success_calls / total_calls) * 100 if total_calls > 0 else 0

                # Determine status based on success rate
                if success_rate >= 98:
                    status = "Healthy"
                    color = "green"
                elif success_rate >= 90:
                    status = "Warning"
                    color = "orange"
                else:
                    status = "Critical"
                    color = "red"

                avg_latency = api_df['duration_ms'].mean()

                api_health.append({
                    "API": api_name,
                    "Status": status,
                    "Success Rate": f"{success_rate:.1f}%",
                    "Total Calls": total_calls,
                    "Errors": error_calls,
                    "Avg Latency (ms)": f"{avg_latency:.1f}",
                    "Color": color
                })

            # Display as a styled dataframe
            api_health_df = pd.DataFrame(api_health)

            # Use colored indicators
            for i, row in enumerate(api_health_df.itertuples()):
                cols = st.columns([3, 1, 2, 1, 1, 2])
                with cols[0]:
                    st.write(f"**{row.API}**")
                with cols[1]:
                    st.write(f":{row.Color}[{row.Status}]")
                with cols[2]:
                    st.write(f"{row.Success_Rate}")
                with cols[3]:
                    st.write(f"{row.Total_Calls}")
                with cols[4]:
                    st.write(f"{row.Errors}")
                with cols[5]:
                    st.write(f"{row.Avg_Latency_ms}")

            # Add header row
            st.write("---")
            cols = st.columns([3, 1, 2, 1, 1, 2])
            with cols[0]:
                st.write("**API**")
            with cols[1]:
                st.write("**Status**")
            with cols[2]:
                st.write("**Success Rate**")
            with cols[3]:
                st.write("**Calls**")
            with cols[4]:
                st.write("**Errors**")
            with cols[5]:
                st.write("**Latency**")

    # Error Rate Over Time
    st.subheader("Error Rate Over Time")

    # Combine API and Error data
    api_data = load_analytics_data(API_USAGE_FILE)
    error_data = load_analytics_data(ERROR_TRACKING_FILE)

    if api_data and error_data:
        api_df = pd.DataFrame(api_data)
        api_df['timestamp'] = pd.to_datetime(api_df['timestamp'])
        api_df['date'] = api_df['timestamp'].dt.date

        error_df = pd.DataFrame(error_data)
        error_df['timestamp'] = pd.to_datetime(error_df['timestamp'])
        error_df['date'] = error_df['timestamp'].dt.date

        # Group by date and count
        api_counts = api_df.groupby('date').size().reset_index(name='api_calls')
        error_counts = error_df.groupby('date').size().reset_index(name='errors')

        # Merge the data
        merged_df = pd.merge(api_counts, error_counts, on='date', how='outer').fillna(0)
        merged_df['error_rate'] = (merged_df['errors'] / merged_df['api_calls']) * 100
        merged_df = merged_df.sort_values('date')

        # Create the chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged_df['date'],
            y=merged_df['error_rate'],
            mode='lines+markers',
            name='Error Rate (%)',
            line=dict(color='red')
        ))

        fig.update_layout(
            title='Daily Error Rate',
            xaxis_title='Date',
            yaxis_title='Error Rate (%)',
            yaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to calculate error rates.")

    # System Resource Monitoring (simulated)
    st.subheader("System Resource Usage")

    # In a real implementation, this would connect to actual system metrics
    # For this example, we'll simulate some system metrics

    # CPU Usage
    cpu_usage = {
        "timestamp": [datetime.now() - timedelta(minutes=i*10) for i in range(24, 0, -1)],
        "usage": [30 + 15 * np.sin(i/5) + np.random.randint(-5, 5) for i in range(24)]
    }

    cpu_df = pd.DataFrame(cpu_usage)

    fig = px.line(
        cpu_df,
        x='timestamp',
        y='usage',
        title='CPU Usage (%)',
        labels={'timestamp': 'Time', 'usage': 'Usage (%)'}
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # Memory Usage
    col1, col2 = st.columns(2)

    with col1:
        # Memory gauge
        memory_usage = 65  # Simulated percentage

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = memory_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Memory Usage"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Disk usage
        disk_usage = 42  # Simulated percentage

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = disk_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Disk Usage"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


def render_error_anomaly_detection():
    """ML-powered anomaly detection for error logs using Isolation Forest."""
    st.header("ML-Powered Error Anomaly Detection")
    error_data = load_analytics_data(ERROR_TRACKING_FILE)
    if not error_data:
        st.info("No error data available for anomaly detection.")
        return
    df = pd.DataFrame(error_data)
    if df.empty or 'timestamp' not in df or 'category' not in df:
        st.info("Insufficient error data for anomaly detection.")
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['category_code'] = df['category'].astype('category').cat.codes
    df['source_code'] = df['source'].astype('category').cat.codes
    # Feature: time (as ordinal), category, source
    X = np.stack([
        df['timestamp'].map(datetime.toordinal),
        df['category_code'],
        df['source_code']
    ], axis=1)
    # Fit Isolation Forest
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(X)
    df['anomaly'] = preds == -1
    st.subheader("Anomaly Detection Results")
    st.write(f"Detected {df['anomaly'].sum()} anomalies out of {len(df)} error records.")
    # Show anomalies
    anomalies = df[df['anomaly']]
    if not anomalies.empty:
        st.dataframe(anomalies[['timestamp', 'category', 'source', 'message']])
    else:
        st.info("No anomalies detected in error logs.")
    # Optional: plot anomalies over time
    st.subheader("Error Anomalies Over Time")
    fig = px.scatter(df, x='timestamp', y='category', color='anomaly',
                    title='Error Anomalies Over Time',
                    color_discrete_map={True: 'red', False: 'blue'})
    st.plotly_chart(fig, use_container_width=True)


def load_analytics_data(file_path: str) -> List[Dict[str, Any]]:
    """Load analytics data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of data entries
    """
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        return []
