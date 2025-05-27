"""
Performance monitoring dashboard for LangGraph 101.
Visualizes system performance metrics, health indicators, and usage statistics.
"""
import os
import json
import time
import glob
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st
import plotly.graph_objects as go
from performance_metrics import get_metrics
from datetime import datetime, timedelta

# Import local modules
from system_initialization import (
    ANALYTICS_DIR,
    ERROR_LOGS_DIR,
    check_system_status,
    get_storage
)
from logging_config import get_contextual_logger

# Configure logger
logger = get_contextual_logger(
    __name__,
    module="performance_dashboard",
    component_type="analytics"
)

class PerformanceDashboard:
    """Dashboard for visualizing system performance metrics."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the performance dashboard.

        Args:
            output_dir: Optional directory to save dashboard visualizations
        """
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "dashboard_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Ensure we have storage access
        self.storage = get_storage()

        # Timestamp for this dashboard generation
        self.timestamp = datetime.datetime.now()
        self.timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")

    def generate_dashboard(self) -> Dict[str, str]:
        """Generate a complete performance dashboard with all charts.

        Returns:
            Dictionary mapping chart type to output file path
        """
        logger.info("Generating performance dashboard")
        output_files = {}

        try:
            # System health visualization
            health_chart = self.generate_health_chart()
            if health_chart:
                output_files['health'] = health_chart

            # API usage trends
            api_chart = self.generate_api_usage_chart()
            if api_chart:
                output_files['api_usage'] = api_chart

            # Error rate analysis
            error_chart = self.generate_error_rate_chart()
            if error_chart:
                output_files['errors'] = error_chart

            # Performance metrics
            perf_chart = self.generate_performance_chart()
            if perf_chart:
                output_files['performance'] = perf_chart

            # Generate summary report
            summary = self.generate_summary_report(output_files)
            output_files['summary'] = summary

            logger.info(f"Dashboard generated successfully with {len(output_files)} charts")
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {str(e)}")

        return output_files

    def generate_health_chart(self) -> Optional[str]:
        """Generate system health visualization.

        Returns:
            Path to saved chart or None if generation failed
        """
        try:
            # Get current system status
            system_status = check_system_status()

            # Setup the figure
            plt.figure(figsize=(10, 6))

            # Extract health check data
            health_checks = system_status.get('health_checks', {})
            check_names = list(health_checks.keys())

            if not check_names:
                logger.warning("No health checks available for visualization")
                return None

            # Map status to numeric value for visualization
            status_map = {
                'ok': 3,
                'warning': 2,
                'critical': 1,
                'unknown': 0
            }

            # Create data for visualization
            values = [status_map.get(health_checks[name].get('status', 'unknown'), 0) for name in check_names]
            colors = ['green' if v == 3 else 'yellow' if v == 2 else 'red' if v == 1 else 'gray' for v in values]

            # Plot
            bars = plt.bar(check_names, values, color=colors)
            plt.ylim(0, 4)
            plt.ylabel('Health Status')
            plt.title('System Health Checks')
            plt.xticks(rotation=45, ha='right')

            # Add value labels
            status_labels = {3: 'OK', 2: 'WARNING', 1: 'CRITICAL', 0: 'UNKNOWN'}
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    status_labels[values[i]],
                    ha='center',
                    rotation=0
                )

            # Save the figure
            output_path = os.path.join(self.output_dir, f"health_status_{self.timestamp_str}.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            logger.debug(f"Health chart generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate health chart: {str(e)}")
            return None

    def generate_api_usage_chart(self) -> Optional[str]:
        """Generate API usage trend visualization.

        Returns:
            Path to saved chart or None if generation failed
        """
        try:
            # Load API usage data
            api_usage_path = os.path.join(ANALYTICS_DIR, "api_usage.json")
            if not os.path.exists(api_usage_path):
                logger.warning(f"API usage data file not found: {api_usage_path}")
                return None

            with open(api_usage_path, 'r') as f:
                api_data = json.load(f)

            if not api_data:
                logger.warning("No API usage data available for visualization")
                return None

            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(api_data)

            # Ensure timestamp field exists
            if 'timestamp' not in df.columns:
                logger.warning("API data missing timestamp field")
                return None

            # Convert timestamps to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            # Group by API name and date
            if 'api_name' not in df.columns:
                logger.warning("API data missing api_name field")
                return None

            # Get counts by day and API
            df['date'] = df['datetime'].dt.date
            api_counts = df.groupby(['date', 'api_name']).size().unstack().fillna(0)

            # Plot
            plt.figure(figsize=(12, 6))
            api_counts.plot(kind='line', marker='o', ax=plt.gca())

            plt.title('API Usage Trends')
            plt.xlabel('Date')
            plt.ylabel('Number of Calls')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()

            # Save the figure
            output_path = os.path.join(self.output_dir, f"api_usage_{self.timestamp_str}.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            logger.debug(f"API usage chart generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate API usage chart: {str(e)}")
            return None

    def generate_error_rate_chart(self) -> Optional[str]:
        """Generate error rate analysis visualization.

        Returns:
            Path to saved chart or None if generation failed
        """
        try:
            # Get all error log files
            error_logs = glob.glob(os.path.join(ERROR_LOGS_DIR, "error_*.log"))

            if not error_logs:
                logger.warning("No error logs found for visualization")
                return None

            # Extract modification times and categorize by day
            error_data = []
            for log_file in error_logs:
                try:
                    mod_time = os.path.getmtime(log_file)
                    date = datetime.datetime.fromtimestamp(mod_time).date()

                    # Try to extract error type from filename or content
                    error_type = "unknown"
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read(1000)  # Read first 1000 chars
                            # Look for error type pattern
                            import re
                            type_match = re.search(r'"error_type":\s*"([^"]+)"', content)
                            if type_match:
                                error_type = type_match.group(1)
                    except:
                        pass

                    error_data.append({
                        'date': date,
                        'error_type': error_type
                    })
                except Exception as e:
                    logger.warning(f"Could not process log file {log_file}: {str(e)}")

            if not error_data:
                logger.warning("No valid error data found for visualization")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(error_data)

            # Group by date and error type
            error_counts = df.groupby(['date', 'error_type']).size().unstack().fillna(0)

            # For last 30 days
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=30)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Reindex with full date range
            full_index = pd.DatetimeIndex([d for d in date_range])
            error_counts = error_counts.reindex(full_index, fill_value=0)

            # Plot
            plt.figure(figsize=(12, 6))
            error_counts.plot(kind='area', stacked=True, alpha=0.7, ax=plt.gca())

            plt.title('Error Rate Trends (Last 30 Days)')
            plt.xlabel('Date')
            plt.ylabel('Number of Errors')
            plt.grid(True, linestyle='--', alpha=0.5)

            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()

            # Save the figure
            output_path = os.path.join(self.output_dir, f"error_rate_{self.timestamp_str}.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            logger.debug(f"Error rate chart generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate error rate chart: {str(e)}")
            return None

    def generate_performance_chart(self) -> Optional[str]:
        """Generate performance metrics visualization.

        Returns:
            Path to saved chart or None if generation failed
        """
        try:
            # Load performance metrics data
            perf_path = os.path.join(ANALYTICS_DIR, "performance_metrics.json")
            if not os.path.exists(perf_path):
                logger.warning(f"Performance metrics file not found: {perf_path}")
                return None

            with open(perf_path, 'r') as f:
                perf_data = json.load(f)

            if not perf_data:
                logger.warning("No performance data available for visualization")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(perf_data)

            # Ensure required fields exist
            required_fields = ['timestamp', 'operation', 'duration_ms']
            if not all(field in df.columns for field in required_fields):
                logger.warning(f"Performance data missing required fields: {required_fields}")
                return None

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['date'] = df['datetime'].dt.date

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Average operation duration by date
            daily_avg = df.groupby(['date', 'operation'])['duration_ms'].mean().unstack()
            daily_avg.plot(kind='line', marker='o', ax=ax1)
            ax1.set_title('Average Operation Duration by Date')
            ax1.set_ylabel('Duration (ms)')
            ax1.set_xlabel('')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            # Plot 2: Operation distribution boxplot
            recent_df = df[df['datetime'] > (datetime.datetime.now() - datetime.timedelta(days=7))]
            operations = recent_df['operation'].unique()

            box_data = [recent_df[recent_df['operation'] == op]['duration_ms'] for op in operations]

            ax2.boxplot(box_data, labels=operations)
            ax2.set_title('Operation Duration Distribution (Last 7 Days)')
            ax2.set_ylabel('Duration (ms)')
            ax2.set_xticklabels(operations, rotation=45, ha='right')
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Save the figure
            output_path = os.path.join(self.output_dir, f"performance_{self.timestamp_str}.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            logger.debug(f"Performance chart generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate performance chart: {str(e)}")
            return None

    def generate_summary_report(self, chart_paths: Dict[str, str]) -> str:
        """Generate a summary report with chart thumbnails and key metrics.

        Args:
            chart_paths: Dictionary of chart types and their file paths

        Returns:
            Path to summary HTML report
        """
        try:
            # Get current system status
            system_status = check_system_status()

            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LangGraph 101 - Performance Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f2f2f2; padding: 10px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .chart-container {{ display: flex; flex-wrap: wrap; }}
                    .chart {{ margin: 10px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric {{
                        margin: 10px; padding: 15px; border-radius: 5px; min-width: 200px;
                        text-align: center; flex: 1;
                    }}
                    .ok {{ background-color: #d4edda; color: #155724; }}
                    .warning {{ background-color: #fff3cd; color: #856404; }}
                    .critical {{ background-color: #f8d7da; color: #721c24; }}
                    .unknown {{ background-color: #e2e3e5; color: #383d41; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>LangGraph 101 - Performance Dashboard</h1>
                    <p>Generated on: {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>Overall Health: <strong>{system_status.get('overall_health', 'unknown').upper()}</strong></p>
                </div>
            """

            # System Health Summary
            html_content += """
                <div class="section">
                    <h2>System Health Summary</h2>
                    <div class="metrics">
            """

            health_checks = system_status.get('health_checks', {})
            for check_name, check_data in health_checks.items():
                status = check_data.get('status', 'unknown')
                message = check_data.get('message', 'No details available')
                html_content += f"""
                        <div class="metric {status}">
                            <h3>{check_name}</h3>
                            <p>{message}</p>
                        </div>
                """

            html_content += """
                    </div>
                </div>
            """

            # System Component Status
            html_content += """
                <div class="section">
                    <h2>System Components Status</h2>
                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Status</th>
                            <th>Details</th>
                        </tr>
            """

            systems = system_status.get('systems', {})
            for component, data in systems.items():
                status = data.get('status', 'unknown')

                # Extract additional details if any
                details = []
                for key, value in data.items():
                    if key != 'status':
                        details.append(f"{key}: {value}")

                details_str = '<br>'.join(details) if details else 'No additional details'

                html_content += f"""
                        <tr>
                            <td>{component}</td>
                            <td>{status}</td>
                            <td>{details_str}</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>
            """

            # Charts Section
            html_content += """
                <div class="section">
                    <h2>Performance Charts</h2>
                    <div class="chart-container">
            """

            for chart_type, chart_path in chart_paths.items():
                if chart_type != 'summary' and os.path.exists(chart_path):
                    # Get relative path for HTML
                    rel_path = os.path.basename(chart_path)
                    title = chart_type.replace('_', ' ').title()

                    html_content += f"""
                        <div class="chart">
                            <h3>{title}</h3>
                            <img src="{rel_path}" alt="{title} Chart">
                        </div>
                    """

            html_content += """
                    </div>
                </div>
            """

            # System Information
            html_content += """
                <div class="section">
                    <h2>System Information</h2>
                    <table>
                        <tr>
                            <th>Property</th>
                            <th>Value</th>
                        </tr>
            """

            system_info = system_status.get('system_info', {})
            for key, value in system_info.items():
                if isinstance(value, dict):
                    value_str = '<br>'.join([f"{k}: {v}" for k, v in value.items()])
                else:
                    value_str = str(value)

                html_content += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value_str}</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>
            """

            # Close HTML
            html_content += """
            </body>
            </html>
            """

            # Write HTML file
            output_path = os.path.join(self.output_dir, f"dashboard_{self.timestamp_str}.html")
            with open(output_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Dashboard summary report generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate summary report: {str(e)}")
            error_path = os.path.join(self.output_dir, f"dashboard_error_{self.timestamp_str}.html")

            # Create a minimal error report
            with open(error_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html><body>
                <h1>Dashboard Generation Error</h1>
                <p>Time: {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Error: {str(e)}</p>
                </body></html>
                """)

            return error_path


def render_performance_dashboard():
    """Render the performance monitoring dashboard"""
    st.title("System Performance Dashboard")

    metrics = get_metrics()
    summary = metrics.get_summary()

    # Create tabs for different metric categories
    tabs = st.tabs(["Operations", "Errors", "Resources", "Cache"])

    with tabs[0]:
        st.header("Operation Timings")
        if summary["operation_timings"]:
            # Create timing data for plotting
            df = pd.DataFrame([
                {
                    "Operation": op,
                    "Average (ms)": stats["avg"],
                    "Min (ms)": stats["min"],
                    "Max (ms)": stats["max"],
                    "Count": stats["count"]
                }
                for op, stats in summary["operation_timings"].items()
            ])

            # Plot timing distributions
            fig = go.Figure()
            for op in df["Operation"]:
                fig.add_trace(go.Box(
                    name=op,
                    y=metrics.operation_timings[op],
                    boxpoints="all"
                ))
            st.plotly_chart(fig)

            # Show timing stats table
            st.dataframe(df)
        else:
            st.info("No operation timing data available yet")

    with tabs[1]:
        st.header("Error Distribution")
        if summary["error_counts"]:
            # Create pie chart of errors
            fig = go.Figure(data=[go.Pie(
                labels=list(summary["error_counts"].keys()),
                values=list(summary["error_counts"].values())
            )])
            st.plotly_chart(fig)

            # Show error counts table
            st.dataframe(pd.DataFrame([
                {"Error Type": err, "Count": count}
                for err, count in summary["error_counts"].items()
            ]))
        else:
            st.info("No errors recorded")

    with tabs[2]:
        st.header("Resource Usage")
        if summary["resource_usage"]:
            # Create resource usage line charts
            for resource, stats in summary["resource_usage"].items():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=metrics.resource_usage[resource],
                    mode="lines+markers",
                    name=resource
                ))
                fig.update_layout(title=f"{resource} Usage Over Time")
                st.plotly_chart(fig)

                # Show current stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Current", f"{stats['current']:.2f}")
                col2.metric("Average", f"{stats['avg']:.2f}")
                col3.metric("Max", f"{stats['max']:.2f}")
        else:
            st.info("No resource usage data available")

    with tabs[3]:
        st.header("Cache Performance")
        if summary["cache_stats"]:
            # Calculate cache hit rate
            hits = summary["cache_stats"]["hits"]
            misses = summary["cache_stats"]["misses"]
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0

            # Display cache metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
            col2.metric("Cache Size", summary["cache_stats"]["size"])
            col3.metric("Total Requests", total)

            # Show cache hit/miss pie chart
            fig = go.Figure(data=[go.Pie(
                labels=["Hits", "Misses"],
                values=[hits, misses],
                hole=0.4
            )])
            st.plotly_chart(fig)
        else:
            st.info("No cache statistics available")

    # Show last update time
    st.sidebar.info(f"Last updated: {summary['last_update']}")


if __name__ == "__main__":
    # Example usage
    print("Generating performance dashboard...")
    dashboard = PerformanceDashboard()
    output_files = dashboard.generate_dashboard()

    print("Dashboard generated. Output files:")
    for chart_type, file_path in output_files.items():
        print(f"- {chart_type}: {file_path}")

    # Open the summary report if available
    if 'summary' in output_files and os.path.exists(output_files['summary']):
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_files['summary'])}")
    else:
        print("No summary report was generated.")
