import csv
import os
import sys
import datetime

# Add the parent directory to sys.path to allow importing analytics_logger and analytics_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)))) # For analytics_logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'analytics_processing'))) # For basic_analyzer

try:
    from analytics_logger import get_db_connection, DATABASE_NAME
    from analytics_processing.basic_analyzer import count_events_by_type, get_model_usage_summary, get_prompt_template_usage_summary
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure analytics_logger.py and analytics_processing/basic_analyzer.py are accessible.")
    # Fallback for DATABASE_NAME if import fails
    DATABASE_NAME = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analytics_data", "analytics.db")
    # Define dummy functions if imports fail, so the script can still be loaded (though not fully functional)
    def get_db_connection(): raise NotImplementedError("Database connection not available due to import error.")
    def count_events_by_type(): return {}
    def get_model_usage_summary(): return {}
    def get_prompt_template_usage_summary(): return {}


REPORTS_DIR = "reports"

def ensure_reports_dir():
    """Ensures the reports directory exists."""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

def generate_event_summary_report_txt(filename_prefix="event_summary"):
    """Generates a text-based report summarizing event counts."""
    ensure_reports_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REPORTS_DIR, f"{filename_prefix}_{timestamp}.txt")

    if not os.path.exists(DATABASE_NAME):
        with open(filename, 'w') as f:
            f.write(f"Report Generation Time: {datetime.datetime.now().isoformat()}\n")
            f.write("Error: Analytics database not found.\n")
        print(f"Report generated with error (DB not found): {filename}")
        return filename

    event_counts = count_events_by_type()
    model_usage = get_model_usage_summary()
    prompt_usage = get_prompt_template_usage_summary()

    with open(filename, 'w') as f:
        f.write(f"Analytics Event Summary Report\n")
        f.write(f"Generated on: {datetime.datetime.now().isoformat()}\n")
        f.write("========================================\n\n")

        f.write("Event Counts by Type:\n")
        if event_counts:
            for event_type, count in event_counts.items():
                f.write(f"- {event_type}: {count}\n")
        else:
            f.write("No event data found.\n")
        f.write("\n")

        f.write("Model Usage Summary:\n")
        if model_usage:
            for model, count in model_usage.items():
                f.write(f"- {model}: {count} interactions\n")
        else:
            f.write("No model usage data found.\n")
        f.write("\n")

        f.write("Prompt Template Usage Summary:\n")
        if prompt_usage:
            for template, count in prompt_usage.items():
                f.write(f"- {template}: {count} uses\n")
        else:
            f.write("No prompt template usage data found.\n")

    print(f"Text report generated: {filename}")
    return filename

def generate_raw_events_report_csv(filename_prefix="raw_events", limit=100):
    """Generates a CSV report of raw events."""
    ensure_reports_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REPORTS_DIR, f"{filename_prefix}_{timestamp}.csv")

    if not os.path.exists(DATABASE_NAME):
        with open(filename, 'w', newline='') as csvfile: # Create empty CSV with header for consistency
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'timestamp', 'event_type', 'user_id', 'session_id', 'model_id', 'prompt_template_id', 'details_json'])
            writer.writerow(['ERROR', 'Database not found', '', '', '', '', '', ''])
        print(f"CSV report generated with error (DB not found): {filename}")
        return filename

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, timestamp, event_type, user_id, session_id, model_id, prompt_template_id, details_json FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        headers = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"CSV report generated: {filename}")
    except sqlite3.Error as e:
        print(f"Error generating CSV report: {e}")
        # Create an empty CSV or one with an error message
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Error'])
            writer.writerow([f"Failed to retrieve data: {e}"])
    finally:
        conn.close()
    return filename

if __name__ == '__main__':
    print("Report Generator")
    print("================")
    # Ensure analytics_logger has run at least once to create DB and log sample data
    # You might need to run `python analytics_logger.py` first if the DB doesn't exist.
    if not os.path.exists(DATABASE_NAME):
        print(f"\nWARNING: Analytics database '{DATABASE_NAME}' not found.")
        print("Please run 'python analytics_logger.py' to initialize the database and log sample events before generating reports.")
    else:
        print(f"\nUsing database: {DATABASE_NAME}")

    print("\nGenerating text summary report...")
    txt_report_path = generate_event_summary_report_txt()
    print(f"Text report saved to: {txt_report_path}")

    print("\nGenerating CSV raw events report (last 100 events)...")
    csv_report_path = generate_raw_events_report_csv(limit=100)
    print(f"CSV report saved to: {csv_report_path}")

    print("\nTo see data in reports, ensure events have been logged via 'analytics_logger.py' or through application integration.")