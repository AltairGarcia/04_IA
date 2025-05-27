import sqlite3
import os
import sys
from collections import Counter

# Add the parent directory to sys.path to allow importing analytics_logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from analytics_logger import get_db_connection, DATABASE_NAME
except ImportError:
    print("Error: Could not import analytics_logger. Ensure it's in the parent directory or PYTHONPATH.")
    # Fallback for DATABASE_NAME if import fails, assuming standard location
    DATABASE_NAME = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "analytics_data", "analytics.db")


def count_events_by_type():
    """Counts the occurrences of each event type."""
    if not os.path.exists(DATABASE_NAME):
        print(f"Database file not found at {DATABASE_NAME}. Run analytics_logger.py to create it.")
        return Counter()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT event_type FROM events")
        event_types = [row['event_type'] for row in cursor.fetchall()]
        return Counter(event_types)
    except sqlite3.Error as e:
        print(f"Error querying event types: {e}")
        return Counter()
    finally:
        conn.close()

def get_recent_events(limit=10):
    """Retrieves the most recent events."""
    if not os.path.exists(DATABASE_NAME):
        print(f"Database file not found at {DATABASE_NAME}. Run analytics_logger.py to create it.")
        return []

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        events = [dict(row) for row in cursor.fetchall()]
        return events
    except sqlite3.Error as e:
        print(f"Error retrieving recent events: {e}")
        return []
    finally:
        conn.close()

def get_model_usage_summary():
    """Generates a summary of AI model usage."""
    if not os.path.exists(DATABASE_NAME):
        print(f"Database file not found at {DATABASE_NAME}. Run analytics_logger.py to create it.")
        return Counter()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT model_id FROM events WHERE model_id IS NOT NULL")
        model_ids = [row['model_id'] for row in cursor.fetchall()]
        return Counter(model_ids)
    except sqlite3.Error as e:
        print(f"Error querying model usage: {e}")
        return Counter()
    finally:
        conn.close()

def get_prompt_template_usage_summary():
    """Generates a summary of prompt template usage."""
    if not os.path.exists(DATABASE_NAME):
        print(f"Database file not found at {DATABASE_NAME}. Run analytics_logger.py to create it.")
        return Counter()

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT prompt_template_id FROM events WHERE prompt_template_id IS NOT NULL")
        template_ids = [row['prompt_template_id'] for row in cursor.fetchall()]
        return Counter(template_ids)
    except sqlite3.Error as e:
        print(f"Error querying prompt template usage: {e}")
        return Counter()
    finally:
        conn.close()


if __name__ == '__main__':
    print("Analytics Processing - Basic Analyzer")
    print("=====================================")

    event_counts = count_events_by_type()
    if event_counts:
        print("\nEvent Counts by Type:")
        for event_type, count in event_counts.items():
            print(f"- {event_type}: {count}")
    else:
        print("\nNo event counts found. The database might be empty or inaccessible.")

    recent_events_data = get_recent_events(5)
    if recent_events_data:
        print("\nMost Recent Events (limit 5):")
        for event in recent_events_data:
            print(f"- {event['timestamp']} | {event['event_type']} | User: {event['user_id']} | Model: {event['model_id']} | Details: {event['details_json']}")
    else:
        print("\nNo recent events found.")

    model_usage = get_model_usage_summary()
    if model_usage:
        print("\nModel Usage Summary:")
        for model, count in model_usage.items():
            print(f"- {model}: {count} interactions")
    else:
        print("\nNo model usage data found.")

    prompt_usage = get_prompt_template_usage_summary()
    if prompt_usage:
        print("\nPrompt Template Usage Summary:")
        for template, count in prompt_usage.items():
            print(f"- {template}: {count} uses")
    else:
        print("\nNo prompt template usage data found.")

    print("\nNote: If you see errors or no data, ensure 'analytics_logger.py' has been run to create and populate the database.")