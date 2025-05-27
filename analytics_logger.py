import sqlite3
import json
import datetime
import os

DATABASE_DIR = "analytics_data"
DATABASE_NAME = os.path.join(DATABASE_DIR, "analytics.db")

def get_db_connection():
    """Establishes a connection to the SQLite database.
    Creates the database and directory if they don't exist.
    """
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initializes the database and creates the 'events' table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            user_id TEXT,
            session_id TEXT,
            model_id TEXT, /* For tracking model usage */
            prompt_template_id TEXT, /* For tracking prompt template usage */
            details_json TEXT /* For other event-specific data */
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_NAME}")

def log_event(event_type: str, user_id: str = None, session_id: str = None,
              model_id: str = None, prompt_template_id: str = None, details: dict = None):
    """Logs an event to the analytics database.

    Args:
        event_type (str): The type of event (e.g., 'api_call', 'feature_usage', 'model_interaction').
        user_id (str, optional): The ID of the user associated with the event.
        session_id (str, optional): The ID of the session associated with the event.
        model_id (str, optional): The ID of the AI model used.
        prompt_template_id (str, optional): The ID of the prompt template used.
        details (dict, optional): A dictionary containing additional event-specific information.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    details_json = json.dumps(details) if details else None

    try:
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, user_id, session_id, model_id, prompt_template_id, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, event_type, user_id, session_id, model_id, prompt_template_id, details_json))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error logging event: {e}")
        # Consider more robust error handling, e.g., logging to a fallback file
    finally:
        conn.close()

if __name__ == '__main__':
    # Initialize the database when the script is run directly (for setup)
    initialize_database()

    # Example usage:
    log_event(event_type='api_call', user_id='user123', session_id='sessionABC', details={'endpoint': '/api/v1/data', 'method': 'GET'})
    log_event(event_type='feature_usage', user_id='user456', details={'feature_name': 'advanced_search'})
    log_event(event_type='model_interaction', user_id='user789', model_id='gpt-4', prompt_template_id='summary_v1', details={'input_length': 500, 'output_length': 150})
    print(f"Sample events logged to {DATABASE_NAME}")

    # Verify by reading back
    conn_verify = get_db_connection()
    cursor_verify = conn_verify.cursor()
    cursor_verify.execute("SELECT * FROM events")
    rows = cursor_verify.fetchall()
    print("\nLogged events:")
    for row in rows:
        print(dict(row))
    conn_verify.close()