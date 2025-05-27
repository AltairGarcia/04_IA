"""
Simple health check server that runs alongside the Streamlit application.
Provides REST API endpoints for health monitoring.
"""
import json
import threading
import requests
from flask import Flask, jsonify
from datetime import datetime
import logging

# Import health functions
from app_health import get_health_summary, run_health_check

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint that returns JSON status."""
    try:
        # Get health summary from the health monitoring system
        health_summary = get_health_summary()
        
        # Convert to JSON-serializable format
        response = {
            "status": "ok" if health_summary["overall_status"] == "ok" else "error",
            "timestamp": datetime.now().isoformat(),
            "overall_status": health_summary["overall_status"],
            "total_checks": health_summary["total_checks"],
            "ok_count": health_summary["ok_count"],
            "warning_count": health_summary["warning_count"],
            "critical_count": health_summary["critical_count"],
            "checks": {}
        }
        
        # Add detailed check information
        for check_name, check_result in health_summary.get("checks", {}).items():
            if hasattr(check_result, 'to_dict'):
                response["checks"][check_name] = check_result.to_dict()
            else:
                response["checks"][check_name] = str(check_result)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        error_response = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_message": "Failed to retrieve or process health summary.",
            "details": str(e)
        }
        return jsonify(error_response), 500

@app.route('/health/simple', methods=['GET'])
def simple_health():
    """Simple health check that just returns OK."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "message": "Health server is running"
    })

@app.route('/status', methods=['GET'])
def status():
    """System status endpoint."""
    try:
        health_results = run_health_check()
        
        # Convert health results to JSON-serializable format
        status_info = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational",
            "checks": {}
        }
        
        for check_name, result in health_results.items():
            if hasattr(result, 'to_dict'):
                status_info["checks"][check_name] = result.to_dict()
            else:
                status_info["checks"][check_name] = {
                    "status": "unknown",
                    "message": str(result)
                }
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "system_status": "error",
            "error": str(e)
        }), 500

def start_health_server(port=8502, debug=False):
    """Start the health server in a separate thread."""
    import time
    import requests
    
    def run_server():
        try:
            app.run(host='127.0.0.1', port=port, debug=debug, use_reloader=False, threaded=True)
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready with retries
    max_retries_simple = 10
    simple_health_ready = False
    for attempt in range(max_retries_simple):
        try:
            time.sleep(0.2)  # Brief wait
            response = requests.get(f"http://127.0.0.1:{port}/health/simple", timeout=2)
            if response.status_code == 200:
                logger.info(f"Simple health endpoint /health/simple confirmed on http://127.0.0.1:{port}")
                simple_health_ready = True
                break
        except requests.exceptions.RequestException:
            if attempt < max_retries_simple - 1:
                time.sleep(0.5) # Longer sleep before retry
                continue
            else:
                logger.warning(f"Simple health endpoint /health/simple not ready on port {port} after {max_retries_simple} attempts.")
                return server_thread # Return thread even if simple health fails, main health check will also fail

    if not simple_health_ready:
        logger.warning(f"Health server's /health/simple endpoint not ready. Main /health check will likely fail.")
        # Still proceed to check main health endpoint as per requirements, but it's unlikely to pass
        # return server_thread # Decided to proceed to main health check

    # New: Poll the main /health endpoint
    max_retries_main = 5
    main_health_ready = False
    if simple_health_ready: # Only attempt main health check if simple one passed
        logger.info(f"Polling main health endpoint /health on http://127.0.0.1:{port}...")
        for attempt_main in range(max_retries_main):
            try:
                time.sleep(0.5) # Short delay between retries
                response_main = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
                if response_main.status_code == 200:
                    try:
                        data = response_main.json()
                        if "overall_status" in data:
                            logger.info(f"Main health endpoint /health confirmed ready and valid on http://127.0.0.1:{port}")
                            main_health_ready = True
                            break
                        else:
                            logger.warning(f"/health endpoint on port {port} returned 200 but missing 'overall_status' key. Attempt {attempt_main + 1}/{max_retries_main}")
                    except json.JSONDecodeError:
                        logger.warning(f"/health endpoint on port {port} returned 200 but content is not valid JSON. Attempt {attempt_main + 1}/{max_retries_main}")
                else:
                    logger.warning(f"/health endpoint on port {port} returned status {response_main.status_code}. Attempt {attempt_main + 1}/{max_retries_main}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error connecting to /health endpoint on port {port}: {e}. Attempt {attempt_main + 1}/{max_retries_main}")
            
            if attempt_main < max_retries_main - 1:
                time.sleep(1) # Wait a bit longer before next retry for main health
            else:
                logger.warning(f"Main health endpoint /health on port {port} did not become fully ready after {max_retries_main} attempts.")

    if simple_health_ready and main_health_ready:
        logger.info(f"Health server confirmed fully running on http://127.0.0.1:{port}")
    elif simple_health_ready and not main_health_ready:
        logger.warning(f"Health server's /health/simple is UP, but main /health endpoint is NOT fully ready on http://127.0.0.1:{port}.")
    else: # simple_health_ready is False
        logger.error(f"Health server failed to start properly on http://127.0.0.1:{port}. Neither /health/simple nor /health are ready.")

    return server_thread

if __name__ == '__main__':
    # Run health server directly
    logger.info("Starting health check server...")
    app.run(host='127.0.0.1', port=8502, debug=True)
