import os
import pandas as pd
import requests
import logging
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# File path to monitor
CSV_FILE_PATH = r"C:\Users\amin\Desktop\smart-space\anomaly_detection\data\logs\sensor_data.csv"

# Initialize the last modified timestamp
last_modified_time = None

# Flask server URLs
BASE_URL = "http://127.0.0.1:5000/api/v1"  # Base URL for your API
RECOMMEND_RULES_ENDPOINT = "/recommendation/recommend_rules"
DETECT_ANOMALIES_ENDPOINT = "/anomaly_detection/detect_dataset_anomalies"

def check_new_logs_and_trigger_routes():
    """Function to monitor the CSV file and trigger routes if there is a new feed."""
    global last_modified_time
    func_name = check_new_logs_and_trigger_routes.__name__

    if not os.path.exists(CSV_FILE_PATH):
        logger.warning(f"[{func_name}] File not found: {CSV_FILE_PATH}")
        return

    # Get the current modified time of the file
    current_modified_time = os.path.getmtime(CSV_FILE_PATH)

    # Check if the file has been updated
    if last_modified_time is None or current_modified_time > last_modified_time:
        last_modified_time = current_modified_time
        logger.info(f"[{func_name}] New logs detected! Triggering routes...")

        # Load and process new data (if required)
        try:
            data = pd.read_csv(CSV_FILE_PATH)
            logger.info(f"[{func_name}] Loaded data: {data.tail()}")
        except Exception as e:
            logger.error(f"[{func_name}] Error reading CSV file: {e}")
            return

        # Trigger Flask routes
        try:
            response1 = requests.get(f"{BASE_URL}{RECOMMEND_RULES_ENDPOINT}")
            logger.info(f"[{func_name}] Triggered {RECOMMEND_RULES_ENDPOINT}, Response: {response1.status_code}")

            response2 = requests.get(f"{BASE_URL}{DETECT_ANOMALIES_ENDPOINT}")
            logger.info(f"[{func_name}] Triggered {DETECT_ANOMALIES_ENDPOINT}, Response: {response2.status_code}")
        except Exception as e:
            logger.error(f"[{func_name}] Error triggering routes: {e}")
    else:
        logger.info(f"[{func_name}] No new logs detected.")

def start_logs_monitor():
    """Start the scheduler to monitor logs."""
    func_name = start_logs_monitor.__name__

    scheduler = BackgroundScheduler()
    scheduler.add_job(check_new_logs_and_trigger_routes, 'interval', minutes=1)
    scheduler.start()

    logger.info(f"[{func_name}] Logs monitor scheduler started.")
