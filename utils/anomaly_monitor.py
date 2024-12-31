import os
import logging
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'anomaly_detection'))
DATA_FILE_ANOMALY = os.path.join(BASE_DIR, "data", "logs", "sensor_data_anomaly.csv")

BASE_URL = "http://127.0.0.1:5000/api/v1"
NODE_BASE_URL = "http://127.0.0.1:8001"
DETECT_ANOMALIES_ENDPOINT = "/anomaly_detection/detect_dataset_anomalies"
NODE_ANOMALY_RESPONSE_ENDPOINT = "/anomaly_response"

last_modified_time_anomaly = None


def send_to_node(api_url, data):
    try:
        response = requests.post(api_url, json=data)
        logger.info(f"Sent data to {api_url}, Node.js response status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending data to {api_url}: {e}")


def check_anomaly_logs():
    global last_modified_time_anomaly

    if not os.path.exists(DATA_FILE_ANOMALY):
        logger.warning(f"File not found: {DATA_FILE_ANOMALY}")
        return

    current_modified_time = os.path.getmtime(DATA_FILE_ANOMALY)
    if last_modified_time_anomaly is None or current_modified_time > last_modified_time_anomaly:
        last_modified_time_anomaly = current_modified_time
        logger.info("New anomaly logs detected! Triggering anomaly detection...")

        try:
            data = pd.read_csv(DATA_FILE_ANOMALY)
            logger.info(f"Loaded anomaly data: {data.tail()}")
        except Exception as e:
            logger.error(f"Error reading anomaly CSV file: {e}")
            return

        try:
            response = requests.get(f"{BASE_URL}{DETECT_ANOMALIES_ENDPOINT}")
            if response.status_code == 200:
                logger.info("Anomaly detection triggered successfully.")
                send_to_node(f"{NODE_BASE_URL}{NODE_ANOMALY_RESPONSE_ENDPOINT}", response.json())
            else:
                logger.error(f"Failed to trigger anomaly detection: {response.status_code}")
        except Exception as e:
            logger.error(f"Error triggering anomaly detection: {e}")
    else:
        logger.info("No new anomaly logs detected.")


def start_anomaly_monitor():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_anomaly_logs, 'interval', minutes=1)
    scheduler.start()
    logger.info("Anomaly monitor scheduler started.")
