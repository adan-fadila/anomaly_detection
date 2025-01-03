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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE_RECOMMENDATION = os.path.join(BASE_DIR, "data", "logs", "sensor_data_recommendation.csv")

BASE_URL = "http://127.0.0.1:5000/api/v1"
NODE_BASE_URL = "http://127.0.0.1:8001"
RECOMMEND_RULES_ENDPOINT = "/recommendation/recommend_rules"
NODE_RECOMMEND_RESPONSE_ENDPOINT = "/recommendation_response"

last_modified_time_recommendation = None
last_recommendations = None


def send_to_node(api_url, data):
    try:
        response = requests.post(api_url, json=data)
        logger.info(f"Sent data to {api_url}, Node.js response status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending data to {api_url}: {e}")


def check_recommendation_logs():
    global last_modified_time_recommendation, last_recommendations

    if not os.path.exists(DATA_FILE_RECOMMENDATION):
        logger.warning(f"File not found: {DATA_FILE_RECOMMENDATION}")
        return

    current_modified_time = os.path.getmtime(DATA_FILE_RECOMMENDATION)
    if last_modified_time_recommendation is None or current_modified_time > last_modified_time_recommendation:
        last_modified_time_recommendation = current_modified_time
        logger.info("New recommendation logs detected! Triggering recommendation rules...")

        try:
            data = pd.read_csv(DATA_FILE_RECOMMENDATION)
            logger.info(f"Loaded recommendation data: {data.tail()}")
        except Exception as e:
            logger.error(f"Error reading recommendation CSV file: {e}")
            return

        try:
            response = requests.get(f"{BASE_URL}{RECOMMEND_RULES_ENDPOINT}")
            if response.status_code == 200:
                recommendations = response.json()
                logger.info("Fetched recommendations successfully.")

                # Process the recommendations into a dictionary grouped by feature
                processed_recommendations = {}
                for rec in recommendations:
                    feature = rec["feature"]
                    if feature not in processed_recommendations:
                        processed_recommendations[feature] = []
                    processed_recommendations[feature].append({
                        "recommendation": rec["recommendation"],
                        "recommended_time": rec["recommended_time"]
                    })

                differences = {
                    key: processed_recommendations[key]
                    for key in processed_recommendations
                    if last_recommendations is None or last_recommendations.get(key) != processed_recommendations[key]
                }

                if differences:
                    logger.info(f"Sending updated recommendations: {differences}")
                    send_to_node(f"{NODE_BASE_URL}{NODE_RECOMMEND_RESPONSE_ENDPOINT}", differences)
                else:
                    logger.info("No new recommendations to send.")

                last_recommendations = processed_recommendations
            else:
                logger.error(f"Failed to fetch recommendation rules: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching recommendation rules: {e}")
    else:
        logger.info("No new recommendation logs detected.")


def start_recommendation_monitor():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_recommendation_logs, 'interval', minutes=1)
    scheduler.start()
    logger.info("Recommendation monitor scheduler started.")
