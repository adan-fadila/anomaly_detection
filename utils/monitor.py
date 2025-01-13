import os
import logging
import requests
import base64


from threading import Lock
from apscheduler.schedulers.background import BackgroundScheduler
from utils.data_manager import Data_Set_Manager
from managers.anomaly_manager import AnomalyDetectionManager
from algorithms.recommendations.bayesian import BayesianRecommendation
from config.constant import POINTWISE, COLLECTIVE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE_RECOMMENDATION = os.path.join(BASE_DIR, "data", "logs", "sensor_data_recommendation.csv")

ANOMALY_DATA_FRAME_FILE = os.path.join(BASE_DIR, 'data', 'logs', 'sensor_data_anomaly.csv')
ANOMALY_DATASET_FILE = os.path.join(BASE_DIR, 'data', 'csv', 'DailyDelhiClimateTrain.csv')

NODE_BASE_URL = "http://127.0.0.1:8001"
NODE_ANOMALY_RESPONSE_ENDPOINT = "/anomaly_response"
NODE_COLL_ANOMALY_RESPONSE_ENDPOINT = "/coll_anomaly_response"
NODE_RECOMMEND_RESPONSE_ENDPOINT = "/recommendation_response"

last_modified_time_anomaly = None
last_modified_time_recommendation = None
last_recommendations = None

        
        


# Create a global threading lock
monitor_lock = Lock()

def send_to_node(api_url, data):
    try:
        response = requests.post(api_url, json=data)
        logger.info(f"Sent data to {api_url}, Node.js response status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending data to {api_url}: {e}")




def detect_anomalies():
    try:
        # Initialize dataset managers and process datasets
        dataset_manager = Data_Set_Manager(ANOMALY_DATASET_FILE)
        dataset = dataset_manager.process_dataset()

        dataset_manager_2 = Data_Set_Manager(ANOMALY_DATA_FRAME_FILE)
        dataset_2 = dataset_manager_2.process_dataset()

        # Validate datasets
        if dataset.empty:
            logger.error("The dataset is empty or could not be loaded.")
            return {'error': 'The dataset is empty or could not be loaded'}

        required_columns = ['date', 'meantemp']
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return {'error': f"Required columns ({', '.join(missing_columns)}) are missing from the dataset"}

        # Extract relevant data
        df = dataset_2[['date', 'meantemp']]

        # Initialize anomaly detection manager
        anomaly_manager = AnomalyDetectionManager()

        # Detect anomalies
        anomaly_result, plot_image = anomaly_manager.detect_anomalies(df, dataset, POINTWISE)
        coll_anomaly, coll_plot = anomaly_manager.detect_anomalies(df, dataset, COLLECTIVE)

        # Extract anomalies and format the date
        anomalies = anomaly_result[anomaly_result['is_anomaly'] == True]
        anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))

        # Encode plots to Base64
        plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')
        coll_plot_base64 = base64.b64encode(coll_plot.getvalue()).decode('utf-8')

        # Prepare response data
        anomaly_response = {
            'anomalies': anomalies.to_dict(orient='records'),
            'plot_image': plot_image_base64,
        }
        collective_response = {
            'collective_plot': coll_plot_base64
        }

        # Send data to Node.js endpoints
        try:
            send_to_node(f"{NODE_BASE_URL}{NODE_ANOMALY_RESPONSE_ENDPOINT}", anomaly_response)
            logger.info("Anomaly detection response sent successfully to Node.js endpoint.")
        except Exception as node_error:
            logger.error(f"Failed to send anomaly detection response: {node_error}")

        try:
            send_to_node(f"{NODE_BASE_URL}{NODE_COLL_ANOMALY_RESPONSE_ENDPOINT}", collective_response)
            logger.info("Collective anomaly response sent successfully to Node.js endpoint.")
        except Exception as node_error:
            logger.error(f"Failed to send collective anomaly response: {node_error}")

        # Final response
        response = {
            'response_anomaly': anomaly_response,
            'response_collective': collective_response
        }

        logger.info("Anomaly detection process completed successfully.")
        return response

    except ValueError as ve:
        logger.error(f"Value error during anomaly detection: {ve}")
        return {'error': f'Value error: {str(ve)}'}

    except KeyError as ke:
        logger.error(f"Key error during anomaly detection: {ke}")
        return {'error': f'Key error: {str(ke)}'}

    except Exception as e:
        logger.exception(f"Unexpected error during anomaly detection: {e}")
        return {'error': f'Internal server error: {str(e)}'}




def check_anomaly_logs():
    global last_modified_time_anomaly
    
    if not os.path.exists(ANOMALY_DATASET_FILE):
        logger.warning(f"File not found: {ANOMALY_DATASET_FILE}")
        return

    current_modified_time = os.path.getmtime(ANOMALY_DATASET_FILE)
    if last_modified_time_anomaly is None or current_modified_time > last_modified_time_anomaly:
        last_modified_time_anomaly = current_modified_time
        logger.info("New anomaly logs detected! Triggering anomaly detection...")
        # load_csv(DATA_FILE_ANOMALY)
        detect_anomalies()
    else:
        logger.info("No new anomaly logs detected.")

def process_and_send_recommendations():
    global last_recommendations

    try:
        # Fetch recommendations
        recommender = BayesianRecommendation(DATA_FILE_RECOMMENDATION)
        recommendations = recommender.recommend_rules()
        logger.info("Fetched recommendations successfully.")

        # Process recommendations
        processed_recommendations = {}
        for rec in recommendations:
            feature = rec["device"]
            if feature not in processed_recommendations:
                processed_recommendations[feature] = []
            processed_recommendations[feature].append({
                "recommendation": rec["recommendation"],
                "recommended_time": rec["recommended_time"]
            })

        # Identify differences
        differences = {
            key: processed_recommendations[key]
            for key in processed_recommendations
            if last_recommendations is None or last_recommendations.get(key) != processed_recommendations[key]
        }

        # Send updates to node server
        if differences:
            logger.info(f"Sending updated recommendations: {differences}")
            response = requests.post(f"{NODE_BASE_URL}{NODE_RECOMMEND_RESPONSE_ENDPOINT}", json=differences)
            if response.status_code == 200:
                logger.info("Recommendations sent successfully.")
            else:
                logger.error(f"Failed to send recommendations: {response.status_code}")
        else:
            logger.info("No new recommendations to send.")

        last_recommendations = processed_recommendations

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
    except ValueError as e:
        logger.error(f"Invalid data provided for recommendations: {e}")
    except requests.RequestException as e:
        logger.error(f"Error communicating with node server: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        
        
        


def check_recommendation_logs():
    global last_modified_time_recommendation

    if not os.path.exists(DATA_FILE_RECOMMENDATION):
        logger.warning(f"File not found: {DATA_FILE_RECOMMENDATION}")
        return

    current_modified_time = os.path.getmtime(DATA_FILE_RECOMMENDATION)
    if last_modified_time_recommendation is None or current_modified_time > last_modified_time_recommendation:
        last_modified_time_recommendation = current_modified_time
        logger.info("New recommendation logs detected! Triggering recommendation rules...")
        # data = load_csv(DATA_FILE_RECOMMENDATION)
        # if data is not None:
        process_and_send_recommendations()
    else:
        logger.info("No new recommendation logs detected.")
        
        
        

def start_monitor():
    scheduler = BackgroundScheduler()

    # Job 1: Check anomaly logs
    scheduler.add_job(
        check_anomaly_logs,
        'interval',
        minutes=1,
        id='anomaly_monitor_job',
        max_instances=1,
        replace_existing=True
    )

    # Job 2: Check recommendation logs
    scheduler.add_job(
        check_recommendation_logs,
        'interval',
        minutes=1,
        id='recommendation_monitor_job',
        max_instances=1,
        replace_existing=True
    )

    scheduler.start()
    logger.info("Unified monitor scheduler started.")
