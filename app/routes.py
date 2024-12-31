import os
import pandas as pd
import json
from flask import Blueprint, request, jsonify
from managers.anomaly_manager import AnomalyDetectionManager
from utils.data_manager import Data_Set_Manager
from algorithms.recommendations.bayesian import BayesianRecommendation

# Determine project root directory and dataset path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_ANOMALY = os.path.join(BASE_DIR, 'data', 'logs', 'sensor_data_anomaly.csv')
DATA_FILE_RECOMMENDATION = os.path.join(BASE_DIR, 'data', 'logs', 'sensor_data_recommendation.csv')

CONFIG_FILE = os.path.join(BASE_DIR,'config','config.json')

# Initialize Blueprints
anomaly_detection_bp = Blueprint('anomaly_detection', __name__)
recommendation_bp = Blueprint('recommendation', __name__)

# Utility function: Load algorithm configuration
def load_algorithm_config(config_file=CONFIG_FILE):
    """
    Load the algorithm configuration from a JSON file.
    Returns a list of algorithms to use or defaults to ["stl"].
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            list_= config.get("algorithms")
        return list_ 
    except FileNotFoundError:
        return ValueError(f"File Not Found Error : {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
















############################################################################################################
# Route: Detect Anomalies with Payload
# to test this api you have the two following optios:
#
# a - with curl command : 
#                       1- in the terminal run python server.py
#                       2- in another terminal run the following command:
#                       3- curl -v http://localhost:5000/api/v1/anomaly_detection/detect_anomalies -H "Content-Type: application/json" -d "{\"sensor_values\": [70.0, 72.5, 168.4, 75.3, 74.1]}"
#
# b - with postman :
#                       1- in the terminal run python server.py
#                       2- open postman and create a new POST request with the following url http://localhost:5000/api/v1/anomaly_detection/detect_anomalies_with_payload
#                       3- in the headers section add a new key "Content-Type" with the value "application/json"
#                       4- in the body section select the raw option and add the following json payload:
#                           {
#                               "sensor_values": [
#                                   {"sensor_value": 70.0},
#                                   {"sensor_value": 72.5},
#                                   {"sensor_value": 68.4},
#                                   {"sensor_value": 75.3},
#                                   {"sensor_value": 74.1},
#                                   {"sensor_value": 69.9},
#                                   {"sensor_value": 71.2},
#                                   {"sensor_value": 70.8},
#                                   {"sensor_value": 69.5},
#                                   {"sensor_value": 73.4}
#                               ]
#                           }
#                       5- click on the send button

  
@anomaly_detection_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    try:
        # Step 1: Validate Content-Type
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415

        # Step 2: Parse and validate JSON payload
        data = request.get_json()
        if not data or 'sensor_values' not in data:
            return jsonify({'error': 'No valid sensor values found in the request'}), 400

        sensor_values = data['sensor_values']
        if not isinstance(sensor_values, list) or not all(isinstance(val, (float, dict)) for val in sensor_values):
            return jsonify({'error': 'Invalid format for sensor_values. Must be a list of floats or objects.'}), 400

        # Step 3: Construct the DataFrame
        dfs = pd.DataFrame([
            {
                'date': pd.to_datetime(sensor.get("timestamp", "2017/01/02"), errors='coerce'),
                'meantemp': sensor.get("sensor_value")
            } if isinstance(sensor, dict) else
            {
                'date': pd.to_datetime("2017/01/02", errors='coerce'),
                'meantemp': sensor
            }
            for sensor in sensor_values if isinstance(sensor, dict) or sensor is not None
        ])

        if dfs.empty:
            return jsonify({'error': 'No valid sensor values provided'}), 400

        # Step 4: Initialize AnomalyDetectionManager
        algorithms_to_use = load_algorithm_config()
        anomaly_manager = AnomalyDetectionManager(algorithms_to_use)

        # Step 5: Load dataset
        dataset_manager = Data_Set_Manager(DATA_FILE_ANOMALY)
        dataset = dataset_manager.process_dataset()

        # Step 6: Perform anomaly detection
        result = anomaly_manager.detect_anomalies(dfs, dataset)

        # Step 7: Filter anomalies
        anomalies = result[result['is_anomaly'] == True]

        # Ensure proper formatting of 'date' for JSON response
        anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))

        # Step 8: Prepare and send response
        response = {'anomalies': anomalies.to_dict(orient='records')}
        return jsonify(response), 200

    except ValueError as ve:
        # Handle ValueErrors (e.g., parsing issues)
        return jsonify({'error': f'ValueError: {ve}'}), 400
    except Exception as e:
        # Handle unexpected errors gracefully
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

############################################################################################################
# Route: Detect Anomalies with dataset or logs file
#
# to test this api you have the two following optios:
#
# a - with curl command :
#
#          1- make sure to comment out the scheduler in the app dir / __init__.py file other wise this api
#             will be invoked automatically every 1 minute
#
#          2- in the terminal run python server.py
# 
#          3- in another terminal run the following command:
#            curl -v http://localhost:5000/api/v1/anomaly_detection/detect_dataset_anomalies
#
#
# b - with postman :
#          1- make sure to comment out the scheduler in the app dir / __init__.py file other wise this api
#             will be invoked automatically every 1 minute
#
#          2- in the terminal run python server.py
#
#          3- open postman and create a new GET request with the following url http://localhost:5000/api/v1/anomaly_detection/detect_dataset_anomalies
#          4- click on the send button
#
########################################################################################################################################
       

@anomaly_detection_bp.route('/detect_dataset_anomalies', methods=['GET'])
def detect_dataset_anomalies():
    try:
        # Step 1: Load the dataset
        dataset_manager = Data_Set_Manager(DATA_FILE_ANOMALY)
        dataset = dataset_manager.process_dataset()

        if dataset.empty:
            return jsonify({'error': 'The dataset is empty or could not be loaded'}), 500

        # Step 2: Prepare DataFrame for anomaly detection
        # Use 'date' and 'meantemp' columns directly
        if 'date' not in dataset.columns or 'meantemp' not in dataset.columns:
            return jsonify({'error': "Required columns ('date', 'meantemp') are missing from the dataset"}), 500

        # Create DataFrame with necessary columns
        df = dataset[['date', 'meantemp']]
        # print(df)

        # Step 3: Load algorithms configuration
        print("loading algorithms")
        algorithms_to_use = load_algorithm_config()
        print("algorithms loaded")
        print(algorithms_to_use)

        # Step 4: Initialize AnomalyDetectionManager
        anomaly_manager = AnomalyDetectionManager(algorithms_to_use)

        # Step 5: Perform anomaly detection
        result = anomaly_manager.detect_anomalies(df, dataset)

        # Step 6: Filter anomalies
        anomalies = result[result['is_anomaly'] == True]
        print(anomalies)
        
        anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))


        # Step 7: Prepare and send response
        response = {'anomalies': anomalies.to_dict(orient='records')}
        return jsonify(response), 200

    except Exception as e:
        # Handle unexpected errors gracefully
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
############################################################################################################
#
# Route: Recommend Rules
#
# to test this api you have the two following optios:
# 
# a - with curl command :
#
#   " make sure to comment out the scheduler in the app dir / __init__.py file other wise
#     this api will be invoked automatically every 1 minute"
#
#
#          1- in the terminal run python server.py
#          2- in another terminal run the following command:
#          3- curl -v http://localhost:5000/api/v1/recommendation/recommend_rules

# b - with postman :
#
#          " make sure to comment out the scheduler in the app dir / __init__.py file other wise
#            this api will be invoked automatically every 1 minute"
#
#          1- in the terminal run python server.py
#          2- open postman and create a new GET request with the following url http://localhost:5000/api/v1/recommendation/recommend_rules
#          3- click on the send button

@recommendation_bp.route('/recommend_rules', methods=['GET'])
def recommend_rules():
    try:
        # Create an instance of BayesianRecommendation using the global file path
        recommender = BayesianRecommendation(DATA_FILE_RECOMMENDATION)
        # Generate recommendations
        recommendations = recommender.recommend_rules()
        # Return as JSON
        return jsonify(recommendations)
    except FileNotFoundError as e:
        # Handle file-related errors
        return jsonify({"error": "Data file not found.", "details": str(e)}), 404
    except ValueError as e:
        # Handle issues like invalid data in the file
        return jsonify({"error": "Invalid data provided for recommendations.", "details": str(e)}), 400
    except Exception as e:
        # Catch-all for unexpected errors
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500
