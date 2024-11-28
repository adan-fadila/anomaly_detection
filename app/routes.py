import os
import pandas as pd
import json
from flask import Blueprint, request, jsonify
from managers.anomaly_manager import AnomalyDetectionManager
from utils.data_manager import Data_Set_Manager
from algorithms.recommendations.bayesian import BayesianRecommendation

# Determine project root directory and dataset path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'csv', 'DailyDelhiClimateTrain.csv')
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')

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
        return config.get("algorithms")
    except FileNotFoundError:
        return ["stl", "arima", "sarima", "dbscan"]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

# Route: Detect Anomalies
@anomaly_detection_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Detect anomalies in sensor data
    ---
    tags:
      - Anomaly Detection
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            sensor_values:
              type: array
              items:
                type: object
                properties:
                  timestamp:
                    type: string
                    format: date-time
                  sensor_value:
                    type: number
    responses:
      200:
        description: Successfully detected anomalies
        schema:
          type: object
          properties:
            anomalies:
              type: array
              items:
                type: object
      400:
        description: Invalid input
      500:
        description: Internal server error
    """
    try:
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415

        data = request.get_json()
        if not data or 'sensor_values' not in data:
            return jsonify({'error': 'No valid sensor values found in the request'}), 400

        sensor_values = data['sensor_values']
        dfs = pd.DataFrame([
            {'date': pd.to_datetime(sensor.get("timestamp", "2017/01/02")), 'meantemp': sensor.get("sensor_value")}
            for sensor in sensor_values if sensor.get("sensor_value") is not None
        ])
        if dfs.empty:
            return jsonify({'error': 'No valid sensor values provided'}), 400

        algorithms_to_use = load_algorithm_config()
        anomaly_manager = AnomalyDetectionManager(algorithms_to_use)
        dataset_manager = Data_Set_Manager(DATA_FILE)
        dataset = dataset_manager.process_dataset()

        result = anomaly_manager.detect_anomalies(dfs, dataset)
        anomalies = result[result['is_anomaly'] == True]

        response = {'anomalies': anomalies.to_dict(orient='records')}
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({'error': f'ValueError: {ve}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {e}'}), 500

# Route: Recommend Rules
@recommendation_bp.route('/recommend_rules', methods=['GET'])
def recommend_rules():
    """
    Get recommendation rules
    ---
    tags:
      - Recommendations
    responses:
      200:
        description: Successfully retrieved recommendations
        schema:
          type: array
          items:
            type: string
      500:
        description: Internal server error
    """
    try:
        bayesian_model = BayesianRecommendation(DATA_FILE)
        recommendations = bayesian_model.recommend_rules()
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {e}'}), 500
