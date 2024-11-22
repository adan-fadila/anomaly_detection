import os
import pandas as pd
import json
from flask import Blueprint, request, jsonify
from algorithm_manager.anomaly_detection_manager import AnomalyDetectionManager
from model.dataset_utils import Data_Set_Manager
from scripts.baysian_model_script import BayesianModel

# Determine project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'dataset', 'mock_data.csv')

# Initialize Blueprint
anomaly_detection_bp = Blueprint('anomaly_detection', __name__)

# Load algorithm configuration
def load_algorithm_config():
    try:
        config_path = os.path.join(BASE_DIR, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get("algorithms", ["stl"])
    except FileNotFoundError:
        return ["stl"]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

# Route: Detect Anomalies
@anomaly_detection_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    try:
        # Validate Content-Type
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415

        # Parse request JSON data
        data = request.get_json()
        if not data or 'sensor_values' not in data:
            return jsonify({'error': 'No valid sensor values found in the request'}), 400

        sensor_values = data['sensor_values']
        dfs = pd.DataFrame(columns=['date', 'meantemp'])

        # Populate DataFrame
        for sensor in sensor_values:
            sensor_value = sensor.get("sensor_value")
            if sensor_value is None:
                return jsonify({'error': 'Missing sensor value in one or more entries'}), 400

            dfs = pd.concat([dfs, pd.DataFrame({'date': [pd.to_datetime('2017/01/02')], 'meantemp': [sensor_value]})], ignore_index=True)

        # Load algorithm configuration
        algorithms_to_use = load_algorithm_config()

        # Initialize anomaly detection manager
        manager = AnomalyDetectionManager(algorithms_to_use)
        dataset_utils = Data_Set_Manager()
        dataset = dataset_utils.process_dataset()

        # Detect anomalies
        result = manager.detect_anomalies(dfs, dataset)
        anomalies = result[result['is_anomaly'] == True]

        # Prepare response
        response = {'anomalies': anomalies.to_dict(orient='records')}
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({'error': f'ValueError: {ve}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {e}'}), 500

# Route: Recommend Rules
@anomaly_detection_bp.route('/recommend_rules', methods=['GET'])
def recommend_rules():
    try:
        print("Processing recommendations...")
        recommendations = BayesianModel(DATA_FILE).recommend_rules()
        print(f"Recommendations generated: {recommendations}")
        return jsonify(recommendations), 200
    except Exception as e:
        print(f"Error in recommend_rules: {e}")
        return jsonify({'error': f'Internal Server Error: {e}'}), 500

