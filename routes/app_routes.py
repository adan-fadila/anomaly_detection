from flask import Blueprint, request, jsonify
import pandas as pd
import json
from algorithm_manager.anomaly_detection_manager import AnomalyDetectionManager
from model.dataset_utils import Data_Set_Manager

anomaly_detection_bp = Blueprint('anomaly_detection', __name__)

def load_algorithm_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config.get("algorithms", ["stl"])

@anomaly_detection_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    try:
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        if 'sensor_values' not in data:
            return jsonify({'error': 'No sensor values found in the request'}), 400
        
        sensor_values = data['sensor_values']
        dfs = pd.DataFrame(columns=['date', 'meantemp'])
        for sensor_value in sensor_values:
            sensor_value = sensor_value.get("sensor_value", None)
            if sensor_value is None:
                return jsonify({'error': 'Missing sensor value'}), 400

            df = pd.DataFrame({
                'date': [pd.to_datetime('2017/01/02')],
                'meantemp': [sensor_value] 
            })
            # Drop columns in df that are empty or all-NA
            df_cleaned = df.dropna(axis=1, how='all')
            dfs = pd.concat([dfs, df_cleaned], ignore_index=True)
        
        algorithms_to_use = load_algorithm_config()

        manager = AnomalyDetectionManager(algorithms_to_use)
        dataset_utils = Data_Set_Manager()
        dataset = dataset_utils.process_dataset()
        result = manager.detect_anomalies(dfs, dataset)

        anomalies = result[result['is_anomaly'] == True]
        response = {
            'anomalies': anomalies.to_dict(orient='records')
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
