import os
import pandas as pd
import json
from flask import Blueprint, request, jsonify
from managers.anomaly_manager import AnomalyDetectionManager
from utils.data_manager import Data_Set_Manager
from algorithms.recommendations.bayesian import BayesianRecommendation
import base64
from config.constant import POINTWISE, COLLECTIVE

# Determine project root directory and dataset paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANOMALY_DATA_FRAME_FILE = os.path.join(BASE_DIR, 'data', 'logs', 'sensor_data_anomaly.csv')
ANOMALY_DATASET_FILE = os.path.join(BASE_DIR, 'data', 'csv', 'DailyDelhiClimateTrain.csv')
DATA_FILE_RECOMMENDATION = os.path.join(BASE_DIR, 'data', 'logs', 'sensor_data_recommendation.csv')

CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'config.json')

# Initialize Blueprints
anomaly_detection_bp = Blueprint('anomaly_detection', __name__)
recommendation_bp = Blueprint('recommendation', __name__)

############################################################################################################
# Route: Detect Anomalies with Payload
@anomaly_detection_bp.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Detect anomalies in sensor data.
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
                  sensor_value:
                    type: number
                  timestamp:
                    type: string
                    format: date-time
              example: [
                {"sensor_value": 70.0, "timestamp": "2023-01-01T10:00:00"},
                {"sensor_value": 72.5, "timestamp": "2023-01-01T10:10:00"}
              ]
    responses:
      200:
        description: Successfully detected anomalies.
        schema:
          type: object
          properties:
            anomalies:
              type: array
              items:
                type: object
                properties:
                  date:
                    type: string
                    format: date
                  meantemp:
                    type: number
      400:
        description: Invalid input data or format.
      500:
        description: Internal server error.
    """
    try:
        if request.content_type != 'application/json':
            return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415

        data = request.get_json()
        if not data or 'sensor_values' not in data:
            return jsonify({'error': 'No valid sensor values found in the request'}), 400

        sensor_values = data['sensor_values']
        if not isinstance(sensor_values, list) or not all(isinstance(val, (float, dict)) for val in sensor_values):
            return jsonify({'error': 'Invalid format for sensor_values. Must be a list of floats or objects.'}), 400

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

        anomaly_manager = AnomalyDetectionManager()
        dataset_manager = Data_Set_Manager(ANOMALY_DATA_FRAME_FILE)
        dataset = dataset_manager.process_dataset()

        result = anomaly_manager.detect_anomalies(dfs, dataset, POINTWISE)

        anomalies = result[result['is_anomaly'] == True]
        anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))

        response = {'anomalies': anomalies.to_dict(orient='records')}
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({'error': f'ValueError: {ve}'}), 400
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

############################################################################################################
# Route: Detect Anomalies with Dataset
@anomaly_detection_bp.route('/detect_dataset_anomalies', methods=['GET'])
def detect_dataset_anomalies():
    """
    Detect anomalies using a dataset.
    ---
    tags:
      - Anomaly Detection
    responses:
      200:
        description: Successfully detected anomalies.
        schema:
          type: object
          properties:
            response_anomaly:
              type: object
              properties:
                anomalies:
                  type: array
                  items:
                    type: object
                    properties:
                      date:
                        type: string
                        format: date
                      meantemp:
                        type: number
                plot_image:
                  type: string
                  format: byte
            response_collective:
              type: object
              properties:
                collective_plot:
                  type: string
                  format: byte
      500:
        description: Internal server error.
    """
    try:
        dataset_manager = Data_Set_Manager(ANOMALY_DATASET_FILE)
        dataset = dataset_manager.process_dataset()

        dataset_manager_2 = Data_Set_Manager(ANOMALY_DATA_FRAME_FILE)
        dataset_2 = dataset_manager_2.process_dataset()

        if dataset.empty:
            return jsonify({'error': 'The dataset is empty or could not be loaded'}), 500

        if 'date' not in dataset.columns or 'meantemp' not in dataset.columns:
            return jsonify({'error': "Required columns ('date', 'meantemp') are missing from the dataset"}), 500

        df = dataset_2[['date', 'meantemp']]

        anomaly_manager = AnomalyDetectionManager()
        anomaly_result, plot_image = anomaly_manager.detect_anomalies(df, dataset, POINTWISE)
        coll_anomaly, coll_plot = anomaly_manager.detect_anomalies(df, dataset, COLLECTIVE)

        anomalies = anomaly_result[anomaly_result['is_anomaly'] == True]
        anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))

        plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')
        coll_plot_base64 = base64.b64encode(coll_plot.getvalue()).decode('utf-8')

        response = {
            'response_anomaly': {
                'anomalies': anomalies.to_dict(orient='records'),
                'plot_image': plot_image_base64,
            },
            'response_collective': {
                'collective_plot': coll_plot_base64
            }
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

############################################################################################################
# Route: Recommend Rules
@recommendation_bp.route('/recommend_rules', methods=['GET'])
def recommend_rules():
    """
    Recommend rules based on sensor data.
    ---
    tags:
      - Recommendations
    responses:
      200:
        description: Successfully generated recommendations.
        schema:
          type: object
          additionalProperties:
            type: string
      404:
        description: Data file not found.
      500:
        description: Internal server error.
    """
    try:
        recommender = BayesianRecommendation(DATA_FILE_RECOMMENDATION)
        recommendations = recommender.recommend_rules()
        return jsonify(recommendations)
    except FileNotFoundError as e:
        return jsonify({"error": "Data file not found.", "details": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": "Invalid data provided for recommendations.", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500
