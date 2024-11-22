# from flask import Flask,request,jsonify,json
# import pandas as pd
# from datetime import datetime
# from algorithm_manager.anomaly_detection_manager import AnomalyDetectionManager
# from model.dataset_utils import Data_Set_Manager

# app = Flask(__name__)


# # Download latest version



# def load_algorithm_config():
#     with open('config.json', 'r') as f:
#         config = json.load(f)
#     return config.get("algorithms", ["stl"])

# @app.route('/detect_anomalies', methods=['POST'])
# def detect_anomalies():
#     try:
#         # Get the data from the request (assume JSON format)
#         print(f"Raw data: {request.data}")
#         if request.content_type != 'application/json':
#             return jsonify({"error": "Invalid Content-Type, expected 'application/json'"}), 415
#         data = request.get_json()
#         if data is None:
#             return jsonify({"error": "Invalid JSON data"}), 400
#         print("values")
#         if 'sensor_values' not in data:
#             return jsonify({'error': 'No sensor values found in the request'}), 400
        
#         sensor_values = data['sensor_values']
#         dfs = pd.DataFrame(columns=['date', 'meantemp'])
#         # Example: Extract the sensor_value from the first entry
#         for sensor_value in sensor_values:
#             sensor_value = sensor_value.get("sensor_value", None)
#             if sensor_value is None :
#                 return jsonify({'error': 'Missing sensor value'}), 400

#         # Create the DataFrame
#             df = pd.DataFrame({
#                 'date': [pd.to_datetime('2017/01/02')],
#                 'meantemp': [sensor_value]  # Use the actual sensor_value here
#             })
#             dfs = pd.concat([dfs, df], ignore_index=True)
#         algorithms_to_use = load_algorithm_config()

#         manager = AnomalyDetectionManager(algorithms_to_use)
#         dataset_utils = Data_Set_Manager()
#         dataset = dataset_utils()
#         result = manager.detect_anomalies(dfs,dataset)

#         anomalies = result[result['is_anomaly'] == True]
#         response = {
#             'anomalies': anomalies.to_dict(orient='records')
#         }

#         return jsonify(response),200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)



# # # curl -v http://localhost:5000/detect_anomalies -H "Content-Type: application/json" -d "{\"sensor_values\": [{\"sensor_value\": 70.0}, {\"sensor_value\": 72.5}, {\"sensor_value\": 68.4}, {\"sensor_value\": 75.3}, {\"sensor_value\": 74.1}, {\"sensor_value\": 69.9}, {\"sensor_value\": 71.2}, {\"sensor_value\": 70.8}, {\"sensor_value\": 69.5}, {\"sensor_value\": 73.4}]}"
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from routes.app_routes import anomaly_detection_bp
from scripts.baysian_model_script import BayesianModel
import requests

app = Flask(__name__)
app.register_blueprint(anomaly_detection_bp)

data_file = 'data.csv'
bayesian_model = BayesianModel(data_file)

NODEJS_SERVER_URL = 'http://localhost:8001/api/recommendations'


def run_bayesian_model():
    print("Running Bayesian Model...")
    recommendations = bayesian_model.recommend_rules()
    try:
        response = requests.post(NODEJS_SERVER_URL, json={'recommendations': recommendations})
        if response.status_code == 200:
            print("Recommendations sent successfully.")
        else:
            print(f"Failed to send recommendations: {response.status_code}")
    except Exception as e:
        print(f"Error sending recommendations: {e}")


# Set up the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_bayesian_model, 'interval', minutes=30)
scheduler.start()

# Shut down the scheduler when exiting the app
@app.before_first_request
def start_scheduler():
    run_bayesian_model()  # Run once at startup

@app.teardown_appcontext
def shutdown_scheduler(exception=None):
    scheduler.shutdown()


if __name__ == '__main__':
    app.run(debug=True)
