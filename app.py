# # # curl -v http://localhost:5000/detect_anomalies -H "Content-Type: application/json" -d "{\"sensor_values\": [{\"sensor_value\": 70.0}, {\"sensor_value\": 72.5}, {\"sensor_value\": 68.4}, {\"sensor_value\": 75.3}, {\"sensor_value\": 74.1}, {\"sensor_value\": 69.9}, {\"sensor_value\": 71.2}, {\"sensor_value\": 70.8}, {\"sensor_value\": 69.5}, {\"sensor_value\": 73.4}]}"
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from routes.app_routes import anomaly_detection_bp
from scripts.baysian_model_script import BayesianModel
import requests
import os

# Initialize Flask app
app = Flask(__name__)

# Dynamically determine the dataset file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'dataset', 'mock_data.csv')

# Initialize the Bayesian Model
bayesian_model = BayesianModel(DATA_FILE)

# Register the Blueprint for anomaly detection routes
app.register_blueprint(anomaly_detection_bp)

# Node.js server URL for recommendations
NODEJS_SERVER_URL = 'http://localhost:8001/api/recommendations'

# Function to run the Bayesian model and send recommendations
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

# Initialize background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_bayesian_model, 'interval', minutes=2)
scheduler_started = False  # Add a flag to ensure the scheduler starts only once

@app.before_request
def start_scheduler():
    global scheduler_started
    if not scheduler_started:
        print("Starting Scheduler...")
        run_bayesian_model()  # Execute once during app startup
        scheduler.start()
        scheduler_started = True

@app.teardown_appcontext
def shutdown_scheduler(exception=None):
    if scheduler.started:
        scheduler.shutdown()

if __name__ == '__main__':
    app.run(debug=True)
