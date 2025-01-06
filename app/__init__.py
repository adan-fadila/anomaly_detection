import matplotlib
matplotlib.use('Agg')  # on a mac matplotlib should run on the main thread.

from flask import Flask
from flasgger import Swagger
from app.routes import recommendation_bp, anomaly_detection_bp 
from config.config import Config
from utils.anomaly_monitor import start_anomaly_monitor
from utils.recommendation_monitor import start_recommendation_monitor

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize Swagger
    swagger = Swagger(app)
    
    # Start the feed monitor scheduler
    start_anomaly_monitor()
    start_recommendation_monitor()

    # Register Blueprints
    app.register_blueprint(recommendation_bp, url_prefix='/api/v1/recommendation')
    app.register_blueprint(anomaly_detection_bp, url_prefix='/api/v1/anomaly_detection')

    return app
