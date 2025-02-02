import matplotlib
matplotlib.use('Agg')  # on a mac matplotlib should run on the main thread.

from flask import Flask
from flasgger import Swagger
from app.routes import recommendation_bp, anomaly_detection_bp 
from config.config import Config
from utils.monitor import Monitor

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize Swagger
    swagger = Swagger(app)
    
    # Initialize and start the unified monitor
    monitor = Monitor()
    monitor.start_monitor()

    # Register Blueprints
    app.register_blueprint(recommendation_bp, url_prefix='/api/v1/recommendation')
    app.register_blueprint(anomaly_detection_bp, url_prefix='/api/v1/anomaly_detection')

    return app

