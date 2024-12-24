from flask import Flask
from flasgger import Swagger
from app.routes import recommendation_bp, anomaly_detection_bp 
from config.config import Config
from utils.logs_monitor import start_logs_monitor


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize Swagger
    swagger = Swagger(app)
    
    # Start the feed monitor scheduler
    start_logs_monitor()


    # Register Blueprints
    app.register_blueprint(recommendation_bp, url_prefix='/api/v1/recommendation')
    app.register_blueprint(anomaly_detection_bp, url_prefix='/api/v1/anomaly_detection')

    return app
