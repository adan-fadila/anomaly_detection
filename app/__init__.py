from flask import Flask
from flasgger import Swagger
from app.routes import recommendation_bp, anomaly_detection_bp
from config.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize Swagger
    swagger = Swagger(app)

    # Register Blueprints
    app.register_blueprint(recommendation_bp, url_prefix='/api/v1/recommendation')
    app.register_blueprint(anomaly_detection_bp, url_prefix='/api/v1/anomaly_detection')

    return app
