import requests
from .logger import logger
import json
class NodeCommunicator:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.endpoints = {
            'anomaly': "/anomaly_response",
            'collective_anomaly': "/coll_anomaly_response",
            'recommendation': "/recommendation_response"
        }

    def send_to_node(self, endpoint_key, data):
        try:
            api_url = f"{self.base_url}{self.endpoints[endpoint_key]}"
            response = requests.post(api_url, json=(data))
            logger.info(f"Sent data to {api_url}, Node.js response status: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Error sending data to {endpoint_key} endpoint: {e}")
            raise
    