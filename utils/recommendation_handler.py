import os
from .logger import logger
from algorithms.recommendations.bayesian import BayesianRecommendation
from .node_communicator import NodeCommunicator

class RecommendationHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_file = os.path.join(base_dir, "data", "logs", "sensor_data_recommendation.csv")
        self.node_communicator = NodeCommunicator()
        self.last_modified_time = None
        self.last_recommendations = None

    def process_and_send_recommendations(self):
        try:
            # Fetch recommendations
            recommender = BayesianRecommendation(self.data_file)
            recommendations = recommender.recommend_rules()
            logger.info("Fetched recommendations successfully.")

            # Process recommendations
            processed_recommendations = {}
            for rec in recommendations:
                feature = rec["device"]
                if feature not in processed_recommendations:
                    processed_recommendations[feature] = []
                processed_recommendations[feature].append({
                    "recommendation": rec["recommendation"],
                    "recommended_time": rec["recommended_time"]
                })

            # Identify differences
            differences = {
                key: processed_recommendations[key]
                for key in processed_recommendations
                if self.last_recommendations is None or 
                self.last_recommendations.get(key) != processed_recommendations[key]
            }

            # Send updates to node server
            if differences:
                self.node_communicator.send_to_node('recommendation', differences)
                logger.info("Recommendations sent successfully.")
            else:
                logger.info("No new recommendations to send.")

            self.last_recommendations = processed_recommendations

        except Exception as e:
            logger.error(f"Error in recommendation processing: {e}")

    def check_logs(self):
        if not os.path.exists(self.data_file):
            logger.warning(f"File not found: {self.data_file}")
            return

        current_modified_time = os.path.getmtime(self.data_file)
        if self.last_modified_time is None or current_modified_time > self.last_modified_time:
            self.last_modified_time = current_modified_time
            logger.info("New recommendation logs detected! Triggering recommendation rules...")
            self.process_and_send_recommendations()
        else:
            logger.info("No new recommendation logs detected.")