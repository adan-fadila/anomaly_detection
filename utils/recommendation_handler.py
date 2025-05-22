import os
from .logger import logger
from algorithms.recommendations.bayesian import BayesianRecommendation
from .node_communicator import NodeCommunicator

class RecommendationHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_frame_file = os.path.join(base_dir, 'SmartSchool-Server', 'logs', 'sensor_data.csv')
        self.data_file =os.path.join(base_dir, 'SmartSchool-Server', 'logs', 'sensor_data.csv')
        self.node_communicator = NodeCommunicator()
        self.last_modified_time = None
        self.last_recommendations = None
        self.logger=logger

    def process_and_send_recommendations(self):
        try:
            # Fetch recommendations using BayesianRecommendation
            recommender = BayesianRecommendation(self.data_file)
            recommendations = recommender.recommend_rules()
            self.logger.info("Fetched recommendations successfully.")
            
            # Check if there's an error in the recommendations
            if "error" in recommendations:
                self.logger.error(f"Error in recommendations: {recommendations['error']}")
                return
            
            # Get the current recommended rules
            current_rules = recommendations.get("recommended_rules", [])
            
            # Compare with last sent recommendations
            if self.last_recommendations is None:
                # First time - send all recommendations
                self.logger.info("First time generating recommendations - sending all rules")
                send_recommendations = True
            elif set(current_rules) != set(self.last_recommendations):
                # Rules have changed - send new recommendations
                self.logger.info("Recommendations have changed - sending updated rules")
                send_recommendations = True
            else:
                # No changes - don't send
                self.logger.info("No changes in recommendations - not sending")
                send_recommendations = False
            
            # Send updates to node server if there are differences
            if send_recommendations:
                # Send the complete response as returned by recommend_rules()
                self.node_communicator.send_to_node('recommendation', recommendations)
                self.logger.info(f"Recommendations sent successfully. Sent {len(current_rules)} rules.")
                
                # Update last sent recommendations
                self.last_recommendations = current_rules.copy()
            else:
                self.logger.info("No new recommendations to send.")
        
        except Exception as e:
            self.logger.error(f"Error in recommendation processing: {e}")
            self.logger.error("Exception details:", exc_info=True)

    def check_logs(self):
        if not os.path.exists(self.data_frame_file):
            logger.warning(f"File not found: {self.data_frame_file}")
            return

        current_modified_time = os.path.getmtime(self.data_frame_file)
        if self.last_modified_time is None or current_modified_time > self.last_modified_time:
            self.last_modified_time = current_modified_time
            logger.info("New recommendation logs detected! Triggering recommendation rules...")
            self.process_and_send_recommendations()
        else:
            logger.info("No new recommendation logs detected.")