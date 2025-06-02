import os
import json
from datetime import datetime, timedelta
from .logger import logger
from algorithms.recommendations.bayesian import BayesianRecommendation
from .node_communicator import NodeCommunicator

class RecommendationHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_frame_file = os.path.join(base_dir, 'SmartSchool-Server', 'logs', 'sensor_data.csv')
        self.data_file = os.path.join(base_dir, 'SmartSchool-Server', 'logs', 'sensor_data.csv')
        self.node_communicator = NodeCommunicator()
        self.last_modified_time = None
        self.last_recommendations = None
        self.logger = logger
        
        # File to store last sent recommendations persistently
        self.last_rules_file = os.path.join(base_dir, 'anomaly_detection', 'logs', 'last_sent_rules.json')
        
        # DEBUG: Log the file path
        self.logger.info(f"JSON file will be saved to : {self.last_rules_file}")
        
    def _load_last_sent_rules(self):
        """Load the last sent rules from persistent storage"""
        try:
            self.logger.info(f"Attempting to load rules from: {self.last_rules_file}")
            self.logger.info(f"File exists: {os.path.exists(self.last_rules_file)}")
            
            if os.path.exists(self.last_rules_file):
                with open(self.last_rules_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if file is older than 1 week - if so, delete it
                file_date = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
                if datetime.now() - file_date > timedelta(days=7):
                    os.remove(self.last_rules_file)
                    self.logger.info("Last rules file was older than 1 week - deleted")
                    return []
                    
                rules = data.get('rules', [])
                self.logger.info(f"Loaded {len(rules)} rules from file")
                return rules
            else:
                self.logger.info("No existing rules file found")
                return []
        except Exception as e:
            self.logger.error(f"Error loading last sent rules: {e}")
            return []
    
    def _save_last_sent_rules(self, rules):
        """Save the current rules to persistent storage (overwrites previous)"""
        try:
            self.logger.info(f"Attempting to save {len(rules)} rules to: {self.last_rules_file}")
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(self.last_rules_file)
            self.logger.info(f"Checking directory: {directory}")
            self.logger.info(f"Directory exists: {os.path.exists(directory)}")
            
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Directory created/exists: {os.path.exists(directory)}")
            
            data = {
                'rules': rules,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.last_rules_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Verify file was created
            if os.path.exists(self.last_rules_file):
                file_size = os.path.getsize(self.last_rules_file)
                self.logger.info(f"✅ JSON file saved successfully! Size: {file_size} bytes")
                self.logger.info(f"File location: {self.last_rules_file}")
            else:
                self.logger.error("❌ JSON file was not created!")
                
        except Exception as e:
            self.logger.error(f"❌ Error saving last sent rules: {e}")
            self.logger.error("Exception details:", exc_info=True)
    
    def _get_new_rules_only(self, current_rules, last_sent_rules):
        """Compare rules one by one and return only new ones"""
        new_rules = []
        for rule in current_rules:
            if rule not in last_sent_rules:
                new_rules.append(rule)
        
        self.logger.info(f"Comparison result: {len(new_rules)} new rules out of {len(current_rules)} total rules")
        return new_rules

    def process_and_send_recommendations(self):
        try:
            self.logger.info("🔄 Starting recommendation processing...")
            
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
            self.logger.info(f"Current rules count: {len(current_rules)}")
            
            # Load last sent rules from persistent storage
            last_sent_rules = self._load_last_sent_rules()
            self.logger.info(f"Last sent rules count: {len(last_sent_rules)}")
            
            # Compare with last sent recommendations
            if not last_sent_rules:
                # First time - send all recommendations (keep original body format)
                self.logger.info("🆕 First time generating recommendations - sending all rules")
                self.node_communicator.send_to_node('recommendation', recommendations)
                self.logger.info(f"Recommendations sent successfully. Sent {len(current_rules)} rules.")
                
                # Save current rules as last sent
                self._save_last_sent_rules(current_rules)
                
            else:
                # Get only new rules by comparing one by one
                new_rules = self._get_new_rules_only(current_rules, last_sent_rules)
                
                if new_rules:
                    # Create body with original format but only new rules
                    new_recommendations = recommendations.copy()
                    new_recommendations["recommended_rules"] = new_rules
                    
                    self.logger.info(f"📤 Found {len(new_rules)} new rules - sending only new ones")
                    self.node_communicator.send_to_node('recommendation', new_recommendations)
                    self.logger.info(f"New recommendations sent successfully. Sent {len(new_rules)} new rules.")
                    
                    # Save ALL current rules (not just new ones) as last sent
                    self._save_last_sent_rules(current_rules)
                else:
                    # No new rules - don't send
                    self.logger.info("✅ No new recommendations to send.")
        
        except Exception as e:
            self.logger.error(f"❌ Error in recommendation processing: {e}")
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
    
    def debug_file_status(self):
        """Debug method to check file status"""
        self.logger.info(f"=== DEBUG FILE STATUS ===")
        self.logger.info(f"Expected file path: {self.last_rules_file}")
        self.logger.info(f"File exists: {os.path.exists(self.last_rules_file)}")
        self.logger.info(f"Directory exists: {os.path.exists(os.path.dirname(self.last_rules_file))}")
        
        if os.path.exists(self.last_rules_file):
            size = os.path.getsize(self.last_rules_file)
            mtime = os.path.getmtime(self.last_rules_file)
            self.logger.info(f"File size: {size} bytes")
            self.logger.info(f"Last modified: {datetime.fromtimestamp(mtime)}")
        
        # List all files in the logs directory
        logs_dir = os.path.dirname(self.last_rules_file)
        if os.path.exists(logs_dir):
            files = os.listdir(logs_dir)
            self.logger.info(f"Files in logs directory: {files}")
        
        self.logger.info(f"=========================")