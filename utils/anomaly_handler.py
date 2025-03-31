import os
import base64
import io
from .logger import logger
from utils.data_manager import Data_Set_Manager
from managers.anomaly_manager import AnomalyDetectionManager
from config.constant import POINTWISE, COLLECTIVE,SEASONALITY,TREND
from .node_communicator import NodeCommunicator
from config.constant import FEATEURES
class AnomalyHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dataset_file = os.path.join(base_dir, 'data', 'csv', 'DailyDelhiClimateTrain.csv')
        self.data_frame_file = os.path.join(base_dir, 'data', 'logs', 'sensor_data_anomaly.csv')
        self.node_communicator = NodeCommunicator()
        self.last_modified_time = None

    def detect_anomalies(self,feature,new_lines, anomaly_type):

        try:
            # Initialize dataset managers and process datasets
            dataset_manager = Data_Set_Manager(self.dataset_file)
            dataset = dataset_manager.process_dataset(feature=feature)

            data_frame_file = io.StringIO(''.join(new_lines))
        
        # Initialize dataset managers
        # For the new data that was just appended
            # dataset_manager_2 = Data_Set_Manager(data_frame_file)
            # dataset_2 = dataset_manager_2.process_dataset(feature=feature)
            
            # For the historical data (full dataset)
            dataset_manager_2 = Data_Set_Manager(data_frame_file)
            dataset_2 = dataset_manager_2.process_dataset(feature=feature)

            # Validate datasets
            if dataset.empty:
                logger.error("The dataset is empty or could not be loaded.")
                return {'error': 'The dataset is empty or could not be loaded'}

            required_columns = ['date', feature]
            missing_columns = [col for col in required_columns if col not in dataset.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return {'error': f"Required columns ({', '.join(missing_columns)}) are missing from the dataset"}

            # Extract relevant data
            df = dataset_2[['date', feature]]

            # Initialize anomaly detection manager
            anomaly_manager = AnomalyDetectionManager()

            # Detect anomalies
            anomaly_result,plot_image = anomaly_manager.detect_anomalies(df, dataset, anomaly_type,feature)
            if len(anomaly_result) == 0:
                return None
            # coll_anomaly, coll_plot = anomaly_manager.detect_anomalies(df, dataset, COLLECTIVE)

            # Extract anomalies and format the date
            if(anomaly_type == POINTWISE):
                anomalies = anomaly_result
                anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                    'plot_image': plot_image_base64,
                    'name': "living room temperature pointwise anomaly"
                }
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            elif(anomaly_type == SEASONALITY):
                anomalies = anomaly_result
                

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                    'plot_image': plot_image_base64,
                    'name': "living room temperature seasonality anomaly"
                }
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            elif(anomaly_type == SEASONALITY):
                anomalies = anomaly_result

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                     'plot_image': plot_image_base64,
                }
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            elif(anomaly_type == TREND):
                anomalies = anomaly_result

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                     'plot_image': plot_image_base64,
                     'name': "living room temperature trend anomaly"
                }
                self.node_communicator.send_to_node('trend_anomaly', anomaly_response)
            

            return anomaly_response

        except Exception as e:
            logger.exception(f"Error in anomaly detection: {e}")
            return {'error': str(e)}

    def check_logs(self):
        if not os.path.exists(self.data_frame_file):
            logger.warning(f"File not found: {self.dataset_file}")
            return
            
        current_modified_time = os.path.getmtime(self.data_frame_file)
        
        # Initialize tracking variables if they don't exist
        if not hasattr(self, 'last_trend_line'):
            self.last_trend_line = 0
        if not hasattr(self, 'last_seasonal_line'):
            self.last_seasonal_line = 0
        if not hasattr(self, 'last_pointwise_line'):
            self.last_pointwise_line = 0
        
        if self.last_modified_time is None or current_modified_time > self.last_modified_time:
            self.last_modified_time = current_modified_time
            logger.info("New anomaly logs detected! Checking for anomalies...")
            
            # Read all lines from the file
            with open(self.data_frame_file, 'r') as file:
                all_lines = file.readlines()
            
            current_line_count = len(all_lines)
            
            # Check if we should run trend anomaly detection
           
            
            # Check if we should run seasonal anomaly detection
            if current_line_count - self.last_seasonal_line >= 100:
                logger.info(f"Found {current_line_count - self.last_seasonal_line} new lines. Running seasonal anomaly detection.")
                # Pass only the new lines for seasonal detection
                new_lines_for_seasonal = all_lines[self.last_seasonal_line:current_line_count]
                self.detect_for_features(new_lines_for_seasonal, SEASONALITY)
                self.last_seasonal_line = current_line_count
            
            # # For pointwise, we can run it on any new data
            if current_line_count - self.last_pointwise_line <=50:
                logger.info(f"Found {current_line_count - self.last_pointwise_line} new lines. Running pointwise anomaly detection.")
                # Pass only the new lines for pointwise detection
                new_lines_for_pointwise = all_lines[self.last_pointwise_line:current_line_count]
                self.detect_for_features(new_lines_for_pointwise, POINTWISE)
                self.last_pointwise_line = current_line_count


            if current_line_count - self.last_trend_line >= 400:
                logger.info(f"Found {current_line_count - self.last_trend_line} new lines. Running trend anomaly detection.")
                # Pass only the new lines for trend detection
                new_lines_for_trend = all_lines[self.last_trend_line:current_line_count]
                self.detect_for_features(new_lines_for_trend, TREND)    
                self.last_trend_line = current_line_count
            
            # If no anomaly detection was run
            # if (current_line_count - self.last_trend_line < 30 and 
            #     current_line_count - self.last_seasonal_line < 20 and 
            #     current_line_count <= self.last_pointwise_line):
            #     logger.info("Not enough new data for any anomaly detection.")
        else:
            logger.info("No new anomaly logs detected.")

    def detect_for_features(self,newline,anomaly_type):
        for feature in FEATEURES:
            self.detect_anomalies(feature,newline,anomaly_type)