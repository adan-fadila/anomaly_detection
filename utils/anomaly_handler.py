import os
import base64
from .logger import logger
from utils.data_manager import Data_Set_Manager
from managers.anomaly_manager import AnomalyDetectionManager
from config.constant import POINTWISE, COLLECTIVE
from .node_communicator import NodeCommunicator

class AnomalyHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dataset_file = os.path.join(base_dir, 'data', 'csv', 'DailyDelhiClimateTrain.csv')
        self.data_frame_file = os.path.join(base_dir, 'data', 'logs', 'sensor_data_anomaly.csv')
        self.node_communicator = NodeCommunicator()
        self.last_modified_time = None

    def detect_anomalies(self):
        try:
            # Initialize dataset managers and process datasets
            dataset_manager = Data_Set_Manager(self.dataset_file)
            dataset = dataset_manager.process_dataset()

            dataset_manager_2 = Data_Set_Manager(self.data_frame_file)
            dataset_2 = dataset_manager_2.process_dataset()

            # Validate datasets
            if dataset.empty:
                logger.error("The dataset is empty or could not be loaded.")
                return {'error': 'The dataset is empty or could not be loaded'}

            required_columns = ['date', 'meantemp']
            missing_columns = [col for col in required_columns if col not in dataset.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return {'error': f"Required columns ({', '.join(missing_columns)}) are missing from the dataset"}

            # Extract relevant data
            df = dataset_2[['date', 'meantemp']]

            # Initialize anomaly detection manager
            anomaly_manager = AnomalyDetectionManager()

            # Detect anomalies
            anomaly_result, plot_image = anomaly_manager.detect_anomalies(df, dataset, POINTWISE)
            coll_anomaly, coll_plot = anomaly_manager.detect_anomalies(df, dataset, COLLECTIVE)

            # Extract anomalies and format the date
            anomalies = anomaly_result[anomaly_result['is_anomaly'] == True]
            anomalies['date'] = anomalies['date'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d'))

            # Encode plots to Base64
            plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')
            coll_plot_base64 = base64.b64encode(coll_plot.getvalue()).decode('utf-8')

            # Prepare response data
            anomaly_response = {
                'anomalies': anomalies.to_dict(orient='records'),
                'plot_image': plot_image_base64,
            }
            collective_response = {
                'collective_plot': coll_plot_base64
            }

            # Send data to Node.js endpoints
            self.node_communicator.send_to_node('anomaly', anomaly_response)
            self.node_communicator.send_to_node('collective_anomaly', collective_response)

            return {
                'response_anomaly': anomaly_response,
                'response_collective': collective_response
            }

        except Exception as e:
            logger.exception(f"Error in anomaly detection: {e}")
            return {'error': str(e)}

    def check_logs(self):
        if not os.path.exists(self.dataset_file):
            logger.warning(f"File not found: {self.dataset_file}")
            return

        current_modified_time = os.path.getmtime(self.dataset_file)
        if self.last_modified_time is None or current_modified_time > self.last_modified_time:
            self.last_modified_time = current_modified_time
            logger.info("New anomaly logs detected! Triggering anomaly detection...")
            self.detect_anomalies()
        else:
            logger.info("No new anomaly logs detected.")