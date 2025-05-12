import os
import base64
import io
from .logger import logger
from utils.data_manager import Data_Set_Manager
from managers.anomaly_manager import AnomalyDetectionManager
from config.constant import POINTWISE, COLLECTIVE,SEASONALITY,TREND,POINTWISE_WINDOW_SIZE,SEASONALITY_WINDOW_SIZE,TREND_WINDOW_SIZE,POINTWISE_STEP_SIZE,SEASONALITY_STEP_SIZE,COLLECTIVE_STEP_SIZE,COLLECTIVE_WINDOW_SIZE,TREND_STEP_SIZE
from .node_communicator import NodeCommunicator
from config.constant import FEATEURES,COLUMNS
import pandas as pd
class AnomalyHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_frame_file = os.path.join(base_dir, 'SmartSchool-Server', 'logs', 'sensor_data.csv')
        self.node_communicator = NodeCommunicator()
        self.last_modified_time = None

    def detect_anomalies(self,feature,new_lines, anomaly_type):

        try:
            all_lines = [COLUMNS] + new_lines
            data_frame_file = io.StringIO(''.join(all_lines))
            dataset_2 = pd.read_csv(data_frame_file)
            if(dataset_2 is None ):
                print("Dataset is None")
                return None
            required_columns = ['timestamp', feature]
            df = dataset_2[required_columns]
            if df is None or df.empty:
                print("DataFrame is None or empty")
                return None

            anomaly_manager = AnomalyDetectionManager()

            anomaly_result,plot_image = anomaly_manager.detect_anomalies(df, df, anomaly_type,feature)
            if len(anomaly_result) == 0:
                return None
            print("Anomaly result:", anomaly_result)
            
            if(anomaly_type == POINTWISE):
                anomaly_result=anomaly_result.to_dict(orient="records")
                anomalies = anomaly_result

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                    'plot_image': plot_image_base64,
                    'name': f"living room {feature} pointwise anomaly"
                }
                for row in anomaly_response['anomalies']:
                    for k, v in row.items():
                        if isinstance(v, pd.Timestamp):
                            row[k] = str(v)
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            elif(anomaly_type == COLLECTIVE):
                anomalies = anomaly_result
                

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                    'plot_image': plot_image_base64,
                    'name': f"living room {feature} collective anomaly"
                }
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            elif(anomaly_type == COLLECTIVE):
                anomalies = anomaly_result

                # Encode plots to Base64
                plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

                # Prepare response data
                anomaly_response = {
                    'anomalies': anomalies,
                    'plot_image': plot_image_base64,
                    'name': "living room temperature colective anomaly"
                }
               
                self.node_communicator.send_to_node('anomaly', anomaly_response)
            

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
        if not hasattr(self, 'last_Col_line'):
            self.last_Col_line = 0
        if not hasattr(self, 'last_pointwise_line'):
            self.last_pointwise_line = 0        
        if self.last_modified_time is None or current_modified_time > self.last_modified_time:
            self.last_modified_time = current_modified_time
            logger.info("New anomaly logs detected! Checking for anomalies...")
            
            # Read all lines from the file
            with open(self.data_frame_file, 'r') as file:
                all_lines = file.readlines()
            
            current_line_count = len(all_lines) - 1

            if current_line_count - self.last_pointwise_line >=POINTWISE_STEP_SIZE and current_line_count > POINTWISE_WINDOW_SIZE:
                print(f"Found {current_line_count - self.last_pointwise_line} new lines. Running pointwise anomaly detection.")
                # Pass only the new lines for pointwise detection
                new_lines_for_pointwise = all_lines[-(POINTWISE_WINDOW_SIZE + POINTWISE_STEP_SIZE):]
                self.detect_for_features(new_lines_for_pointwise, POINTWISE)
                self.last_pointwise_line = current_line_count

            if current_line_count - self.last_Col_line >=COLLECTIVE_STEP_SIZE and current_line_count > 2*POINTWISE_WINDOW_SIZE:
                print(f"Found {current_line_count - self.last_Col_line} new lines. Running collective anomaly detection.")
                # Pass only the new lines for pointwise detection
                new_lines_for_pointwise = all_lines[-2*(COLLECTIVE_WINDOW_SIZE):]
                self.detect_for_features(new_lines_for_pointwise, COLLECTIVE)
                self.last_Col_line = current_line_count

           
        else:
            logger.info("No new anomaly logs detected.")

    def detect_for_features(self,newline,anomaly_type):
        for feature in FEATEURES:
            self.detect_anomalies(feature,newline,anomaly_type)