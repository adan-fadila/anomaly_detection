from typing import List
import pandas as pd
from algorithms.anomaly_detection.stl_algorithm import STLAlgorithm
from algorithms.anomaly_detection.arima_algorithm import ARIMAAlgorithm
from algorithms.anomaly_detection.sarima_algorithm import SARIMAAlgorithm
from algorithms.anomaly_detection.dbscan_algorithm import DBSCANAlgorithm

class AnomalyDetectionManager:
    def __init__(self, algorithms: List[str]):
        # Map algorithm names to actual classes
        self.algorithms_map = {
            'stl': STLAlgorithm(),
            'arima': ARIMAAlgorithm(),
            # 'sarima': SARIMAAlgorithm(),
            # 'dbscan': DBSCANAlgorithm(),
        }
        self.selected_algorithms = self._load_selected_algorithms(algorithms)

    def _load_selected_algorithms(self, algorithms: List[str]):
        return [self.algorithms_map[algo] for algo in algorithms if algo in self.algorithms_map]

    def detect_anomalies(self, df: pd.DataFrame):
        # Ensure 'timestamp' column is present
        if 'timestamp' not in df.columns:
            raise ValueError("The dataset must contain a 'timestamp' column.")

        # Dictionary to aggregate anomalies
        anomaly_map = {}

        # Loop through each column (excluding 'timestamp')
        for column in df.columns:
            if column == 'timestamp':
                continue

            for algorithm in self.selected_algorithms:
                anomalies = algorithm.detect_anomalies(df[['timestamp', column]], df[['timestamp', column]])
                if not anomalies.empty:
                    for _, row in anomalies.iterrows():
                        key = (row['timestamp'], column, row[column])  # Unique identifier for each anomaly
                        if key not in anomaly_map:
                            anomaly_map[key] = {
                                'timestamp': row['timestamp'],
                                'column': column,
                                'value': row[column],
                                'algorithms': [algorithm.__class__.__name__],  # Add algorithm name
                            }
                        else:
                            anomaly_map[key]['algorithms'].append(algorithm.__class__.__name__)

        # Combine results into a DataFrame
        anomaly_details = list(anomaly_map.values())
        anomaly_details_df = pd.DataFrame(anomaly_details)
        return anomaly_details_df
