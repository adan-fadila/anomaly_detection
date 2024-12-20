from typing import List
import pandas as pd
from algorithms.stl_algorithm import STLAlgorithm
from algorithms.arima_algorithm import ARIMAAlgorithm
from algorithms.sarima_algorithm import SARIMAAlgorithm
from algorithms.dbscan_algorithm import DBSCANAlgorithm

class AnomalyDetectionManager:
    def __init__(self, algorithms: List[str]):
        # Map algorithm names to actual classes
        self.algorithms_map = {
            'stl': STLAlgorithm(),
            'arima': ARIMAAlgorithm(),
            'sarima': SARIMAAlgorithm(),
            'dbscan': DBSCANAlgorithm()
        }
        self.selected_algorithms = self._load_selected_algorithms(algorithms)

    def _load_selected_algorithms(self, algorithms: List[str]):
        
        return [self.algorithms_map[algo] for algo in algorithms if algo in self.algorithms_map]

    def detect_anomalies(self, df: pd.DataFrame,dataset):
        # Detect anomalies using the selected algorithms
        combined_anomalies = pd.DataFrame(index=df.index, columns=['anomaly_score','anomaly_val'])
        combined_anomalies['anomaly_score'] = 0

        for algorithm in self.selected_algorithms:
            anomalies = algorithm.detect_anomalies(df,dataset)
            if not anomalies.empty :
                combined_anomalies.loc[anomalies.index, 'anomaly_score'] += 1
                combined_anomalies.loc[ anomalies.index,'anomaly_val'] = anomalies.meantemp

        combined_anomalies['is_anomaly'] = combined_anomalies['anomaly_score'] > 0 
        return combined_anomalies
