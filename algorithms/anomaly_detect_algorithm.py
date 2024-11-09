from abc import ABC, abstractmethod
import pandas as pd

class AnomalyDetectionAlgorithm(ABC):
    @abstractmethod
    def detect_anomalies(self, df,dataset):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        pass
