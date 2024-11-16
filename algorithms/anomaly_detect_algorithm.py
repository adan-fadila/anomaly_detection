from abc import ABC, abstractmethod
import pandas as pd

class AnomalyDetectionAlgorithm(ABC):
    @abstractmethod
    def __init__(self):
        self.feature = 'meantemp'
    @abstractmethod
    def detect_anomalies(self, df,dataset):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :param dataset:
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        pass
    @abstractmethod
    def process_data(self,dataset):
        """
        process the data and preparing for the model.

        """