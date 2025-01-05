from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

class AnomalyDetectionAlgorithm(ABC):
    @abstractmethod
    def __init__(self):
        self.feature = 'meantemp'
        self.name = self.__class__.__name__  # Automatically set the name to the class name

    @abstractmethod
    def detect_anomalies(self, df,dataset):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :param dataset:
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        pass
    @staticmethod
    def compute_features(window):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :param dataset:
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        mean_val = np.mean(window)
        std_val = np.std(window)
        skew_val = skew(window)
        kurt_val = kurtosis(window)
        return [mean_val, std_val, skew_val, kurt_val]
    @abstractmethod
    def process_data(self,dataset):
        """
        process the data and preparing for the model.

        """
        
        # Room_id,Timestamp,Temperature,AC_Status,AC_Desired_Temperature,,Light_Device_Status
