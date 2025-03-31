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
    
    def compute_features(self,window):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :param dataset:
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        return window
    @abstractmethod
    def process_data(self,dataset):
        """
        process the data and preparing for the model.

        """
        
        # Room_id,Timestamp,Temperature,AC_Status,AC_Desired_Temperature,,Light_Device_Status
