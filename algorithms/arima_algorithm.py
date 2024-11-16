from algorithms.anomaly_detect_algorithm import AnomalyDetectionAlgorithm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class ARIMAAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.pdq= (1,1,1)
        self.threshold = 3
        self.windowSize = 30
        pass

    def process_data(self,dataset):
        return dataset[self.feature]
    
    def detect_anomalies(self, df, dataset):

        """
        Check if a series of new points in a DataFrame are anomalies based on a sliding window ARIMA model.
        
        Parameters:
            data (list or np.ndarray): Historical data for the sliding window.
            new_points_df (pd.DataFrame): DataFrame containing new points to check.
            window_size (int): Size of the sliding window.
            threshold (float): Residual threshold for anomaly detection (in standard deviations).
            column_name (str): The column in the DataFrame containing the new points.
        
        Returns:
            pd.DataFrame: A copy of the DataFrame with an added 'is_anomaly' column.
        """
        print(df)
        data = self.process_data(dataset)

        if len(data) < self.windowSize:
            raise ValueError("Not enough data for the specified window size.")
        
        anomalies = []
        historical_data = list(data) 

        for _, row in df.iterrows():
            new_point = row[self.feature]
            window_data = historical_data[-self.windowSize:]
            
            model = ARIMA(window_data, order=self.pdq)  
            model_fit = model.fit()
            
            predicted_value = model_fit.forecast(steps=1)[0]
            print(predicted_value)
            residual = new_point - predicted_value
            
            if abs(residual) > self.threshold * np.std(window_data):
                anomalies.append(row)
            historical_data.append(new_point)
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.reset_index(drop=True, inplace=True)
        print(anomalies_df)
        return anomalies_df
