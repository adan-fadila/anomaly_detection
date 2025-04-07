from core.base_anomaly import AnomalyDetectionAlgorithm
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
        self.threshold_factor = 4
        self.window_size = 10
        pass

    def process_data(self,dataset,feature='temperature'):
        return dataset[feature]
    
    def detect_anomalies(self, df,feature):

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
        data = self.process_data(df,feature=feature)
        print(f"ARIMA:Data: {data}")
        if len(data) < self.window_size:
            raise ValueError("Not enough data for the specified window size.")
        
        anomalies = []
        # df = data.reset_index(drop=True)  # ensure index is clean and ordered
        print(f"ARIMA:DataFrame: {df}")
        for i in range(self.window_size, len(df)):
            window_data = df[feature].iloc[i - self.window_size:i].tolist()
            new_point = df[feature].iloc[i]
            timestamp = df['timestamp'].iloc[i]

            try:
                model = ARIMA(window_data, order=self.pdq)
                model_fit = model.fit()
                predicted_value = model_fit.forecast(steps=1)[0]
            except Exception as e:
                print(f"ARIMA fitting error at index {i}: {e}")
                continue

            residual = new_point - predicted_value

            if abs(residual) > self.threshold_factor * np.std(window_data):
                anomalies.append({'timestamp': timestamp, feature: new_point})

        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.reset_index(drop=True, inplace=True)
        print(f"ARIMA: Anomalies DataFrame: {anomalies_df}")
        return anomalies_df