
from core.base_anomaly import AnomalyDetectionAlgorithm
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from config.constant import POINTWISE, COLLECTIVE


class DBSCANAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.eps = 3
        self.min_samples =2
        self.window_size = 100
        self.step_size = 50

    def process_data(self, dataset):
        return dataset[self.feature]


    
    def detect_anomalies(self, df, dataset):
        """
        Detect anomalies in a new dataset (df) using DBSCAN clustering over sliding windows.

        Args:
            df (pd.DataFrame): New data to analyze.
            dataset (pd.DataFrame): Historical dataset.

        Returns:
            pd.DataFrame: DataFrame containing anomalies from the new dataset (df) with their indices and values.
        """
        big_window_size = 100  # Big window size for anomaly detection
        small_window_size = 10
        step_size = 10

        # Combine datasets and preprocess
        combined_data = pd.concat([dataset, df], ignore_index=True)
        data = self.process_data(combined_data)
        
        # Initialize DBSCAN
        
        dbscan = DBSCAN(eps=6, min_samples=3)
        big_window = np.array(data[-big_window_size:], dtype=np.float64)
        
        # Extract features for small windows
        small_window_features = []
        for start_small in range(0, big_window_size - small_window_size + 1, step_size):
            small_window = big_window[start_small:start_small + small_window_size]
            
            features = self.compute_features(small_window)
            
            small_window_features.append(features)
        
        small_window_features = np.array(small_window_features)
        
        clusters = dbscan.fit_predict(small_window_features)

        
        anomalies = []
        for i, cluster in enumerate(clusters):
            if cluster == -1:  
                start_small = i * step_size
                end_small = start_small + small_window_size
                start_idx = len(combined_data) - big_window_size + start_small
                end_idx = len(combined_data) - big_window_size + end_small
                anomalies.append((start_idx, end_idx))  # Append as tuple (start_idx, end_idx)


        # Filter anomalies based on the df_start_idx and df_end_idx
        df_start_idx = len(dataset)  
        df_end_idx = len(combined_data)  
        df_anomalies = [
            (max(start, df_start_idx), min(end, df_end_idx))
            for start, end in anomalies
            if start < df_end_idx and end > df_start_idx
        ]

        # Create DataFrame for anomalies
        anomaly_df = pd.DataFrame(df_anomalies, columns=["Start_Index", "End_Index"])
        print(f"Anomalies detected: {anomaly_df}")
        return anomaly_df



