
from algorithms.anomaly_detect_algorithm import AnomalyDetectionAlgorithm
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler



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
        # DBSCAN initialization
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        anomaly_indices = []
        
        # Combine the datasets
        combined_data = pd.concat([dataset, df])
        data = self.process_data(combined_data)
        
        # Get the last window of data
        start = len(data) - self.window_size  # Start from the last window
        if start < 0:  # If the data is smaller than window size, process from the start
            start = 0

        window = data[start:start + self.window_size]
        
        window_array = np.array(window, dtype=np.float64).reshape(-1, 1)
        
        clusters = dbscan.fit_predict(window_array)
        
        anomalies_in_window = np.where(clusters == -1)[0] + start
        
        anomaly_indices.extend(anomalies_in_window)
        
        print("Anomalies indices in last window:", anomaly_indices)
        
        new_data_start_idx = len(dataset)
        new_data_end_idx = len(combined_data)
        anomalies_in_new_data = [
            idx for idx in anomaly_indices 
            if new_data_start_idx <= idx < new_data_end_idx
        ]
        print("Anomalies in new data:", anomalies_in_new_data)
        
        # Create the anomalies DataFrame
        anomalies_df = pd.DataFrame(anomalies_in_new_data, columns=['index'])
        
        # Reset the index of df before accessing meantemp (ensure alignment)
        df_reset = combined_data.reset_index(drop=True)
        
        # Add the 'meantemp' column (or any other relevant column from df)
        anomalies_df['meantemp'] = df_reset.loc[anomalies_df['index'], 'meantemp'].values
        anomalies_df.reset_index(drop=True, inplace=True)
        
        return anomalies_df





    




