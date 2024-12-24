
from core.base_anomaly import AnomalyDetectionAlgorithm
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import logging


from core.base_anomaly import AnomalyDetectionAlgorithm
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import logging

class DBSCANAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.eps = 3  # Maximum distance between samples for them to be considered in the same neighborhood
        self.min_samples = 2  # Minimum number of samples in a neighborhood to form a core point
        self.window_size = 100  # Size of the sliding window
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=logging.INFO
        )
        self.logger.info(f"{self.__class__.__name__} class instantiated")

    def detect_anomalies(self, df, dataset):
        """
        Detect anomalies for each column (except timestamp) using DBSCAN.

        Parameters:
            df (pd.DataFrame): Incoming new data with columns including `timestamp`.
            dataset (pd.DataFrame): Historical data with columns including `timestamp`.

        Returns:
            pd.DataFrame: Anomalies detected, including timestamp, column, and value.
        """
        self.logger.info(f"{self.__class__.__name__} - detect_anomalies method invoked")

        # Ensure 'timestamp' column exists
        if 'timestamp' not in df.columns:
            raise ValueError("The dataset must contain a 'timestamp' column.")

        anomaly_details = []

        # Combine historical and new data for processing
        combined_data = pd.concat([dataset, df])
        combined_data = combined_data.reset_index(drop=True)

        # Process each column except 'timestamp'
        for column in combined_data.columns:
            if column == 'timestamp':
                continue

            self.logger.info(f"Processing column: {column}")

            # Extract relevant data and drop NaN values
            data = combined_data[column].dropna()

            if len(data) < self.window_size:
                self.logger.warning(f"Not enough data in column {column} for the specified window size.")
                continue

            # Get the last window of data
            start = max(len(data) - self.window_size, 0)
            window = data.iloc[start:]

            # Reshape data for DBSCAN
            window_array = np.array(window, dtype=np.float64).reshape(-1, 1)

            # Apply DBSCAN
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            clusters = dbscan.fit_predict(window_array)

            # Find anomaly indices (-1 indicates anomalies in DBSCAN)
            anomalies_in_window = np.where(clusters == -1)[0] + start

            # Filter anomalies that are in the new data
            new_data_start_idx = len(dataset)
            anomalies_in_new_data = [
                idx for idx in anomalies_in_window if new_data_start_idx <= idx < len(combined_data)
            ]

            # Add anomaly details to the result
            for idx in anomalies_in_new_data:
                anomaly_details.append({
                    'timestamp': combined_data.loc[idx, 'timestamp'],
                    'column': column,
                    'value': combined_data.loc[idx, column],
                    'algorithm': self.__class__.__name__,
                })

        # Return anomalies as a DataFrame
        anomalies_df = pd.DataFrame(anomaly_details)
        anomalies_df['algorithm'] = self.__class__.__name__

        return anomalies_df
