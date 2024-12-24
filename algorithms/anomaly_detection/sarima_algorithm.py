from core.base_anomaly import AnomalyDetectionAlgorithm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")


class SARIMAAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.pdq = (1, 1, 1)  # SARIMA model order
        self.seasonal_order = (2, 1, 1, 4)  # Seasonal order for SARIMA
        self.threshold = 3  # Anomaly detection threshold (in standard deviations)
        self.window_size = 100  # Size of the sliding window
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=logging.INFO
        )
        self.logger.info(f"{self.__class__.__name__} class instantiated")

    def detect_anomalies(self, df, dataset):
        """
        Detect anomalies using SARIMA for time-series data.

        Parameters:
            df (pd.DataFrame): Current data with new readings.
            dataset (pd.DataFrame): Historical dataset for comparison.

        Returns:
            pd.DataFrame: DataFrame containing detected anomalies.
        """
        self.logger.info(f"{self.__class__.__name__} - detect_anomalies method invoked")

        # Log initial columns
        self.logger.info(f"Initial df columns: {df.columns.tolist()}")
        self.logger.info(f"Initial dataset columns: {dataset.columns.tolist()}")

        # Ensure 'timestamp' exists
        for name, frame in [("df", df), ("dataset", dataset)]:
            if 'timestamp' not in frame.columns:
                raise ValueError(f"The '{name}' DataFrame must contain a 'timestamp' column.")

        # Align columns and validate
        common_columns = list(set(df.columns).intersection(dataset.columns))
        if len(common_columns) <= 1:  # Only 'timestamp' or less in common
            raise ValueError("The input DataFrames must share at least one data column besides 'timestamp'.")

        self.logger.info(f"Common columns: {common_columns}")

        df = df[common_columns]
        dataset = dataset[common_columns]

        # Combine datasets and sort by timestamp
        combined_data = pd.concat([dataset, df], ignore_index=True)
        combined_data.sort_values(by='timestamp', inplace=True)
        combined_data.reset_index(drop=True, inplace=True)

        self.logger.info(f"Combined dataset columns: {combined_data.columns.tolist()}")

        anomaly_details = []

        # Process each column except 'timestamp'
        for column in [col for col in combined_data.columns if col != 'timestamp']:
            self.logger.info(f"Processing column: {column}")

            # Drop missing values
            column_data = combined_data[['timestamp', column]].dropna()
            if column_data.empty:
                self.logger.warning(f"Column '{column}' has no data after dropping NaNs. Skipping.")
                continue

            # Ensure column has enough data
            if len(column_data) < self.window_size:
                self.logger.warning(f"Not enough data in column '{column}' for the specified window size. Skipping.")
                continue

            # Split historical and current data
            historical_data = column_data.iloc[:-len(df)][column] if len(df) > 0 else column_data[column]
            current_data = column_data.iloc[-len(df):][column]
            timestamps = column_data.iloc[-len(df):]['timestamp']

            try:
                # Fit SARIMA model on historical data
                model = SARIMAX(historical_data, order=self.pdq, seasonal_order=self.seasonal_order)
                model_fit = model.fit(disp=False)
                self.logger.info(f"SARIMA model successfully fitted for column '{column}'.")

                # Forecast values for current data
                forecast_values = model_fit.forecast(steps=len(current_data))

                # Calculate residuals and identify anomalies
                residuals = current_data.values - forecast_values
                std_dev = np.std(historical_data)

                for idx, (actual_value, forecasted_value, timestamp) in enumerate(zip(current_data.values, forecast_values, timestamps)):
                    residual = actual_value - forecasted_value
                    if abs(residual) > self.threshold * std_dev:
                        anomaly_details.append({
                            'timestamp': timestamp,
                            'column': column,
                            'value': actual_value,
                            'residual': residual,
                            'algorithm': 'SARIMA',
                        })

            except Exception as e:
                self.logger.error(f"Error processing column '{column}': {e}")
                continue

        # Return anomalies as a DataFrame
        anomalies_df = pd.DataFrame(anomaly_details)
        self.logger.info(f"Anomalies detected: {len(anomalies_df)}")
        return anomalies_df
