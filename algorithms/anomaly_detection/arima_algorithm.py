from core.base_anomaly import AnomalyDetectionAlgorithm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import logging


class ARIMAAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.pdq = (1, 1, 1)
        self.threshold = 3
        self.window_size = 30
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=logging.INFO
        )
        self.logger.info(f"{self.__class__.__name__} class instantiated")

    def detect_anomalies(self, df, dataset):
        self.logger.info(f"{self.__class__.__name__} - detect_anomalies method invoked")

        column = df.columns[1]  # Dynamically detect column to process (ignores 'timestamp')
        data = dataset[column]

        if len(data) < self.window_size:
            raise ValueError("Not enough data for the specified window size.")

        anomalies = []
        historical_data = list(data.dropna())  # Exclude NaN values from historical data

        for _, row in df.iterrows():
            new_point = row[column]
            # Skip rows with missing or invalid data
            if pd.isna(new_point):
                continue

            window_data = historical_data[-self.window_size:]
            if len(window_data) < self.window_size:
                continue  # Skip if there's not enough data for the window

            try:
                model = ARIMA(window_data, order=self.pdq)
                model_fit = model.fit()
                predicted_value = model_fit.forecast(steps=1)[0]
                residual = new_point - predicted_value

                if abs(residual) > self.threshold * np.std(window_data):
                    anomalies.append(row)
            except Exception as e:
                self.logger.error(f"ARIMA model fitting failed: {e}")

            historical_data.append(new_point)

        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.reset_index(drop=True, inplace=True)
        anomalies_df['algorithm'] = self.__class__.__name__

        return anomalies_df
