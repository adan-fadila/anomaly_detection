from core.base_anomaly import AnomalyDetectionAlgorithm
import logging
from statsmodels.tsa.seasonal import STL
import pandas as pd
class STLAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=logging.INFO
        )
        self.logger.info(f"{self.__class__.__name__} class instantiated")

    def detect_anomalies(self, df, dataset):
        self.logger.info(f"{self.__class__.__name__} - detect_anomalies method invoked")
        # Process data
        column = df.columns[1]  # Assumes 'timestamp' is first
        data = pd.concat([dataset, df], ignore_index=True)
        data.sort_values(by='timestamp', inplace=True)
        data.reset_index(drop=True, inplace=True)

        # STL decomposition
        stl = STL(data[column], period=365)
        result = stl.fit()
        residuals = result.resid
        threshold = residuals.std() * 0.4
        new_points_residuals = residuals.iloc[-len(df):].reset_index(drop=True)

        # Detect anomalies
        anomalies = df[new_points_residuals.abs() > threshold].copy()
        anomalies.reset_index(drop=True, inplace=True)
        anomalies['algorithm'] = self.__class__.__name__

        print(f"Anomalies detected by {self.__class__.__name__}: {anomalies}")
        return anomalies
