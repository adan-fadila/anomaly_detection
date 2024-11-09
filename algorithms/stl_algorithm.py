from algorithms.anomaly_detect_algorithm import AnomalyDetectionAlgorithm
from statsmodels.tsa.seasonal import STL
import pandas as pd

class STLAlgorithm(AnomalyDetectionAlgorithm):
    def detect_anomalies(self, df,dataset):
        dataset = pd.concat([dataset, df], ignore_index=True)
        dataset.sort_values(by='date', inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        stl = STL(dataset['meantemp'], period=365)
        result = stl.fit()
        residuals = result.resid
        threshold = residuals.std() * 4
        new_points_residuals = residuals.iloc[-len(df):]
        new_points_residuals = new_points_residuals.reset_index(drop=True)
        anomalies = df[new_points_residuals.abs() > threshold].copy()
        anomalies.reset_index(drop=True, inplace=True)
        print(f"anomalies: {anomalies}")
        return anomalies
