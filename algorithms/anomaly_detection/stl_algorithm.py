from core.base_anomaly import AnomalyDetectionAlgorithm
from statsmodels.tsa.seasonal import STL
import pandas as pd

class STLAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.std = 4

    def process_data(self, dataset):
        return dataset[self.feature]
    
    
    def detect_anomalies(self, df,dataset):
        dataset = pd.concat([dataset, df], ignore_index=True)
        dataset.sort_values(by='date', inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        data = self.process_data(dataset)
        stl = STL(data, period=365)
        result = stl.fit()
        residuals = result.resid
        threshold = residuals.std() * self.std
        new_points_residuals = residuals.iloc[-len(df):]
        new_points_residuals = new_points_residuals.reset_index(drop=True)
        anomalies = df[new_points_residuals.abs() > threshold].copy()
        anomalies['date'] = df.loc[new_points_residuals.abs() > threshold, 'date']

        anomalies.reset_index(drop=True, inplace=True)
        print(anomalies)
        return anomalies
    