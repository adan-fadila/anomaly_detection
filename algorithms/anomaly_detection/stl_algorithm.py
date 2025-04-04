from core.base_anomaly import AnomalyDetectionAlgorithm
from statsmodels.tsa.seasonal import STL
import pandas as pd

class STLAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.threshold_factor = 2.5
        self.window_size = 80


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
        window_resid = residuals[-self.window_size:]
        res_mean = window_resid.mean()
        res_std = window_resid.std()
        
     
        lower_bound = res_mean - self.threshold_factor * res_std
        upper_bound = res_mean + self.threshold_factor * res_std      
        new_points_residuals = residuals.iloc[-len(df):]
        new_points_residuals = new_points_residuals.reset_index(drop=True)
        anomaly_condition = (new_points_residuals > upper_bound) | (new_points_residuals < lower_bound)
        anomalies = df[anomaly_condition].copy()
        anomalies.reset_index(drop=True, inplace=True)

        return anomalies
    