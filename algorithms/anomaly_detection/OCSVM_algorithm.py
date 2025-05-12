import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from scipy.stats import skew, kurtosis
import os
from core.base_anomaly import AnomalyDetectionAlgorithm
import joblib
DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OneClassSVMAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self,model_path, large_window_size, threshold,step_size,scaler_path):
        super().__init__()
        self.step_size = step_size
        self.window_size = large_window_size
        self.threshold = threshold
        if scaler_path != "none":
            scaler = os.path.join(DIR,scaler_path)
    
            self.scaler = joblib.load(scaler)
        else:
            self.scaler = None
        try:
            self.model = joblib.load(model_path)
            print("Loaded pre-trained One-Class SVM model.")
        except Exception as e:
            print(f"Error loading One-Class SVM model: {e}")
            self.model = None
    def process_data(self, dataset):
        """Extract the target feature from the dataset"""
        return dataset[self.feature]
    
    
    def extract_raw_features_from_data(self, data_scaled):
        """Extract raw features (windows of data) for anomaly detection."""
        features = []
        indices = []
        
        for i in range(0, len(data_scaled) - self.window_size + 1, self.step_size):
            window = data_scaled[i:i + self.window_size]
            feat = [                              # raw window
                np.mean(window.astype(float)),
                np.std(window.astype(float)),
                skew(window.astype(float)),
                kurtosis(window.astype(float))
            ]
            features.append(feat)
            indices.append(i)
            flat_features = []
            for f in features:
                flat_f = []
                for val in f:
                    if isinstance(val, np.ndarray):
                        flat_f.append(val.item())  # Assumes 1-element arrays
                    else:
                        flat_f.append(val)
                flat_features.append(flat_f)
        return np.array(flat_features), indices


    
    def detect_anomalies(self, df,feature):
        """
        Detect collective anomalies in the data using One-Class SVM.
        
        Args:
            df (int): The number of records to analyze from the end of the dataset
            dataset (pd.DataFrame or np.ndarray): The complete dataset
            
        Returns:
            list: List of dictionaries containing anomaly intervals with 'start' and 'end' keys
        """
        dataset = df
        try:
            # Check if dataset is a NumPy array or Pandas DataFrame and process accordingly
            if isinstance(dataset, pd.DataFrame):
               
                # If it's a DataFrame, extract the feature column (assuming first column if not specified)
                data = dataset[feature].values.reshape(-1, 1)
            else:
                # If it's a NumPy array, directly reshape it
                data = dataset.values.reshape(-1, 1)
            if self.scaler is not None:
                
                data_scaled = self.scaler.transform(data)
            else:
                data_scaled = data
            
            target_data = data_scaled
            
            features = []
          
            features,ind = self.extract_raw_features_from_data(target_data)

            if len(features) == 0:
                print("No features extracted. Data length may be less than window size.")
                return []
            
            if self.model is None:
                print("Error: No model loaded.")
                return []
            
            features_array = np.array(features)
            last_window_features = features_array[-1]

            last_window_features_2d = last_window_features.reshape(1, -1)

            decision_scores = self.model.decision_function(last_window_features_2d)
            
            
            # Collective anomaly detection
            for i, score in enumerate(decision_scores):
                start_idx = ind[-1]
                end_idx = min(start_idx +self.window_size, len(df))
                if end_idx == len(df):
                    start_idx = len(df) - self.window_size
                
                print(f"window score: {score}")
                
                
            
         
            
            return score <= self.threshold
        
        except Exception as e:
            print(f"Error detecting collective anomalies: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_raw_features_from_data(self, data):
        """
        Extract raw features (windows of data) for One-Class SVM.
        
        Args:
            data (pd.DataFrame): DataFrame with 'value_scaled' column
            
        Returns:
            tuple: (features array, indices array)
        """
        features = []
        indices = []
        
        for i in range(0, len(data) - self.window_size + 1, self.step_size):
            window = data['value_scaled'].iloc[i:i + self.window_size].values
            
            # Skip windows with NaN values
            if not np.isnan(window).any():
                features.append(window)
                indices.append(i)
        
        if len(features) > 0:
            return np.array(features), indices
        else:
            return np.array([]), []