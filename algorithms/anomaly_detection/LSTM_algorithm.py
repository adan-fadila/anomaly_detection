
    
from core.base_anomaly import AnomalyDetectionAlgorithm
import tensorflow as tf

from config.constant import COLLECTIVE_LSTM_model, POINTWISE_LSTM_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import joblib
DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LSTMAlgorithm(AnomalyDetectionAlgorithm):
    """
    A unified LSTM-based anomaly detection algorithm that can handle different types of anomalies:
    - Pointwise: Detects individual data points that deviate from expected patterns
    - Seasonal: Detects anomalies in seasonal patterns
    - Trend: Detects anomalies in overall data trends
    
    This implementation uses pre-trained models for each anomaly type.
    """
    
    def __init__(self, feature='temperature',
                 anomaly_type='pointwise',
                 seq_length=10,
                 threshold_factor=1.0,
                 window_size=None,
                 step_size=1,
                 model = None,
                 model_path=None,
                 scaler=None):
        """
        Initialize the LSTM anomaly detection algorithm with a pre-trained model.
        
        Args:
            feature (str): The feature column to analyze for anomalies
            anomaly_type (str): Type of anomaly to detect ('pointwise', 'seasonal', or 'trend')
            seq_length (int): Length of input sequences for LSTM
            threshold_factor (float): Factor to multiply std error for anomaly threshold
            window_size (int): Size of window for collective anomaly detection (for seasonal/trend)
            step_size (int): Step size for sliding window (for collective anomalies)
            model_path (str): Path to pre-trained model weights (required)
        """
        super().__init__()
        
        self.feature = feature  # Default feature to analyze
        self.anomaly_type = anomaly_type
        self.seq_length = seq_length
        self.threshold_factor = threshold_factor
        self.step_size = step_size
        self.window_size = window_size
        model_path = os.path.join(DIR,model_path)
        scaler = os.path.join(DIR,scaler)
        self.scaler = joblib.load(scaler)
        if model == "COLLECTIVE_LSTM_model":
            self.model = COLLECTIVE_LSTM_model
        elif model == "POINTWISE_LSTM_model":
            self.model = POINTWISE_LSTM_model
        else:
            raise ValueError(f"Unknown LSTM model architecture: {model}")
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                print(f"Model weights loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                raise ValueError(f"Could not load model weights from {model_path}: {e}")
        else:
            raise ValueError(f"Model path {model_path} does not exist or was not provided")
    
    def process_data(self, dataset):
        """Extract the target feature from the dataset"""
        return dataset[self.feature]
    
    def create_sequences(self, data, seq_length):
        """Convert time series data to sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
 
        
    def detect_pointwise_anomalies(self, df):
        try:
            df_feature = self.process_data(df)

            # Ensure proper shape for scaler
            data_scaled = self.scaler.transform(df_feature.values.reshape(-1, 1))
            print(f"data_scaled: {data_scaled}")

            if data_scaled.shape[0] < self.seq_length + 1:
                raise ValueError(f"Not enough data: need at least {self.seq_length + 1} rows.")

            input_seq = data_scaled[:self.seq_length]            # shape: (seq_len, 1)
            true_value = float(data_scaled[self.seq_length])     # scalar

            input_seq = input_seq.reshape((1, self.seq_length, 1))  # LSTM expects 3D: (batch, time, features)

            predicted_value = self.model.predict(input_seq, verbose=0)[0, 0]
            error = abs(predicted_value - true_value)
            is_anomaly = error > self.threshold_factor

            print(f"Predicted value: {predicted_value}, True value: {true_value}, Error: {error}, Is anomaly: {is_anomaly}")

            if is_anomaly:
                timestamp = df.iloc[self.seq_length].values[0] # Assuming timestamp is the index
                feature = df_feature[self.seq_length]  # Get actual value before scaling
                print(f"Anomaly detected at {timestamp}: {feature}")
                result_df = pd.DataFrame({
                    "timestamp": [timestamp],
                    self.feature: [feature]
                })
            return result_df

        except Exception as e:
            print(f"Error detecting pointwise anomalies: {e}")
            return pd.DataFrame(columns=["timestamp", "temperature"])
    
    
    def detect_collective_anomalies(self, df,feature=None):
        try:
            df_feature = self.process_data(df)
            data_scaled = self.scaler.transform(df_feature.values.reshape(-1, 1))

            n_future = 20  # How many future points to predict
            if data_scaled.shape[0] < self.seq_length + n_future:
                raise ValueError(f"Not enough data: need at least {self.seq_length + n_future} rows.")

            # Input for prediction
            input_seq = data_scaled[:self.seq_length]  # shape: (seq_len, 1)
            true_values = data_scaled[self.seq_length:self.seq_length + n_future]  # shape: (n_future, 1)

            input_seq = input_seq.reshape((1, self.seq_length, 1))  # LSTM expects (batch, time, features)

            # Predict future sequence
            predicted_values = self.model.predict(input_seq, verbose=0).reshape(-1, 1)  # shape: (n_future, 1)

            # Compute error for the sequence
            errors = np.abs(predicted_values - true_values)
            mean_error = np.mean(errors)

            # Decide anomaly
            is_anomaly = mean_error > self.threshold_factor
            print(f"Mean error: {mean_error}, Is collective anomaly: {is_anomaly}")

            
            return is_anomaly

        except Exception as e:
            print(f"Error detecting collective anomalies: {e}")
            return False

                
    def detect_anomalies(self, df, dataset):
        """
        Main method to detect anomalies based on the specified anomaly type.
        
        Args:
            df (pd.DataFrame): The data to analyze for anomalies
            dataset (pd.DataFrame): Historical data for context (no training)
            
        Returns:
            pd.DataFrame: DataFrame containing detected anomalies
        """
       
        try:
     
            if self.anomaly_type == 'pointwise':
                return self.detect_pointwise_anomalies(df)
            elif self.anomaly_type == 'collective':
                return self.detect_collective_anomalies(df=df)
   
            else:
                print(f"Unsupported anomaly type: {self.anomaly_type}")
                return pd.DataFrame(columns=['date', self.feature])
                
        except Exception as e:
            return pd.DataFrame(columns=['date', self.feature])

    

