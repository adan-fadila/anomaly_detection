# from core.base_anomaly import AnomalyDetectionAlgorithm
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense,Dropout
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# import os
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore")
# import pandas as pd
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# WEIGHTS_FILE = os.path.join(BASE_DIR,'anomaly_detection' ,'models_weights', 'lstm_anomaly_seasonality.weights.h5')
# class LSTMAlgorithm(AnomalyDetectionAlgorithm):
#     def __init__(self,model,threshold_factor,seq_length,weight_path,step_size):
#         super().__init__()
#         self.step_size = step_size
#         self.seq_length = seq_length
#         self.threshold_factor = threshold_factor
#         self.window_size = seq_length
#         self.scaler = MinMaxScaler()
#         self.model = model
#         try:
#             self.model.load_weights(weight_path)
#         except Exception as e:
#             print(f"Error loading weights: {e}")
#         pass

#     def process_data(self,dataset):
#         return dataset[self.feature]
    
#     def detect_anomalies(self, df, dataset):
#         dataset = self.process_data(dataset)
#         not_processed_df = df
#         df = self.process_data(df)
#         reshaped_df = df
#         print(f"********************LSTM********************")

#         try:
#             if reshaped_df.ndim == 1:  # If df is a 1D Series
#                 reshaped_df = df.values.reshape(-1, 1)
#             df_scaled = self.scaler.fit_transform(reshaped_df).flatten()  # Assuming you have a saved scaler
#         except Exception as e:
#             print(f"Error scaling data: {e}")
#             return None

#         try:
#             window_size = self.window_size  # Assuming you saved the window size during training
            
           

#             # Combine dataset and df for predictions
#             combined_data = np.concatenate((dataset, df_scaled))  # Combine the last `window_size` of dataset with df
            
#             # Initialize the first window from the combined dataset (last `window_size` points)
#             current_window = combined_data[-window_size:].reshape(1, window_size, 1)

#             # List to store predictions
#             predictions = []

#             # Sliding window approach for prediction
#             for i in range(len(df_scaled) - window_size + 1):
#                 # Predict the next value using the current window
#                 pred = self.model.predict(current_window, verbose=0)

#                 # Append the predicted value
#                 predictions.append(pred[0, 0])

#                 # Slide the window by one step: remove the oldest point and add the predicted value
#                 if i + window_size < len(df_scaled):
#                     # The new window will include the previous prediction and exclude the oldest value from the current window
#                     current_window = np.roll(current_window, shift=-1, axis=1)
#                     current_window[0, -1, 0] = df_scaled[i + window_size]  # Insert the next data point into the window

#             # True values after the window_size index
#             y_true = df_scaled[window_size -1:]

#             # Ensure predictions are the same length as y_true
#             predictions = np.array(predictions)  # Convert to numpy array

#             # Compute errors (absolute differences between actual and predicted values)
#             if len(predictions) == len(y_true):
#                 errors = np.abs(y_true - predictions)
#             else:
#                 print(f"Shape mismatch between predictions and y_true: {len(predictions)} vs {len(y_true)}")
#                 return None

#             print(f"********************errors: {errors}********************")

#             mean_error = np.mean(errors)
#             std_error = np.std(errors)
#             threshold = mean_error + self.threshold_factor * std_error  # Adjust multiplier if needed
#             print(f"********************threshold: {threshold}********************")

#             # Identify anomalies based on errors exceeding the threshold
#             anomalies = errors > threshold  # Boolean mask for anomalies
#             print(f"Anomalies identified: {np.sum(anomalies)} anomalies found.")
            
#             print(f"Shape of errors before thresholding: {errors.shape}")
#             print(f"Shape of anomalies after thresholding: {anomalies.shape}")

#             # Collect the indices of anomalies
#             anomaly_indices = np.where(anomalies)[0] + window_size -1  # Adjust for offset due to window size

#             anomalies_list = []  # List to store anomalies
#             for idx in anomaly_indices:
#                 if idx > 0 and idx < len(df):  # Ensure index is valid
#                     anomalies_list.append({
                        
#                         'date': not_processed_df.iloc[idx]['date'],  # Date column from df
#                         self.feature: not_processed_df.iloc[idx][self.feature],  # Feature value from df
#                     })
#                     print("idx:",not_processed_df.iloc[idx]['date'])
#                 else:
#                     print(f"Skipping invalid index {idx} for anomaly in DataFrame.")

            
#             anomalies = anomalies_list  # Update anomalies with valid list

#             print(f"********************anomalies: {anomalies}********************")

#             # Create a DataFrame of anomalies
#             anomalies_df = pd.DataFrame(anomalies)
#             anomalies_df.reset_index(drop=True, inplace=True)
#             print(f"Anomalies DataFrame: {anomalies_df}")
#         except Exception as e:
#             print(f"Error detecting anomalies: {e}")
#             anomalies_df = pd.DataFrame()


#         return anomalies_df
    
from core.base_anomaly import AnomalyDetectionAlgorithm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.seasonal import STL

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

class LSTMAlgorithm(AnomalyDetectionAlgorithm):
    """
    A unified LSTM-based anomaly detection algorithm that can handle different types of anomalies:
    - Pointwise: Detects individual data points that deviate from expected patterns
    - Seasonal: Detects anomalies in seasonal patterns
    - Trend: Detects anomalies in overall data trends
    
    This implementation uses pre-trained models for each anomaly type.
    """
    
    def __init__(self, 
                 feature,
                 anomaly_type='pointwise',
                 seq_length=40,
                 threshold_factor=1.0,
                 window_size=None,
                 step_size=1,
                 model = None,
                 model_path=None):
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
        
        self.feature = feature
        self.anomaly_type = anomaly_type
        self.seq_length = seq_length
        self.threshold_factor = threshold_factor
        self.step_size = step_size
        self.model = model
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
  
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
        if isinstance(dataset, pd.DataFrame):
            return dataset[self.feature]
        return dataset
    
    def create_sequences(self, data, seq_length):
        """Convert time series data to sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def detect_pointwise_anomalies(self, df, dataset, data_scaled):
        """Detect pointwise anomalies in the data"""
        print(f"Detecting pointwise anomalies...")
        not_processed_df = df
        
        try:
            # Create sequences from dataset for context
            dataset_seq_len = min(len(dataset), self.seq_length)
            context_data = data_scaled[:dataset_seq_len]
            
            # Combine dataset context with data for predictions
            combined_data = np.concatenate((context_data, data_scaled))
            
            # Create window for prediction
            current_window = combined_data[:self.seq_length].reshape(1, self.seq_length, 1)
            
            # List to store predictions
            predictions = []
            
            # Sliding window approach for prediction
            for i in range(len(data_scaled) - self.seq_length + 1):
                # Predict the next value using the current window
                pred = self.model.predict(current_window, verbose=0)
                
                # Append the predicted value
                predictions.append(pred[0, 0])
                
                # Slide the window by one step
                if i + self.seq_length < len(data_scaled):
                    current_window = np.roll(current_window, shift=-1, axis=1)
                    current_window[0, -1, 0] = data_scaled[i + self.seq_length]
            
            # True values after the window_size index
            y_true = data_scaled[self.seq_length - 1:]
            
            # Ensure predictions are the same length as y_true
            predictions = np.array(predictions)
            
            # Compute errors (absolute differences between actual and predicted values)
            if len(predictions) == len(y_true):
                errors = np.abs(y_true - predictions)
            else:
                print(f"Shape mismatch between predictions and y_true: {len(predictions)} vs {len(y_true)}")
                return pd.DataFrame(columns=['date', self.feature])
            
            # Calculate threshold
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            threshold = mean_error + self.threshold_factor * std_error
            print(f"Threshold: {threshold}")
            
            # Identify anomalies
            anomalies = errors > threshold
            print(f"Anomalies identified: {np.sum(anomalies)} anomalies found.")
            
            # Get indices of anomalies
            anomaly_indices = np.where(anomalies)[0] + self.seq_length - 1
            
            anomalies_list = []
            for idx in anomaly_indices:
                if 0 <= idx < len(not_processed_df):
                    anomalies_list.append({
                        'date': not_processed_df.iloc[idx]['date'],
                        self.feature: not_processed_df.iloc[idx][self.feature],
                    })
            
            # Create DataFrame of anomalies
            anomalies_df = pd.DataFrame(anomalies_list)
            if not anomalies_list:
                anomalies_df = pd.DataFrame(columns=['date', self.feature])
            
        except Exception as e:
            print(f"Error detecting pointwise anomalies: {e}")
            anomalies_df = pd.DataFrame(columns=['date', self.feature])
        
        return anomalies_df
    
    def detect_collective_anomalies(self, df, dataset, is_seasonal=True):
        """Detect collective anomalies (seasonal) in the data"""
        anomaly_type = "seasonal" if is_seasonal else "trend"
        print(f"Detecting {anomaly_type} anomalies...")

        anomaly_indices = []  # Initialize anomaly_indices to an empty list to avoid uninitialized error

        try:
            # Check if df is a NumPy array or Pandas DataFrame and process accordingly
            if isinstance(df, pd.DataFrame):
                data = dataset[self.feature].values.reshape(-1, 1)  # Extract feature column
            else:
                data = dataset.reshape(-1, 1)  # If it's a NumPy array, directly reshape it

            # Normalize the data
            data_scaled = self.scaler.fit_transform(data)
            
            # Convert data to sequences
            def create_sequences(data, seq_length):
                X = []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                return np.array(X)

            # Create sequences for LSTM model
            X_input = create_sequences(data_scaled[-df-self.seq_length:], self.seq_length)
          
            # Predict using the trained LSTM model
            predicted = self.model.predict(X_input)
     
            # Calculate reconstruction error (absolute differences between actual and predicted values)
            reconstruction_error = np.abs(predicted - (data_scaled[-len(predicted):].reshape(-1, 1)))
            threshold = np.mean(reconstruction_error)+self.threshold_factor*np.std(reconstruction_error)
            print(f"threshold: {threshold}")

            collective_anomalies = []  # List to store anomaly intervals
            
            # Collective anomaly detection
            for i in range(0, len(reconstruction_error), self.step_size):
                window_error = np.mean(reconstruction_error[i:i + self.window_size])
                print("window_error:", window_error)
                
                if window_error >= threshold:
                    collective_anomalies.append({"start": i, "end": min(i + self.window_size, df)})


            # Plot dataset
            plt.figure(figsize=(12, 6))
            plt.plot(range(df), dataset[-df:], label="Dataset", color="blue")

            # Highlight anomaly windows
            for anomaly in collective_anomalies:
                plt.axvspan(anomaly["start"], anomaly["end"], color="red", alpha=0.3, label="Anomaly" if anomaly == collective_anomalies[0] else "")

            # Labels and title
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("Dataset Plot with Anomalies Highlighted")
            plt.legend()

            # Save the plot as PNG
            plt.savefig("dataset_plot.png", dpi=300, bbox_inches="tight")
            plt.show()
           
            print(collective_anomalies)
            return collective_anomalies
           
        except Exception as e:
            print(f"Error detecting collective anomalies: {e}")
            raise  

                
    def detect_anomalies(self, df, dataset):
        """
        Main method to detect anomalies based on the specified anomaly type.
        
        Args:
            df (pd.DataFrame): The data to analyze for anomalies
            dataset (pd.DataFrame): Historical data for context (no training)
            
        Returns:
            pd.DataFrame: DataFrame containing detected anomalies
        """
        # Process data
        # processed_dataset = self.process_data(dataset)
        # processed_df = self.process_data(df)
        
        # Reshape data if needed
        # if processed_df.ndim == 1:
        #     reshaped_df = processed_df.values.reshape(-1, 1)
        # else:
        #     reshaped_df = processed_df
            
        # if processed_dataset.ndim == 1:
        #     reshaped_dataset = processed_dataset.values.reshape(-1, 1)
        # else:
        #     reshaped_dataset = processed_dataset
        
        # Scale the data
        try:
     
            if self.anomaly_type == 'pointwise':
                self.scaler.fit(dataset)
                data_scaled = self.scaler.transform(dataset).flatten()
                dataset_scaled = self.scaler.transform(dataset).flatten()
                return self.detect_pointwise_anomalies(df, dataset_scaled, data_scaled)
            elif self.anomaly_type == 'seasonal':
                return self.detect_collective_anomalies(df=df, dataset=dataset, is_seasonal=True)
            elif self.anomaly_type == 'trend':
                return self.detect_collective_anomalies(df, dataset,  is_seasonal=False)
            else:
                print(f"Unsupported anomaly type: {self.anomaly_type}")
                return pd.DataFrame(columns=['date', self.feature])
                
        except Exception as e:
            print(f"Error in detect_anomalies: {e}")
            return pd.DataFrame(columns=['date', self.feature])

    

