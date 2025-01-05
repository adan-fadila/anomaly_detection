# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import pandas as pd
# import warnings
# from core.base_anomaly import AnomalyDetectionAlgorithm
# import os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# WEIGHTS_FILE = os.path.join(BASE_DIR,'anomaly_detection' ,'models_weights', 'pytorch_LSTM_model.pth')
# warnings.filterwarnings("ignore")

# class LSTMAlgorithm(AnomalyDetectionAlgorithm, nn.Module):  # Multiple inheritance
#     def __init__(self):
#         super().__init__()  # Calls the __init__ method of AnomalyDetectionAlgorithm
#         nn.Module.__init__(self)  # Explicitly calls the __init__ of nn.Module
        
#         self.threshold = 3
#         self.window_size = 70
#         self.scaler = MinMaxScaler()
        
#         input_size = 1
#         hidden_size = 32  # Reduced hidden size
#         num_layers = 2
#         output_size = 1

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # Define LSTM layers and linear layer
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(hidden_size, output_size)

#         # Load pre-trained weights
#         print("Loading pre-trained weights...")
#         try:
#             self.load_model_weights(WEIGHTS_FILE)
#             print("Pre-trained weights loaded successfully!")
#         except Exception as e:
#             print(f"Error loading pre-trained weights: {e}")

#     def load_model_weights(self, path):
#         checkpoint = torch.load(path)
#         self.load_state_dict(checkpoint)

#     def process_data(self, dataset):
#         return dataset[self.feature]

#     def detect_anomalies(self, df, dataset):
#         print(f"********************LSTM (PyTorch)********************")

#         df_scaled = self.scaler.transform(df)  # Assuming you have a saved scaler
#         window_size = self.window_size

#         # Create sliding windows
#         X_df = []
#         for i in range(len(df_scaled) - window_size):
#             X_df.append(df_scaled[i:i + window_size])
#         X_df = np.array(X_df).reshape(-1, window_size, 1)

#         # Convert to PyTorch tensors
#         X_df_tensor = torch.tensor(X_df, dtype=torch.float32)

#         # Initialize hidden and cell states
#         h0 = torch.zeros(self.num_layers, X_df_tensor.size(0), self.hidden_size)
#         c0 = torch.zeros(self.num_layers, X_df_tensor.size(0), self.hidden_size)

#         # Make predictions
#         with torch.no_grad():
#             out, _ = self.lstm(X_df_tensor, (h0, c0))
#             out = self.dropout(out)
#             predictions = self.fc(out[:, -1, :]).numpy().flatten()

#         # Compute errors
#         y_true = df_scaled[window_size:]  # True values are after the window
#         errors = np.abs(y_true - predictions)
#         mean_error = np.mean(errors)
#         std_error = np.std(errors)
#         threshold = mean_error + 4 * std_error  # Adjust multiplier if needed

#         # Identify anomalies
#         anomalies = errors > threshold
#         anomaly_indices = np.where(anomalies)[0] + window_size  # Adjust for offset due to window size
#         anomalies = []
#         for idx in anomaly_indices:
#             anomalies.append({
#                 'date': df.iloc[idx]['date'],  # Date column from df
#                 self.feature: df.iloc[idx][self.feature],  # Feature value from df
#             })

#         print(f"********************anomalies: {anomalies}********************")

#         # Create a DataFrame of anomalies
#         anomalies_df = pd.DataFrame(anomalies)
#         anomalies_df.reset_index(drop=True, inplace=True)

#         return anomalies_df

from core.base_anomaly import AnomalyDetectionAlgorithm
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FILE = os.path.join(BASE_DIR,'anomaly_detection' ,'models_weights', 'lstm_weights.weights.h5')
class LSTMAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.threshold = 3
        self.window_size = 5
        self.scaler = MinMaxScaler()
        self.model = Sequential([
            LSTM(32, activation='relu', input_shape=(self.window_size, 1), return_sequences=True),  # Smaller number of units
            Dropout(0.2),  # Prevent overfitting
            LSTM(16, activation='relu'),  # Reduce complexity with fewer units
            Dropout(0.2),
            Dense(1, activation='linear')  # Final output layer
        ])
        try:
            self.model.load_weights(WEIGHTS_FILE)
        except Exception as e:
            print(f"Error loading weights: {e}")
        pass

    def process_data(self,dataset):
        return dataset[self.feature]
    
    def detect_anomalies(self, df, dataset):
        dataset = self.process_data(dataset)
        not_processed_df = df
        df = self.process_data(df)
        reshaped_df = df
        print(f"********************LSTM********************")

        try:
            if reshaped_df.ndim == 1:  # If df is a 1D Series
                reshaped_df = df.values.reshape(-1, 1)
            df_scaled = self.scaler.fit_transform(reshaped_df).flatten()  # Assuming you have a saved scaler
        except Exception as e:
            print(f"Error scaling data: {e}")
            return None

        try:
            window_size = self.window_size  # Assuming you saved the window size during training
            
           

            # Combine dataset and df for predictions
            combined_data = np.concatenate((dataset[-window_size:], df_scaled))  # Combine the last `window_size` of dataset with df
            
            # Initialize the first window from the combined dataset (last `window_size` points)
            current_window = combined_data[-window_size:].reshape(1, window_size, 1)

            # List to store predictions
            predictions = []

            # Sliding window approach for prediction
            for i in range(len(df_scaled) - window_size):
                # Predict the next value using the current window
                pred = self.model.predict(current_window, verbose=0)

                # Append the predicted value
                predictions.append(pred[0, 0])

                # Slide the window by one step: remove the oldest point and add the predicted value
                if i + window_size < len(df_scaled):
                    # The new window will include the previous prediction and exclude the oldest value from the current window
                    current_window = np.roll(current_window, shift=-1, axis=1)
                    current_window[0, -1, 0] = df_scaled[i + window_size]  # Insert the next data point into the window

            # True values after the window_size index
            y_true = df_scaled[window_size:]

            # Ensure predictions are the same length as y_true
            predictions = np.array(predictions)  # Convert to numpy array

            # Compute errors (absolute differences between actual and predicted values)
            if len(predictions) == len(y_true):
                errors = np.abs(y_true - predictions)
            else:
                print(f"Shape mismatch between predictions and y_true: {len(predictions)} vs {len(y_true)}")
                return None

            print(f"********************errors: {errors}********************")

            mean_error = np.mean(errors)
            std_error = np.std(errors)
            threshold = mean_error + 0.75 * std_error  # Adjust multiplier if needed
            print(f"********************threshold: {threshold}********************")

            # Identify anomalies based on errors exceeding the threshold
            anomalies = errors > threshold  # Boolean mask for anomalies
            print(f"Anomalies identified: {np.sum(anomalies)} anomalies found.")
            
            print(f"Shape of errors before thresholding: {errors.shape}")
            print(f"Shape of anomalies after thresholding: {anomalies.shape}")

            # Collect the indices of anomalies
            anomaly_indices = np.where(anomalies)[0] + window_size  # Adjust for offset due to window size
            print(f"Anomaly indices: {anomaly_indices}")

            anomalies_list = []  # List to store anomalies
            for idx in anomaly_indices:
                if idx >= 0 and idx < len(df):  # Ensure index is valid
                    anomalies_list.append({
                        
                        'date': not_processed_df.iloc[idx]['date'],  # Date column from df
                        self.feature: not_processed_df.iloc[idx][self.feature],  # Feature value from df
                    })
                else:
                    print(f"Skipping invalid index {idx} for anomaly in DataFrame.")

            anomalies = anomalies_list  # Update anomalies with valid list

            print(f"********************anomalies: {anomalies}********************")

            # Create a DataFrame of anomalies
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_df.reset_index(drop=True, inplace=True)
            print(f"Anomalies DataFrame: {anomalies_df}")
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            anomalies_df = pd.DataFrame()

        return anomalies_df

        