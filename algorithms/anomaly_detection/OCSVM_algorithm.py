import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from config.constant import POINTWISE, COLLECTIVE,SEASONALITY,TREND,algorithms,SEASONALITY_algorithms
from core.base_anomaly import AnomalyDetectionAlgorithm
import joblib
class OneClassSVMAlgorithm(AnomalyDetectionAlgorithm):
    def __init__(self,  model_path, large_window_size, threshold,step_size):
        super().__init__()
        self.step_size = step_size
        self.window_size = large_window_size
        self.threshold = threshold
        self.scaler = StandardScaler()

        try:
            self.model = joblib.load(model_path)
            print("Loaded pre-trained One-Class SVM model.")
        except Exception as e:
            print(f"Error loading One-Class SVM model: {e}")
            self.model = None
    def process_data(self, dataset):
        return dataset[self.feature].values.reshape(-1, 1)
    
    def extract_raw_features_from_data(self, data_scaled):
        """Extract raw features (windows of data) for anomaly detection."""
        features = []
        indices = []
        
        for i in range(0, len(data_scaled) - self.window_size + 1, self.step_size):
            window = data_scaled[i:i + self.window_size]
            features.append(window)
            indices.append(i)
            
        return np.array(features), indices

    
    def detect_anomalies(self, df, dataset):
        """
        Detect collective anomalies in the data using One-Class SVM.
        
        Args:
            df (int): The number of records to analyze from the end of the dataset
            dataset (pd.DataFrame or np.ndarray): The complete dataset
            
        Returns:
            list: List of dictionaries containing anomaly intervals with 'start' and 'end' keys
        """
        try:
            # Check if dataset is a NumPy array or Pandas DataFrame and process accordingly
            if isinstance(dataset, pd.DataFrame):
                # If it's a DataFrame, extract the feature column (assuming first column if not specified)
                feature_col = self.feature if hasattr(self, 'feature') else dataset.columns[0]
                data = dataset[feature_col].values.reshape(-1, 1)
            else:
                # If it's a NumPy array, directly reshape it
                data = dataset.reshape(-1, 1)
            
            # Normalize the data
            data_scaled = self.scaler.fit_transform(data)
            
            # Extract the last 'df' records

            target_data = data_scaled[-df:]
            
            # Extract windows (features) for One-Class SVM
            features = []
            for i in range(0, len(target_data) - self.window_size + 1, self.step_size):  
                window = target_data[i:i + self.window_size].flatten()  # Flatten for One-Class SVM
                features.append(window)

            # If there is leftover data at the end, add the last window
            if (len(target_data) - self.window_size) % self.step_size != 0:
                last_window = target_data[-self.window_size:].flatten()
                features.append(last_window)
            
            if len(features) == 0:
                print("No features extracted. Data length may be less than window size.")
                return []
            
            # Check if model is loaded
            if self.model is None:
                print("Error: No model loaded.")
                return []
            
            # Get decision scores from the model
            features_array = np.array(features)
            decision_scores = self.model.decision_function(features_array)
            
            # Determine anomalies based on threshold
            threshold =-(np.abs( (np.mean(decision_scores) +self.threshold*np.std(decision_scores))))

            print(f"threshold: {threshold}")
            
            collective_anomalies = []  # List to store anomaly intervals
            # Collective anomaly detection
            for i, score in enumerate(decision_scores):
                start_idx = i * self.step_size
                end_idx = min(start_idx +self.window_size, df)
                if end_idx == df:
                    start_idx = df - self.window_size
                
                print(f"window score: {score}")
                
                if score <= threshold:  # One-Class SVM: lower scores indicate anomalies
                    collective_anomalies.append({"start": start_idx, "end": end_idx})
            
            print(collective_anomalies)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Plot the last 60 points or all points if less than 60
            plot_length = min(len(target_data), len(dataset))
            
            if isinstance(dataset, pd.DataFrame):
                plt.plot(range(plot_length), dataset[feature_col].values[-plot_length:], label="Dataset", color="blue")
            else:
                plt.plot(range(plot_length), dataset[-plot_length:], label="Dataset", color="blue")
            
            # Highlight anomalous regions
            for anomaly in collective_anomalies:
                start = anomaly["start"]
                end = anomaly["end"]
                if start < plot_length:
                    plt.axvspan(max(0, plot_length - df + start), min(plot_length - 1, plot_length - df + end), 
                            color='red', alpha=0.3, label='Anomaly')
            
            # Labels and title
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("One-Class SVM Anomaly Detection")
            plt.legend(handles=[
                plt.Line2D([0], [0], color='blue', label='Dataset'),
                plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3, label='Anomaly')
            ])
            
            # Save the plot as PNG
            
            
            return collective_anomalies
        
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


    
