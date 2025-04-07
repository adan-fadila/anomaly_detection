from typing import List
import pandas as pd
from algorithms.anomaly_detection.stl_algorithm import STLAlgorithm
from algorithms.anomaly_detection.arima_algorithm import ARIMAAlgorithm
from algorithms.anomaly_detection.sarima_algorithm import SARIMAAlgorithm
# from algorithms.anomaly_detection.collective_detecting.dbscan_coll_algorithm import DBSCANAlgorithm
from algorithms.anomaly_detection.LSTM_algorithm import LSTMAlgorithm
from algorithms.anomaly_detection.OCSVM_algorithm import OneClassSVMAlgorithm
import matplotlib.pyplot as plt
import io
from config.constant import POINTWISE, COLLECTIVE,SEASONALITY,TREND,algorithms,SEASONALITY_algorithms,TREND_algorithms
import os
from config.constant import POINTWISE_LSTM_model,POINTWISE_LSTM_SEQL,POINTWISE_LSTM_THRESHOLD_FACTOR,POINTWISE_LSTM_WINDOW_SIZE,TREND_WINDOW_SIZE,TREND_STEP_SIZE
from config.constant import SEASONALITY_LSTM_model,SEASONALITY_LSTM_SEQL,SEASONALITY_LSTM_WINDOW_SIZE,SEASONALITY_LSTM_STEP_SIZE,SEASONALITY_LSTM_THRESHOLD_FACTOR
from config.constant import TREND_LSTM_model,TREND_LSTM_SEQL,TREND_LSTM_WINDOW_SIZE,TREND_LSTM_STEP_SIZE,TREND_LSTM_THRESHOLD_FACTOR
from config.constant import SEASONALITY_OCSVM_WINDOW_SIZE,SEASONALITY_OCSVM_STEP_SIZE,SEASONALITY_OCSVM_THRESHOLD_FACTOR
from config.constant import TREND_OCSVM_THRESHOLD_FACTOR 
# from algorithms.anomaly_detection.seasonal_detecting.LSTM_seasonality import LSTMAlgorithmCol
from statsmodels.tsa.seasonal import STL
from matplotlib.patches import Rectangle
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR,'config','config.json')
POINTWISE_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights.weights.h5')

POINTWISE_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights.weights.h5')
SEASONALITY_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_anomaly_seasonality.weights.h5')
TREND_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_anomaly_trend.weights.h5')

SEASONALITY_OCSVM_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_season_scaler_model.pkl')
TREND_OCSVM_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_trend_scaler_model.pkl')

SEASONALITY_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_seasonal_model.pkl')
TREND_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_trend_model.pkl')
def load_algorithm_config(type):
    """
    Load the algorithm configuration from a JSON file.
    Returns a list of algorithms to use or defaults to ["stl"].
    """
    try:
        if type == POINTWISE:
            list_= algorithms
            return list_
        if type == SEASONALITY:
            list_= SEASONALITY_algorithms
            return list_ 
        if type == TREND:
            list_= TREND_algorithms
            return list_ 
    except FileNotFoundError:
        return ValueError(f"File Not Found Error : {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    



class AnomalyDetectionManager:
    def __init__(self):
        # Map algorithm names to actual classes
        self.algorithms_map = {
            
            # 'stl': STLAlgorithm(),
            'arima': ARIMAAlgorithm(),
            # 'sarima': SARIMAAlgorithm(),
            # 'LSTM': LSTMAlgorithm('meantemp',
            #      anomaly_type='pointwise',
            #      seq_length=POINTWISE_LSTM_SEQL,
            #      threshold_factor=POINTWISE_LSTM_THRESHOLD_FACTOR,
            #      window_size=POINTWISE_LSTM_WINDOW_SIZE,
            #      step_size=1,
            #      model = POINTWISE_LSTM_model,
            #      model_path=POINTWISE_LSTM_WEIGHTS_FILE),

            #  'LSTM_Seasonality': LSTMAlgorithm('meantemp',
            #      anomaly_type='seasonal',
            #      seq_length=SEASONALITY_LSTM_SEQL,
            #      threshold_factor=SEASONALITY_LSTM_THRESHOLD_FACTOR,
            #      window_size=SEASONALITY_LSTM_WINDOW_SIZE,
            #      step_size=SEASONALITY_LSTM_STEP_SIZE,
            #      model = SEASONALITY_LSTM_model,
            #      model_path=SEASONALITY_LSTM_WEIGHTS_FILE),
            #  'LSTM_Trend': LSTMAlgorithm('meantemp',
            #      anomaly_type='trend',
            #      seq_length=TREND_LSTM_SEQL,
            #      threshold_factor=TREND_LSTM_THRESHOLD_FACTOR,
            #      window_size=TREND_LSTM_WINDOW_SIZE,
            #      step_size=TREND_LSTM_WINDOW_SIZE,
            #      model = TREND_LSTM_model,
            #      model_path=TREND_LSTM_WEIGHTS_FILE),
            'OCSVM_Seasonality': OneClassSVMAlgorithm(
                 model_path=SEASONALITY_OCSVM_WEIGHTS_FILE,
                 large_window_size=SEASONALITY_OCSVM_WINDOW_SIZE,
                 threshold=SEASONALITY_OCSVM_THRESHOLD_FACTOR,
                 step_size=SEASONALITY_OCSVM_STEP_SIZE,
                 scaler_path=SEASONALITY_OCSVM_SCALAR_FILE,
                 ),
             'OCSVM_Trend':  OneClassSVMAlgorithm(
                 model_path=TREND_OCSVM_WEIGHTS_FILE,
                 large_window_size=TREND_WINDOW_SIZE,
                 threshold=TREND_OCSVM_THRESHOLD_FACTOR,
                 step_size=TREND_STEP_SIZE,
                 scaler_path=TREND_OCSVM_SCALAR_FILE,
                 ),
        }
        self.window_size = 100
        

    def _load_selected_algorithms(self, algorithms):
        
        return [self.algorithms_map[algo] for algo in algorithms if algo in self.algorithms_map]


    def _window_data(self, df: pd.DataFrame, dataset: pd.DataFrame): 
        combined_dataset = pd.concat([dataset, df], ignore_index=True)
        combined_dataset['date'] = pd.to_datetime(combined_dataset['date'])
        combined_dataset.set_index('date', inplace=True)
        start_idx = max(0, len(combined_dataset) - self.window_size)
        window_data = combined_dataset.iloc[start_idx:]
        return window_data, start_idx
    
    def plot_dataframe_with_anomalies(self, df, feature_col):
        # Ensure timestamp is a datetime object
        mid_idx = len(df) // 2  # Get the middle index of the DataFrame

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the entire DataFrame
        ax.plot(df, label='Feature', color='blue')  # Plot the entire data
        
        # Highlight the second half (anomalies) with a different color
        ax.plot(df.iloc[mid_idx:], label='Anomaly', color='red', linewidth=2)

        # We won't use date or hour format, so we will remove the formatter for simplicity
        ax.set_xlabel('Index')  # Use index as x-axis label (since 'timestamp' is not available)
        ax.set_ylabel(feature_col)  # y-axis will still display the feature name
        ax.set_title(f'Plot of {feature_col} with Anomalies Highlighted')

        # Add a legend to distinguish between feature and anomalies
        ax.legend()

        # Save the plot as PNG
        plt.savefig("coll_anomaly_detection.png", dpi=300, bbox_inches="tight")
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png')
            buf.seek(0)  # Seek to the beginning of the buffer
            print("Plot saved successfully to buffer")
        except Exception as e:
            print(f"Error saving plot: {e}")

        # Close the plot to release memory
        plt.close()

        # Anomaly dict to represent the start and end points of the second half
        # anomaly = {'start': df.index[mid_idx], 'end': df.index[-1]}

        # Return the buffer containing the plot image and the anomaly dictionary
        return buf
        # Return the buffer containing the plot image

    

    def _plot_anomalies(self, df: pd.DataFrame, combined_anomalies: pd.DataFrame,feature):
        """
        Combine the given DataFrame (df) with the dataset, plot the data with anomalies marked in red,
        and return the plot as an image.
        """
        
        print(f"df Anomalies: {df}")
        anomalies_in_window = combined_anomalies
        anomalies_in_window['timestamp'] = pd.to_datetime(anomalies_in_window['date'])
        print(f"Anomalies in window: {anomalies_in_window}")

        fig, ax = plt.subplots(figsize=(12, 6))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
       

        # Set the 'time' as the index
        df.set_index('timestamp', inplace=True)
        ax.plot(df.index, df[feature], label='Temperature', color='blue', alpha=0.7)
        
        if not anomalies_in_window.empty:
            ax.scatter(
                anomalies_in_window['timestamp'], 
                anomalies_in_window['anomaly_val'], 
                color='red', 
                label='Anomalies (New Data)', 
                s=50, 
                zorder=5
            )
        
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)

        # Add plot details
        ax.set_title("Anomaly Detection")
        ax.set_xlabel("time")
        ax.set_ylabel("Temperature")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        try:
            plt.savefig("p_detection.png", dpi=300, bbox_inches="tight")
            plt.savefig(buf, format='png')
            buf.seek(0)  # Seek to the beginning of the buffer
            print("Plot saved successfully to buffer")
        except Exception as e:
            print(f"Error saving plot: {e}")

        # Close the plot to release memory
        plt.close(fig)

        return buf
    
    def detect_pointwise_anomalies(self, df: pd.DataFrame,selected_algorithms,feature):
        combined_anomalies = pd.DataFrame(columns=['anomaly_score', 'anomaly_val', 'date', 'voting_algorithms'])
        combined_anomalies.set_index('date', inplace=True)

        for algorithm in selected_algorithms:
            anomalies = algorithm.detect_anomalies(df, feature)
            if anomalies is not None and not anomalies.size == 0:
                anomalies.set_index('timestamp', inplace=True)

                for date, row in anomalies.iterrows():
                    if date in combined_anomalies.index:
                        # Update existing entry 
                        combined_anomalies.loc[date, 'anomaly_score'] += 1
                        combined_anomalies.loc[date, 'voting_algorithms'] += f",{algorithm.name}"
                    else:
                        # Add new entry
                        combined_anomalies.loc[date] = {
                            'anomaly_score': 1,
                            'anomaly_val': row[feature],
                            'voting_algorithms': algorithm.name,
                        }

        # Deduplicate voting_algorithms
        combined_anomalies['voting_algorithms'] = combined_anomalies['voting_algorithms'].apply(
            lambda x: ','.join(sorted(set(x.split(','))))
        )

        combined_anomalies.reset_index(inplace=True)
        plot = self._plot_anomalies(df, combined_anomalies,feature=feature)
        print(combined_anomalies)
        return combined_anomalies,plot


    def detect_coll_anomalies(self, len_df,dataset,selected_algorithms,df,anomaly_type,feature):

        vote = 0
        for algorithm in selected_algorithms:
            if algorithm.detect_anomalies(len_df, dataset):
                vote += 1
        if vote > 0:
            plot = self.plot_dataframe_with_anomalies(dataset,feature_col=feature)
            # result = self.convert_anomaly_indices_to_dates(df)
            #         anomaly = {'start': df.index[mid_idx], 'end': df.index[-1]}
            result = {'start': dataset.index[0], 'end': dataset.index[-1]}
            print(result)
            return result,plot
        else:
            print("No anomalies detected.")
            return [],None
        
    
    # def convert_anomaly_indices_to_dates(self,anomalies, dataframe):
    #     """
    #     Convert numeric indices in anomaly definitions to dates from a DataFrame's index
        
    #     Parameters:
    #     anomalies (list): List of dictionaries with 'start' and 'end' keys representing anomaly ranges
    #     dataframe (pandas.DataFrame): DataFrame with a datetime index
        
    #     Returns:
    #     list: List of dictionaries with 'start' and 'end' keys containing datetime values
    #     """
    #     # Check if the dataframe has a datetime index
        
    #     # Create a new list to store the converted anomalies
    #     converted_anomalies = []
    #     for anomaly in anomalies:
    #         # Get the start and end indices
    #         start_idx = anomaly['start']
    #         end_idx = anomaly['end']
            
    #         # Ensure indices are within bounds
    #         start_idx = max(0, min(start_idx, len(dataframe) - 1))
    #         end_idx = max(0, min(end_idx, len(dataframe) - 1))
            
    #         # Convert indices to dates - access the 'date' column directly
    #         start_date = dataframe.iloc[start_idx]['date'].date()
    #         end_date = dataframe.iloc[end_idx]['date'].date()
            
    #         # Add to converted list
    #         converted_anomalies.append({
    #             'start': start_date.isoformat(),
    #             'end': end_date.isoformat()
    #         })
        
    #     return converted_anomalies
        


    # def find_overlapping_ranges(self,algorithm_results):
    #     """
    #     Find all overlapping ranges from multiple algorithm results where ranges from
    #     different algorithms overlap with each other.
        
    #     Parameters:
    #     algorithm_results (list of lists): Each inner list contains range dictionaries with 'start' and 'end' keys
        
    #     Returns:
    #     list: List of dictionaries with 'start' and 'end' keys representing overlapping ranges
    #     """
    #     overlapping_ranges = []
        
    #     # Compare ranges from the first algorithm with ranges from the second algorithm
    #     for range1 in algorithm_results[0]:
    #         for range2 in algorithm_results[1]:
    #             # Check if they overlap
    #             start_overlap = max(range1['start'], range2['start'])
    #             end_overlap = min(range1['end'], range2['end'])
                
    #             if start_overlap < end_overlap:
    #                 # There is an overlap
    #                 overlapping_ranges.append({'start': start_overlap, 'end': end_overlap})
        
    #     final_ranges = []
    #     for new_range in overlapping_ranges:
    #         is_contained = False
    #         for existing_range in final_ranges:
    #             if existing_range['start'] <= new_range['start'] and existing_range['end'] >= new_range['end']:
    #                 is_contained = True
    #                 break
    #             elif new_range['start'] <= existing_range['start'] and new_range['end'] >= existing_range['end']:
    #                 final_ranges.remove(existing_range)
            
    #         if not is_contained:
    #             final_ranges.append(new_range)

    #     return final_ranges

         
    def detect_anomalies(self, df: pd.DataFrame,dataset,anomaly_type: str,feature):
        # data = df[feature]
        print(f"Anomaly Type: {anomaly_type}")
        print(f"Feature: {feature}")
        stl = STL(df[feature], period=9,trend = 13) 
        result = stl.fit()
        len_df = len(df)
        seasonal_component = result.seasonal
        trend_component = result.trend
        algorithms = load_algorithm_config(anomaly_type)
        selected_algorithms = self._load_selected_algorithms(algorithms)
        if(anomaly_type == POINTWISE):
            return self.detect_pointwise_anomalies(df,selected_algorithms,feature=feature)
        if(anomaly_type == SEASONALITY):   
            return self.detect_coll_anomalies(dataset=seasonal_component, len_df=len_df, selected_algorithms=selected_algorithms,df=df,anomaly_type=anomaly_type,feature=feature)
        if(anomaly_type == TREND):   
            return self.detect_coll_anomalies(dataset=trend_component,len_df=len_df, selected_algorithms=selected_algorithms,df=df,anomaly_type=anomaly_type,feature=feature)           
