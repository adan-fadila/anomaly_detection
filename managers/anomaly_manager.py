# from typing import List
# import pandas as pd
# from algorithms.anomaly_detection.stl_algorithm import STLAlgorithm
# from algorithms.anomaly_detection.arima_algorithm import ARIMAAlgorithm
# from algorithms.anomaly_detection.sarima_algorithm import SARIMAAlgorithm
# # from algorithms.anomaly_detection.collective_detecting.dbscan_coll_algorithm import DBSCANAlgorithm
# from algorithms.anomaly_detection.LSTM_algorithm import LSTMAlgorithm
# from algorithms.anomaly_detection.OCSVM_algorithm import OneClassSVMAlgorithm
# import matplotlib.pyplot as plt
# import io
# from config.constant import POINTWISE,COLLECTIVE_algorithms, COLLECTIVE,SEASONALITY,TREND,algorithms,SEASONALITY_algorithms,TREND_algorithms,COLLECTIVE_WINDOW_SIZE,COLLECTIVE_STEP_SIZE
# import os
# from config.constant import POINTWISE_LSTM_model,POINTWISE_LSTM_SEQL
# from config.constant import COLLECTIVE_LSTM_model
# from config.constant import COLLECTIVE_WINDOW_SIZE

# from matplotlib.patches import Rectangle
# import json


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CONFIG_FILE = os.path.join(BASE_DIR,'config','config.json')

# POINTWISE_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights_P1.weights.h5')
# COLLECTIVE_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights_COL.weights.h5')
# TREND_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_anomaly_trend.weights.h5')

# SEASONALITY_OCSVM_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_season_scaler_model.pkl')
# TREND_OCSVM_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_trend_scaler_model.pkl')

# SEASONALITY_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_seasonal_model.pkl')
# TREND_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_trend_model.pkl')
# COLLECTIVE_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_model.joblib')
# def load_algorithm_config(type):
#     """
#     Load the algorithm configuration from a JSON file.
#     Returns a list of algorithms to use or defaults to ["stl"].
#     """
#     try:
#         if type == POINTWISE:
#             list_= algorithms
#             return list_
#         if type == SEASONALITY:
#             list_= SEASONALITY_algorithms
#             return list_ 
#         if type == TREND:
#             list_= TREND_algorithms
#             return list_ 
#         if type == COLLECTIVE:
#             list_= COLLECTIVE_algorithms
#             return list_ 
#     except FileNotFoundError:
#         return ValueError(f"File Not Found Error : {e}")
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Invalid JSON in configuration file: {e}")
    



# class AnomalyDetectionManager:
#     def __init__(self):
#         # Map algorithm names to actual classes
#         self.algorithms_map = {
            
#             # 'stl': STLAlgorithm(),
#            'arima': ARIMAAlgorithm(),
#            'sarima': SARIMAAlgorithm(),
#             'LSTM': LSTMAlgorithm(
#                  anomaly_type='pointwise',
#                  seq_length=POINTWISE_LSTM_SEQL,
#                  threshold_factor=0.5,
#                  window_size=10,
#                  step_size=1,
#                  model = POINTWISE_LSTM_model,
#                  model_path=POINTWISE_LSTM_WEIGHTS_FILE),

           
#                 'OCSVM_Col':  OneClassSVMAlgorithm(
#                  model_path=COLLECTIVE_OCSVM_WEIGHTS_FILE,
#                  large_window_size=COLLECTIVE_WINDOW_SIZE,
#                  threshold=0,
#                  step_size=COLLECTIVE_STEP_SIZE,
#                  scaler_path=TREND_OCSVM_SCALAR_FILE,
#                  ),
#             'LSTM_Coll': LSTMAlgorithm(
#                  anomaly_type='collective',
#                  seq_length=20,    
#                  threshold_factor=0.5,
#                  window_size=20,
#                  step_size=18,
#                  model = COLLECTIVE_LSTM_model,
#                  model_path=COLLECTIVE_LSTM_WEIGHTS_FILE),
#          }
#         self.window_size = 100
        
#     def load_feature_model_config(feature: str, anomaly_type: str, algorithm_name: str):
#         try:
#             with open(CONFIG_FILE, 'r') as f:
#                 config = json.load(f)
#             return config["features"][feature][anomaly_type][algorithm_name]
#         except KeyError:
#             raise ValueError(f"Missing configuration for feature '{feature}', anomaly_type '{anomaly_type}', algorithm '{algorithm_name}'")

#     def _load_selected_algorithms(self, algorithms):
        
#         return [self.algorithms_map[algo] for algo in algorithms if algo in self.algorithms_map]


#     def _window_data(self, df: pd.DataFrame, dataset: pd.DataFrame): 
#         combined_dataset = pd.concat([dataset, df], ignore_index=True)
#         combined_dataset['date'] = pd.to_datetime(combined_dataset['date'])
#         combined_dataset.set_index('date', inplace=True)
#         start_idx = max(0, len(combined_dataset) - self.window_size)
#         window_data = combined_dataset.iloc[start_idx:]
#         return window_data, start_idx
    
#     def plot_dataframe_with_anomalies(self, df, feature_col):
#         mid_idx = COLLECTIVE_WINDOW_SIZE 
#         df = df[feature_col]
#         fig, ax = plt.subplots(figsize=(10, 6))

#         ax.plot(df, label=feature_col, color='blue')  
        
#         ax.plot(df.iloc[-mid_idx:], label='Anomaly', color='red', linewidth=2)

#         ax.set_xlabel('Index')  
#         ax.set_ylabel(feature_col)
#         ax.set_title(f'Plot of {feature_col} with Anomalies Highlighted')

#         ax.legend()

#         plt.savefig("coll_anomaly_detection.png", dpi=300, bbox_inches="tight")
        
#         buf = io.BytesIO()
#         try:
#             plt.savefig(buf, format='png')
#             buf.seek(0)  
#             print("Plot saved successfully to buffer")
#         except Exception as e:
#             print(f"Error saving plot: {e}")

#         plt.close()

       
#         return buf

    

#     def _plot_anomalies(self, df: pd.DataFrame, combined_anomalies: pd.DataFrame,feature):
#         """
#         Combine the given DataFrame (df) with the dataset, plot the data with anomalies marked in red,
#         and return the plot as an image.
#         """
        
#         print(f"df Anomalies: {df}")
#         anomalies_in_window = combined_anomalies
#         anomalies_in_window['timestamp'] = pd.to_datetime(anomalies_in_window['timestamp'])
#         print(f"Anomalies in window: {anomalies_in_window}")

#         fig, ax = plt.subplots(figsize=(12, 6))
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
       

#         df.set_index('timestamp', inplace=True)
#         ax.plot(df.index, df[feature], label='Temperature', color='blue', alpha=0.7)
        
#         if not anomalies_in_window.empty:
#             ax.scatter(
#                 anomalies_in_window['timestamp'], 
#                 anomalies_in_window['anomaly_val'], 
#                 color='red', 
#                 label='Anomalies (New Data)', 
#                 s=50, 
#                 zorder=5
#             )
        
#         ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
#         plt.xticks(rotation=45)

#         # Add plot details
#         ax.set_title("Anomaly Detection")
#         ax.set_xlabel("time")
#         ax.set_ylabel("Temperature")
#         ax.legend()
#         ax.grid(True)

#         buf = io.BytesIO()
#         try:
#             plt.savefig("p_detection.png", dpi=300, bbox_inches="tight")
#             plt.savefig(buf, format='png')
#             buf.seek(0)  
#             print("Plot saved successfully to buffer")
#         except Exception as e:
#             print(f"Error saving plot: {e}")

#         plt.close(fig)

#         return buf
    
#     def detect_pointwise_anomalies(self, df: pd.DataFrame,selected_algorithms,feature):
#         combined_anomalies = pd.DataFrame(columns=['anomaly_score', 'anomaly_val', 'date', 'voting_algorithms'])
#         combined_anomalies.set_index('date', inplace=True)

#         for algorithm in selected_algorithms:
#             anomalies = algorithm.detect_anomalies(df, feature)
#             print(f"Anomalies detected by {algorithm.name}: {anomalies}")
#             if anomalies is not None and not anomalies.size == 0:
#                 anomalies.set_index('timestamp', inplace=True)

#                 for date, row in anomalies.iterrows():
#                     if date in combined_anomalies.index:
#                         combined_anomalies.loc[date, 'anomaly_score'] += 1
#                         combined_anomalies.loc[date, 'voting_algorithms'] += f",{algorithm.name}"
#                     else:
#                         combined_anomalies.loc[date] = {
#                             'anomaly_score': 1,
#                             'anomaly_val': row[feature],
#                             'voting_algorithms': algorithm.name,
#                         }

#         combined_anomalies['voting_algorithms'] = combined_anomalies['voting_algorithms'].apply(
#             lambda x: ','.join(sorted(set(x.split(','))))
#         )

#         combined_anomalies.reset_index(inplace=True)
#         combined_anomalies.rename(columns={'date': 'timestamp'}, inplace=True)

#         plot = self._plot_anomalies(df, combined_anomalies,feature=feature)
#         print(f"Combined Anomalies: {combined_anomalies}")
#         return combined_anomalies,plot


#     def detect_coll_anomalies(self, len_df,dataset,selected_algorithms,df,anomaly_type,feature):

#         vote = 0
#         voted_algorithms = []
#         for algorithm in selected_algorithms:
#             if algorithm.detect_anomalies(dataset,feature):
#                 vote += 1
#                 voted_algorithms.append(algorithm.name)

#         if vote > 0:
#             plot = self.plot_dataframe_with_anomalies(dataset,feature_col=feature)
           
#             result = {'start': dataset.index[0], 'end': dataset.index[-1],'algorithm': voted_algorithms}
#             print(result)
#             return result,plot
#         else:
#             print("No anomalies detected.")
#             return [],None
        
   

         
#     def detect_anomalies(self, df: pd.DataFrame,dataset,anomaly_type: str,feature):

#         print(f"Anomaly Type: {anomaly_type}")
#         print(f"Feature: {feature}")
        
#         len_df = len(df)
#         print(f"df: {df}")
#         print(f"dataset: {dataset}")
#         algorithms = load_algorithm_config(anomaly_type)
#         selected_algorithms = self._load_selected_algorithms(algorithms)
#         if(anomaly_type == POINTWISE):
#             return self.detect_pointwise_anomalies(df,selected_algorithms,feature=feature)
#         # if(anomaly_type == COLLECTIVE):   
#         #     return self.detect_coll_anomalies(dataset=df, len_df=len_df, selected_algorithms=selected_algorithms,df=df,anomaly_type=anomaly_type,feature=feature)
            
from typing import List
import pandas as pd
from algorithms.anomaly_detection.stl_algorithm import STLAlgorithm
from algorithms.anomaly_detection.arima_algorithm import ARIMAAlgorithm
from algorithms.anomaly_detection.sarima_algorithm import SARIMAAlgorithm
from algorithms.anomaly_detection.LSTM_algorithm import LSTMAlgorithm
from algorithms.anomaly_detection.OCSVM_algorithm import OneClassSVMAlgorithm
import matplotlib.pyplot as plt
import io
from config.constant import POINTWISE, COLLECTIVE, SEASONALITY, TREND, COLLECTIVE_WINDOW_SIZE, COLLECTIVE_STEP_SIZE
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'config.json')

def load_algorithm_config(feature, anomaly_type):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    try:
        return config["features"][feature][anomaly_type]
    except KeyError as e:
        raise ValueError(f"Error loading algorithm config: {e}")

class AnomalyDetectionManager:
    def __init__(self):
        self.window_size = 100

    def _load_selected_algorithms(self, feature, anomaly_type):
        algorithms_config = load_algorithm_config(feature, anomaly_type)
        selected = []

        for algo_name, params in algorithms_config.items():
           
            if algo_name == "LSTM":
                selected.append(
                    LSTMAlgorithm(
                        feature=feature,
                        anomaly_type=anomaly_type,
                        **params
                    )
                )
           
            elif algo_name == "OCSVM":
                selected.append(
                    OneClassSVMAlgorithm(
                        **params
                    )
                )
            elif algo_name == "arima":
                selected.append(ARIMAAlgorithm())
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")

        return selected


    def _window_data(self, df: pd.DataFrame, dataset: pd.DataFrame): 
        combined_dataset = pd.concat([dataset, df], ignore_index=True)
        combined_dataset['date'] = pd.to_datetime(combined_dataset['date'])
        combined_dataset.set_index('date', inplace=True)
        start_idx = max(0, len(combined_dataset) - self.window_size)
        window_data = combined_dataset.iloc[start_idx:]
        return window_data, start_idx

    def plot_dataframe_with_anomalies(self, df, feature_col):
        mid_idx = COLLECTIVE_WINDOW_SIZE 
        df = df[feature_col]
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df, label=feature_col, color='blue')  
        ax.plot(df.iloc[-mid_idx:], label='Anomaly', color='red', linewidth=2)

        ax.set_xlabel('Index')  
        ax.set_ylabel(feature_col)
        ax.set_title(f'Plot of {feature_col} with Anomalies Highlighted')

        ax.legend()

        plt.savefig("coll_anomaly_detection.png", dpi=300, bbox_inches="tight")

        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png')
            buf.seek(0)  
            print("Plot saved successfully to buffer")
        except Exception as e:
            print(f"Error saving plot: {e}")

        plt.close()

        return buf

    def _plot_anomalies(self, df: pd.DataFrame, combined_anomalies: pd.DataFrame, feature):
        anomalies_in_window = combined_anomalies
        anomalies_in_window['timestamp'] = pd.to_datetime(anomalies_in_window['timestamp'])

        fig, ax = plt.subplots(figsize=(12, 6))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        ax.plot(df.index, df[feature], label=feature, color='blue', alpha=0.7)

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

        ax.set_title("Anomaly Detection")
        ax.set_xlabel("time")
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        try:
            plt.savefig("p_detection.png", dpi=300, bbox_inches="tight")
            plt.savefig(buf, format='png')
            buf.seek(0)  
            print("Plot saved successfully to buffer")
        except Exception as e:
            print(f"Error saving plot: {e}")

        plt.close(fig)

        return buf

    def detect_pointwise_anomalies(self, df: pd.DataFrame, selected_algorithms, feature):
        combined_anomalies = pd.DataFrame(columns=['anomaly_score', 'anomaly_val', 'date', 'voting_algorithms'])
        combined_anomalies.set_index('date', inplace=True)

        for algorithm in selected_algorithms:
            anomalies = algorithm.detect_anomalies(df, feature)
            print(f"Anomalies detected by {algorithm.name}: {anomalies}")
            if anomalies is not None and not anomalies.size == 0:
                anomalies.set_index('timestamp', inplace=True)

                for date, row in anomalies.iterrows():
                    if date in combined_anomalies.index:
                        combined_anomalies.loc[date, 'anomaly_score'] += 1
                        combined_anomalies.loc[date, 'voting_algorithms'] += f",{algorithm.name}"
                    else:
                        combined_anomalies.loc[date] = {
                            'anomaly_score': 1,
                            'anomaly_val': row[feature],
                            'voting_algorithms': algorithm.name,
                        }

        combined_anomalies['voting_algorithms'] = combined_anomalies['voting_algorithms'].apply(
            lambda x: ','.join(sorted(set(x.split(','))))
        )

        combined_anomalies.reset_index(inplace=True)
        combined_anomalies.rename(columns={'date': 'timestamp'}, inplace=True)

        plot = self._plot_anomalies(df, combined_anomalies, feature=feature)
        print(f"Combined Anomalies: {combined_anomalies}")
        return combined_anomalies, plot

    def detect_coll_anomalies(self, len_df, dataset, selected_algorithms, df, anomaly_type, feature):
        vote = 0
        voted_algorithms = []
        for algorithm in selected_algorithms:
            if algorithm.detect_anomalies(df,feature):
                vote += 1
                voted_algorithms.append(algorithm.name)

        if vote > 0:
            plot = self.plot_dataframe_with_anomalies(dataset, feature_col=feature)
            result = {'start': dataset.index[len(dataset) // 2], 'end': dataset.index[-1], 'voting_algorithms': voted_algorithms}
            print(result)
            return result, plot
        else:
            print("No anomalies detected.")
            return [], None

    def detect_anomalies(self, df: pd.DataFrame, dataset, anomaly_type: str, feature: str):
        print(f"Anomaly Type: {anomaly_type}")
        print(f"Feature: {feature}")
        len_df = len(df)
        selected_algorithms = self._load_selected_algorithms(feature, anomaly_type)

        if anomaly_type == POINTWISE:
            return self.detect_pointwise_anomalies(df, selected_algorithms, feature=feature)
        elif anomaly_type == COLLECTIVE:
            return self.detect_coll_anomalies(
                len_df=len_df,
                dataset=dataset,
                selected_algorithms=selected_algorithms,
                df=df,
                anomaly_type=anomaly_type,
                feature=feature
            )
        else:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")
