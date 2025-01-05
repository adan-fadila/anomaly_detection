from typing import List
import pandas as pd
from algorithms.anomaly_detection.stl_algorithm import STLAlgorithm
from algorithms.anomaly_detection.arima_algorithm import ARIMAAlgorithm
from algorithms.anomaly_detection.sarima_algorithm import SARIMAAlgorithm
from algorithms.anomaly_detection.dbscan_coll_algorithm import DBSCANAlgorithm
from algorithms.anomaly_detection.LSTM_algorithm import LSTMAlgorithm
import matplotlib.pyplot as plt
import io
from config.constant import POINTWISE, COLLECTIVE
import os
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR,'config','config.json')

def load_algorithm_config(config_file=CONFIG_FILE,type=POINTWISE):
    """
    Load the algorithm configuration from a JSON file.
    Returns a list of algorithms to use or defaults to ["stl"].
    """
    try:
        if type == POINTWISE:
            with open(config_file, 'r') as f:
                config = json.load(f)
                list_= config.get("algorithms")
                return list_
        if type == COLLECTIVE:
            with open(config_file, 'r') as f:
                config = json.load(f)
                list_= config.get("collective_algorithms")
                return list_ 
    except FileNotFoundError:
        return ValueError(f"File Not Found Error : {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    



class AnomalyDetectionManager:
    def __init__(self):
        # Map algorithm names to actual classes
        self.algorithms_map = {
            'stl': STLAlgorithm(),
            'arima': ARIMAAlgorithm(),
            'sarima': SARIMAAlgorithm(),
            'dbscan': DBSCANAlgorithm(),
            'LSTM': LSTMAlgorithm()
        }
        self.window_size = 100
        

    def _load_selected_algorithms(self, algorithms: List[str]):
        
        return [self.algorithms_map[algo] for algo in algorithms if algo in self.algorithms_map]


    def _window_data(self, df: pd.DataFrame, dataset: pd.DataFrame): 
        combined_dataset = pd.concat([dataset, df], ignore_index=True)
        combined_dataset['date'] = pd.to_datetime(combined_dataset['date'])
        combined_dataset.set_index('date', inplace=True)
        start_idx = max(0, len(combined_dataset) - self.window_size)
        window_data = combined_dataset.iloc[start_idx:]
        return window_data, start_idx
    

    def _plot_anomalies(self, df: pd.DataFrame, combined_anomalies: pd.DataFrame, dataset: pd.DataFrame):
        """
        Combine the given DataFrame (df) with the dataset, plot the data with anomalies marked in red,
        and return the plot as an image.
        """
        
        window_data,start_idx =  self._window_data(df, dataset)

        anomalies_in_window = combined_anomalies[combined_anomalies['is_anomaly']]
        anomalies_in_window['date'] = pd.to_datetime(anomalies_in_window['date'])

        anomalies_in_window_filtered = anomalies_in_window[anomalies_in_window['date'].isin(window_data.index)]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(window_data.index, window_data['meantemp'], label='Temperature', color='blue', alpha=0.7)
        
        if not anomalies_in_window_filtered.empty:
            ax.scatter(
                anomalies_in_window_filtered['date'], 
                anomalies_in_window_filtered['anomaly_val'], 
                color='red', 
                label='Anomalies (New Data)', 
                s=50, 
                zorder=5
            )
        
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add plot details
        ax.set_title("Anomaly Detection (Last 100 Data Points)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf



    def _plot_coll_anomalies(self, df: pd.DataFrame, combined_anomalies: pd.DataFrame, dataset: pd.DataFrame):
        """
        Combine the given DataFrame (df) with the dataset, plot the data with anomalies marked in red,
        and return the plot as an image.
        """
        
        window_data,start_idx =  self._window_data(df, dataset)
        plt.figure(figsize=(10, 6))
        plt.plot(window_data.index, window_data['meantemp'], label="Data", color='blue')

        # Highlight the collective anomalies
        for _, anomaly in combined_anomalies.iterrows():
            start_index = anomaly['Start_Index']
            end_index = anomaly['End_Index']
           
            # Ensure the indexes are within the data range
            start_date = window_data.index[start_index - start_idx-1]
            end_date = window_data.index[end_index - start_idx-1]
            
            # Highlight the anomaly region in red
            plt.axvspan(start_date, end_date, color='red', alpha=0.3)
        # Add labels and title
        plt.title("Time Series Data with Collective Anomalies")
        plt.xlabel("Date")
        plt.ylabel("meantemp")
        plt.legend()
        # Save the plot to the specified path
        plt.savefig("collective_anomalies.png")
        print(f"Plot saved to collective_anomalies.png")
        
        # Close the plot to free memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf



    def detect_pointwise_anomalies(self, df: pd.DataFrame,dataset,selected_algorithms):
        combined_anomalies = pd.DataFrame(columns=['anomaly_score', 'anomaly_val', 'date', 'voting_algorithms', 'is_anomaly'])
        combined_anomalies.set_index('date', inplace=True)

        for algorithm in selected_algorithms:
            anomalies = algorithm.detect_anomalies(df, dataset)

            if anomalies is not None and not anomalies.empty:
                anomalies.set_index('date', inplace=True)

                for date, row in anomalies.iterrows():
                    if date in combined_anomalies.index:
                        # Update existing entry
                        combined_anomalies.loc[date, 'anomaly_score'] += 1
                        combined_anomalies.loc[date, 'voting_algorithms'] += f",{algorithm.name}"
                    else:
                        # Add new entry
                        combined_anomalies.loc[date] = {
                            'anomaly_score': 1,
                            'anomaly_val': row['meantemp'],
                            'voting_algorithms': algorithm.name,
                            'is_anomaly': True,
                        }

        # Deduplicate voting_algorithms
        combined_anomalies['voting_algorithms'] = combined_anomalies['voting_algorithms'].apply(
            lambda x: ','.join(sorted(set(x.split(','))))
        )

        combined_anomalies.reset_index(inplace=True)
        plot = self._plot_anomalies(df, combined_anomalies, dataset)
        return combined_anomalies,plot



    def detect_collective_anomalies(self, df: pd.DataFrame,dataset,selected_algorithms):
        DBSCANA = DBSCANAlgorithm()
        anomaly = DBSCANA.detect_anomalies(df,dataset)
        plot =self._plot_coll_anomalies(df, anomaly, dataset)
        return anomaly, plot
    


    def detect_anomalies(self, df: pd.DataFrame,dataset,anomaly_type: str):

        algorithms = load_algorithm_config(CONFIG_FILE,anomaly_type)
        selected_algorithms = self._load_selected_algorithms(algorithms)
        if(anomaly_type == POINTWISE):

            return self.detect_pointwise_anomalies(df,dataset,selected_algorithms)
        if(anomaly_type == COLLECTIVE):
            
            return self.detect_collective_anomalies(df,dataset,selected_algorithms)
        
        