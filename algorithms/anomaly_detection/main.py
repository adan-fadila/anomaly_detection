# main.py
from VAE_algorithm import VAEAlgorithm
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
import pandas as pd

class AnomalyDetectionAlgorithm(ABC):
    @abstractmethod
    def detect_anomalies(self, df,dataset):
        """
        Detect anomalies in the given dataframe.
        
        :param df: DataFrame containing the time series data
        :return: DataFrame with anomalies detected (e.g., with anomaly scores)
        """
        pass


def generate_synthetic_data():
    """Generate synthetic data with normal and anomalous points."""
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=5, size=(500, 1))
    anomalous_data = np.random.normal(loc=70, scale=2, size=(10, 1))
    data = np.vstack([normal_data, anomalous_data])
    np.random.shuffle(data)
    return pd.DataFrame(data, columns=['meantemp'])

if __name__ == "__main__":
    # Instantiate the algorithm
    vae_algo = VAEAlgorithm()

    # Generate data
    df = generate_synthetic_data()

    # Train the model
    print("Training the model...")
    vae_algo.train_model(vae_algo.process_data(df))

    # Detect anomalies
    print("Detecting anomalies...")
    anomalies = vae_algo.detect_anomalies(df, df)

    # Output anomalies
    print("Detected Anomalies:")
    print(anomalies)
