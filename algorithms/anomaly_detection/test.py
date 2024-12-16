import unittest
from VAE_algorithm import VAEAlgorithm
import pandas as pd
import numpy as np

class TestDetectAnomalies(unittest.TestCase):
    def setUp(self):
        """Set up synthetic data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(loc=50, scale=5, size=(500, 1))
        anomalous_data = np.random.normal(loc=70, scale=2, size=(10, 1))

        # Label the data: 0 for normal, 1 for anomalies
        normal_labels = np.zeros((500, 1))
        anomaly_labels = np.ones((10, 1))

        # Combine data and labels
        self.data = pd.DataFrame(np.vstack([normal_data, anomalous_data]), columns=['meantemp'])
        self.data['label'] = np.vstack([normal_labels, anomaly_labels])

        np.random.shuffle(self.data.values)
        self.vae_algo = VAEAlgorithm()

    def test_detect_and_print_anomalies(self):
        """Test anomaly detection and print anomalies from a CSV file."""
        # Specify the CSV file name (change this name as needed)
        csv_file_name = "synthetic_meantemp_data.csv"

        # Save synthetic data to CSV file for testing
        self.data[['meantemp']].to_csv(csv_file_name, index=False)

        # Load data from CSV file
        df = pd.read_csv(csv_file_name)

        # Train the model
        self.vae_algo.train_model(self.vae_algo.process_data(df))

        # Detect anomalies
        anomalies = self.vae_algo.detect_anomalies(df, df)

        # Print detected anomalies
        print(f"Detected Anomalies in '{csv_file_name}':")
        print(anomalies)

        # Assertions to ensure anomalies are detected
        self.assertTrue(len(anomalies) > 0, "No anomalies detected.")
        self.assertTrue('meantemp' in anomalies.columns, "Anomalies should contain 'meantemp' column.")

if __name__ == "__main__":
    unittest.main()
