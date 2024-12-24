import pandas as pd
import logging


class Data_Set_Manager:
    def __init__(self, file_path):
        """
        Initialize the DataSetManager with a specified file path.
        """
        self.file_path = file_path
        self.data = None
        self._load_data()

    def _load_data(self):
        """
        Attempt to load the dataset from the file path.
        """
        try:
            logging.info(f"Loading dataset from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logging.info("Dataset successfully loaded")
        except FileNotFoundError as e:
            logging.error(f"Error: The file was not found. Details: {e}")
            self.data = None
        except pd.errors.EmptyDataError as e:
            logging.error(f"Error: The file is empty. Details: {e}")
            self.data = None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the dataset: {e}")
            self.data = None

    def process_dataset(self):
        """
        Process the dataset to ensure it meets the requirements for anomaly detection.

        Returns:
            pd.DataFrame: Processed dataset.
        """
        try:
            if self.data is None:
                raise ValueError("Dataset is not loaded. Please check the file path or contents.")

            # Ensure the dataset has a 'timestamp' column
            if 'timestamp' not in self.data.columns:
                raise ValueError("The dataset must contain a 'timestamp' column.")

            # Drop rows with missing timestamp
            self.data.dropna(subset=['timestamp'], inplace=True)

            # Convert the 'timestamp' column to datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], errors='coerce')

            # Drop rows with invalid timestamps
            self.data.dropna(subset=['timestamp'], inplace=True)

            # Ensure all other columns are processed as numeric where possible
            for column in self.data.columns:
                if column != 'timestamp':
                    self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

            logging.info("Dataset successfully processed")
            return self.data

        except ValueError as ve:
            logging.error(f"ValueError during dataset processing: {ve}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during dataset processing: {e}")
            return None
