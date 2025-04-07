import pandas as pd

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
            self.data = pd.read_csv(self.file_path).copy()
        except FileNotFoundError as e:
            print(f"Error: The file was not found. Details: {e}")
        except pd.errors.EmptyDataError as e:
            print(f"Error: The file is empty. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def process_dataset(self,feature='meantemp'):
        """
        Process the dataset by selecting relevant columns and preparing it.
        """
        if self.data is None:
            print("Error: No data loaded.")
            return None

        try:
            feature = self.data[feature].copy()
            return feature
        except KeyError as e:
            print(f"Error: Missing required columns. Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")
            return None
