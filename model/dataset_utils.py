
import pandas as pd

class Data_Set_Manager:
    def __init__(self):

        try:
    # Attempt to read the CSV file
            self.data = pd.read_csv(r"dataset\DailyDelhiClimateTrain.csv").copy()
        except FileNotFoundError as e:
            # Handle the case where the file is not found
            print(f"Error: The file was not found. Details: {e}")
            # Optionally, provide a fallback behavior, e.g., loading a default file or returning a message
        except pd.errors.EmptyDataError as e:
            # Handle the case where the file is empty
            print(f"Error: The file is empty. Details: {e}")
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred: {e}")
        pass
    
    def process_dataset(self):
        """
        Process the dataset by reading the relevant CSV file and preparing the dataframe.
        """
        # Assuming the dataset is in the 'dataset' folder and the file is 'DailyDelhiClimateTrain.csv'
        

        # Selecting only relevant columns
        meantemp_data = self.data[['date', 'meantemp']]
        meantemp_data = meantemp_data.copy()
        meantemp_data['date'] = pd.to_datetime(meantemp_data['date'])
        meantemp_data['days'] = meantemp_data['date'] - meantemp_data['date'].min()
        
        return meantemp_data
