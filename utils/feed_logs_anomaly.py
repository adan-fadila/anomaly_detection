import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='data_generation.log',
                    filemode='a')
logger = logging.getLogger()

# Constants
BASE_DIR = os.getcwd()
CSV_FILE_PATH = os.path.join(BASE_DIR,"..","data", "logs", "sensor_data_anomaly.csv")
CSV_FILE_PATH_2 = os.path.join(BASE_DIR, "..","data", "csv", "DailyDelhiClimateTrain.csv")
print(CSV_FILE_PATH)
# Function to generate mean temperature anomalies
def generate_anomalies(mean_temp, num_anomalies=1):
    print(f"mean_temp: {mean_temp}")    
    anomalies = mean_temp + np.random.choice([-15, 15], num_anomalies)
    print(f"anomalies: {anomalies}")
    return anomalies


def train_model(df):
    logger.info("Training model on the dataset.")

    # Feature engineering
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month

    # Define features and target
    X = df[['day_of_year', 'month']]
    y = df['meantemp']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    logger.info("Model training completed.")
    return model

def generate_predictions_season(model, start_date, days, seasonal_means):
    logger.info("Generating future predictions.")

    new_dates = [start_date + timedelta(days=i) for i in range(1, days + 1)]
    new_data = []

    for new_date in new_dates:
        day_of_year = new_date.timetuple().tm_yday
        month = new_date.month

        # Wrap the input in a DataFrame to include feature names
        features = pd.DataFrame({'day_of_year': [day_of_year], 'month': [month]})
        mean_temp = model.predict(features)[0]

        new_data.append({
            'date': new_date,
            'meantemp': mean_temp,
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0.5, 20),
            'meanpressure': np.random.uniform(995, 1025)
        })

    # Introduce anomalies in the temperature
    collective_anomaly_indices = np.random.choice(len(new_data)-40 -50, 1, replace=False)
    print(f"collective_anomaly_indices: {collective_anomaly_indices}")
    collective_mean_temp = new_data[collective_anomaly_indices[0]]['meantemp']
    for i in range(40):
        new_data[collective_anomaly_indices[0]+50+i]['meantemp'] = collective_mean_temp - 10
        print(collective_anomaly_indices[0]+50+i)

    return pd.DataFrame(new_data)

def generate_predictions_trend(model, start_date, days, seasonal_means):
    logger.info("Generating future predictions.")

    new_dates = [start_date + timedelta(days=i) for i in range(1, days + 1)]
    new_data = []

    for new_date in new_dates:
        day_of_year = new_date.timetuple().tm_yday
        month = new_date.month

        # Wrap the input in a DataFrame to include feature names
        features = pd.DataFrame({'day_of_year': [day_of_year], 'month': [month]})
        mean_temp = model.predict(features)[0]

        new_data.append({
            'date': new_date,
            'meantemp': mean_temp,
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0.5, 20),
            'meanpressure': np.random.uniform(995, 1025)
        })

    # Introduce anomalies in the temperature
    anomaly_indices = np.random.choice(len(new_data), 3, replace=False)
    collective_anomaly_indices = np.random.choice(len(new_data)-350, 1, replace=False)
    print(f"collective_anomaly_indices: {collective_anomaly_indices}")
    collective_mean_temp = new_data[collective_anomaly_indices[0]]['meantemp']
    for i in range(300):
        new_data[collective_anomaly_indices[0]+i]['meantemp'] += 30
        print(collective_anomaly_indices[0]+i)

 
    return pd.DataFrame(new_data)



def generate_predictions(model, start_date, days, seasonal_means):
    logger.info("Generating future predictions.")

    new_dates = [start_date + timedelta(days=i) for i in range(1, days + 1)]
    new_data = []

    for new_date in new_dates:
        day_of_year = new_date.timetuple().tm_yday
        month = new_date.month

        # Wrap the input in a DataFrame to include feature names
        features = pd.DataFrame({'day_of_year': [day_of_year], 'month': [month]})
        mean_temp = model.predict(features)[0]

        new_data.append({
            'date': new_date,
            'meantemp': mean_temp,
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0.5, 20),
            'meanpressure': np.random.uniform(995, 1025)
        })

    # Introduce anomalies in the temperature
    anomaly_indices = np.random.choice(len(new_data), 4, replace=False)
    
    for index in anomaly_indices:
        new_data[index]['meantemp'] += 15
        print(index)

    return pd.DataFrame(new_data)
def main(anomaly_type):
    try:
        # Read existing data
        
        logger.info("Reading existing data from CSV.")
        df = pd.read_csv(CSV_FILE_PATH_2, parse_dates=['date'])

        # Train the model
        model = train_model(df)

        # Extract the last date
        last_date = df['date'].max()
        logger.info(f"Last date in the dataset: {last_date}")

        # Generate 30 new data points
        seasonal_means = df.groupby('month')['meantemp'].mean()
        if anomaly_type == 'season':
            new_df = generate_predictions_season(model, last_date, 200, seasonal_means)
        elif anomaly_type == 'trend':
            new_df = generate_predictions_trend(model, last_date, 400, seasonal_means)
        elif anomaly_type == 'point':
            new_df = generate_predictions(model, last_date, 30, seasonal_means)
        # Append new data to the DataFrame
        # combined_df = pd.concat([df, new_df], ignore_index=True)

        # Save back to CSV
        logger.info("Saving updated data back to CSV.")
        new_df.to_csv(CSV_FILE_PATH, index=False)
        logger.info("Data generation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main(sys.argv[1])
