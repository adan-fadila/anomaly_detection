import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def generate_synthetic_data(start_time, hours=24):
    # Initialize variables based on observed patterns
    current_time = start_time
    temperature = 21.1  # Initial temperature from the dataset
    humidity = 73.9  # Initial humidity from the dataset
    
    data = []
    end_time = start_time + timedelta(hours=hours)
    
    while current_time <= end_time:
        # Increment time by approximately 1 minute
        current_time += timedelta(minutes=random.randint(1, 2))
        
        # Simulate temperature variation with seasonality (daily cycle)
        hour_of_day = current_time.hour + current_time.minute / 60.0
        temp_variation = 10 * np.sin((hour_of_day / 24.0) * 2 * np.pi)  # Daily sinusoidal variation
        temperature = 50 + temp_variation + random.uniform(-5, 5)
        temperature = max(20, min(80, temperature))  # Keep values within reasonable bounds
        
        # Simulate inverse humidity variation
        humidity = 100 - (temperature * 0.8) + random.uniform(-3, 3)
        humidity = max(15, min(100, humidity))  # Keep within realistic range
        
        data.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), round(temperature, 1), round(humidity, 1)])
    
    return pd.DataFrame(data, columns=["timestamp", "temperature", "humidity"])

# Generate 24 hours of data from the last timestamp of the given dataset
start_time = datetime.strptime("2025-03-28 17:39:03", "%Y-%m-%d %H:%M:%S")
generated_df = generate_synthetic_data(start_time, hours=24)

# Save to CSV
generated_df.to_csv("synthetic_data.csv", index=False)

# Plot temperature and humidity trends
plt.figure(figsize=(12, 5))

# Temperature plot
plt.subplot(2, 1, 1)
plt.plot(generated_df["timestamp"], generated_df["temperature"], marker='o', linestyle='-', label="Temperature (°C)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Trend")
plt.legend()
plt.xticks(rotation=45)

# Humidity plot
plt.subplot(2, 1, 2)
plt.plot(generated_df["timestamp"], generated_df["humidity"], marker='s', linestyle='-', color='r', label="Humidity (%)")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.title("Humidity Trend")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Display the first few rows
print(generated_df.head())
