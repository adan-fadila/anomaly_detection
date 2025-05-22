import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Remove resampling as it's not needed for plotting against the index
    # df_resampled = df.resample("5T").mean()
    
    plt.figure(figsize=(14, 6))
    
    # Plot temperature values against their index
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["temperature"], marker='o', linestyle='-', label="Temperature (°C)", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Trend")
    plt.legend()
    plt.xticks(rotation=30)
    
    # Plot humidity values against their index
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["humidity"], marker='s', linestyle='-', color='r', label="Humidity (%)", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("Humidity (%)")
    plt.title("Humidity Trend")
    plt.legend()
    plt.xticks(rotation=30)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_csv_data("sensibo-04-04-2025.csv")
