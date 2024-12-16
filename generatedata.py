import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_meantemp_data(days=365, anomaly_count=10, seed=42):
    """
    Generate a synthetic meantemp data series with anomalies.

    Args:
        days (int): Total number of data points (days).
        anomaly_count (int): Number of anomalies to inject.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with a 'meantemp' column and an 'anomaly' indicator.
    """
    np.random.seed(seed)
    
    # Generate smooth mean temperature data
    base_temp = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, days))  # Seasonal variation
    noise = np.random.normal(0, 1, days)  # Add Gaussian noise
    meantemp = base_temp + noise
    
    # Inject anomalies
    anomaly_indices = np.random.choice(days, anomaly_count, replace=False)
    anomalies = np.random.uniform(low=-15, high=15, size=anomaly_count)
    meantemp[anomaly_indices] += anomalies
    
    # Create DataFrame
    data = pd.DataFrame({
        'day': np.arange(1, days + 1),
        'meantemp': meantemp,
        'anomaly': 0
    })
    data.loc[anomaly_indices, 'anomaly'] = 1  # Mark anomalies
    
    return data

# Generate the data
data = generate_meantemp_data()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data['day'], data['meantemp'], label='Mean Temperature')
plt.scatter(data[data['anomaly'] == 1]['day'], data[data['anomaly'] == 1]['meantemp'],
            color='red', label='Anomalies', zorder=5)
plt.title('Synthetic Mean Temperature Data with Anomalies')
plt.xlabel('Day')
plt.ylabel('Mean Temperature (°C)')
plt.legend()
plt.grid()
plt.show()

# Save to CSV for testing
data.to_csv('synthetic_meantemp_data.csv', index=False)
print(data.head(10))  # Display first 10 rows
