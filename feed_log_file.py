import csv
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Define the path to save the data
output_path = r"C:\Users\amin\Desktop\smart-space\anomaly_detection\data\logs\sensor_data.csv"

# Generate data
def generate_sensor_data(start_date, end_date):
    data = []
    current_time = start_date
    while current_time <= end_date:
        if 8 <= current_time.hour < 16:
            ac_temp = round(random.uniform(30, 32), 1)
            lights = "OFF"
        elif 16 <= current_time.hour < 22:
            ac_temp = round(random.uniform(23, 25), 1)
            lights = "ON"
        else:
            ac_temp = round(random.uniform(19, 21), 1)
            lights = "OFF"

        data.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M"),
            "ac_temperature": ac_temp,
            "lights": lights
        })
        current_time += timedelta(minutes=10)

    return data

# Inject hard anomaly values
def inject_anomaly_values(data):
    anomaly_days = random.sample(range(4), 2)  # Choose 2 random days for hard anomalies
    for anomaly_day in anomaly_days:
        day_data = [entry for entry in data if datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M").day == anomaly_day + 1]

        # Inject hard anomaly in AC temperature
        for entry in day_data:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M")
            if 12 <= entry_time.hour < 14:  # Inject during midday hours
                entry["ac_temperature"] = round(random.uniform(50, 60), 1)  # Hard anomaly: temperature extremely high

        # Inject hard anomaly in lights
        for entry in day_data:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M")
            if 22 <= entry_time.hour < 24:  # Inject during late night
                entry["lights"] = "ON" if entry["lights"] == "OFF" else "OFF"  # Flip lights status unexpectedly

    return data


# Add three days of specific data
def add_three_days_data(data):
    # Find the last timestamp in the current data
    last_timestamp = datetime.strptime(data[-1]["timestamp"], "%Y-%m-%d %H:%M")
    start_date = last_timestamp + timedelta(minutes=1)
    end_date = start_date + timedelta(days=3)

    # Generate new data for three days
    current_time = start_date
    while current_time <= end_date:
        if 14 <= current_time.hour < 16:
            ac_temp = round(random.uniform(18, 21), 1)
            lights = "ON"
        elif 8 <= current_time.hour < 16:
            ac_temp = round(random.uniform(30, 32), 1)
            lights = "OFF"
        elif 16 <= current_time.hour < 22:
            ac_temp = round(random.uniform(23, 25), 1)
            lights = "ON"
        else:
            ac_temp = round(random.uniform(19, 21), 1)
            lights = "OFF"

        data.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M"),
            "ac_temperature": ac_temp,
            "lights": lights
        })
        current_time += timedelta(minutes=10)

    return data

# Plot temperature
def plot_temperature(csv_file):
    timestamps = []
    temperatures = []

    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamps.append(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M"))
            temperatures.append(float(row["ac_temperature"]))

    plt.figure(figsize=(15, 5))
    plt.plot(timestamps, temperatures, label="AC Temperature")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (\u00b0C)")
    plt.title("AC Temperature Over Time")
    plt.legend()
    plt.grid()
    plt.show()

# Plot lights
def plot_lights(csv_file):
    timestamps = []
    lights_status = []

    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamps.append(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M"))
            lights_status.append(1 if row["lights"] == "ON" else 0)

    plt.figure(figsize=(15, 5))
    plt.step(timestamps, lights_status, label="Lights Status (1=ON, 0=OFF)", where="mid")
    plt.xlabel("Timestamp")
    plt.ylabel("Lights Status")
    plt.title("Lights Status Over Time")
    plt.legend()
    plt.grid()
    plt.show()

# Main script
if __name__ == "__main__":
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=4)

    # Read existing data if file exists
    try:
        with open(output_path, mode="r") as file:
            reader = csv.DictReader(file)
            sensor_data = [row for row in reader]
    except FileNotFoundError:
        sensor_data = []

    # Generate new data and append it
    new_data = generate_sensor_data(start_date, end_date)
    sensor_data.extend(new_data)

    # Inject anomalies and new habits
    sensor_data = inject_anomaly_values(sensor_data)

    # Add three days of specific data
    sensor_data = add_three_days_data(sensor_data)

    # Write to CSV
    with open(output_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "ac_temperature", "lights"])
        writer.writeheader()
        writer.writerows(sensor_data)

    print(f"Sensor data saved to {output_path}")

    # Plot data
    plot_temperature(output_path)
    plot_lights(output_path)
