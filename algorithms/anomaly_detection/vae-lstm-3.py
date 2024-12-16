import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import matplotlib.pyplot as plt

def plot_csv_data(csv_file, column_name):
    """
    Plots the time-series data from a specified column in a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file.
        column_name (str): Name of the column to plot.
    """
    # Load CSV data
    try:
        data_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check if the column exists
    if column_name not in data_df.columns:
        print(f"Column '{column_name}' not found in the CSV file.")
        print(f"Available columns: {list(data_df.columns)}")
        return

    # Extract data from the specified column
    time_series_data = data_df[column_name]

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(time_series_data, label=column_name, color='blue', linewidth=1.5)
    plt.title(f"Time Series Plot of '{column_name}'", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()





# 1. Data Preprocessing
class DataPreprocessor:
    def __init__(self, window_size):
        self.window_size = window_size

    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, mean, std

    def interpolate_missing_data(self, data, max_gap):
        for i in range(len(data)):
            if np.isnan(data[i]):
                gap_start = i
                while i < len(data) and np.isnan(data[i]):
                    i += 1
                gap_end = i

                # Handle boundary cases
                if gap_start == 0 or gap_end >= len(data):
                    continue  # Skip interpolation for gaps at the start or end of the array

                if gap_end - gap_start <= max_gap:
                    data[gap_start:gap_end] = np.interp(
                        np.arange(gap_start, gap_end),
                        [gap_start - 1, gap_end],
                        [data[gap_start - 1], data[gap_end]]
                    )
        return data

    def segment_data(self, data):
        segments = []
        for i in range(len(data) - self.window_size + 1):
            segments.append(data[i:i + self.window_size])
        return np.array(segments)


# 2. Variational Autoencoder with Symmetric Construction
class VAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(32, latent_dim)
        self.log_var_layer = nn.Linear(32, latent_dim)

        # Decoder (Symmetric to Encoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var


# 3. LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        predictions = self.fc(out[:, -1, :])
        return predictions


# 4. Combined SeqVL Model
class SeqVL(nn.Module):
    def __init__(self, input_size, latent_dim, hidden_size, output_size):
        super(SeqVL, self).__init__()
        self.vae = VAE(input_size, latent_dim)
        self.lstm = LSTM(input_size, hidden_size, output_size)

    def forward(self, x):
        x_recon, mu, log_var = self.vae(x)
        predictions = self.lstm(x_recon.unsqueeze(1))  # Add time dimension
        return x_recon, mu, log_var, predictions


# Loss Functions
def vae_loss(x, x_recon, mu, log_var):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp() + 1e-6)  # Stability
    return recon_loss + kl_div


def lstm_loss(predictions, target):
    return nn.MSELoss()(predictions, target)


# Training
def train_seqvl(model, data, target, epochs, learning_rate, lambda_):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_recon, mu, log_var, predictions = model(data)

        # Match dimensions of predictions and target
        predictions = predictions.squeeze(-1)  # Remove last dimension
        target = target[:predictions.size(0)].squeeze(-1)  # Align target shape

        vae_loss_value = vae_loss(data, x_recon, mu, log_var)
        lstm_loss_value = lstm_loss(predictions, target)
        loss = vae_loss_value + lambda_ * lstm_loss_value

        if torch.isnan(loss).any():
            print("Loss contains NaN values. Exiting training.")
            break

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# Load and preprocess CSV data
def load_csv_data(csv_file, column_name, window_size):
    # Load CSV using pandas
    data_df = pd.read_csv(csv_file)
    
    # Extract the relevant column
    time_series_data = data_df[column_name].values
    
    # Handle missing values (NaNs)
    preprocessor = DataPreprocessor(window_size=window_size)
    normalized_data, mean, std = preprocessor.normalize(time_series_data)
    interpolated_data = preprocessor.interpolate_missing_data(normalized_data, max_gap=10)
    
    # Segment data into windows
    segments = preprocessor.segment_data(interpolated_data)
    
    return segments, mean, std


def detect_anomalies(data, x_recon, predictions, target, recon_threshold, pred_threshold):
    anomalies = []
    
    # Compute reconstruction errors
    reconstruction_errors = torch.mean((data - x_recon) ** 2, dim=1).detach().numpy()
    
    # Compute prediction errors
    prediction_errors = torch.abs(predictions.squeeze(-1) - target.squeeze(-1)).detach().numpy()
    
    # Loop through errors to detect anomalies
    for i in range(len(reconstruction_errors)):
        if reconstruction_errors[i] > recon_threshold or prediction_errors[i] > pred_threshold:
            anomalies.append(i)  # Store index of anomalies

    return anomalies


if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "synthetic_meantemp_data.csv"  # Replace with your actual file path
    column_name = "meantemp"       # Replace with the actual column name to plot

    # Plot the data
    plot_csv_data(csv_file, column_name)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. CSV File Path and Configuration
    csv_file = "synthetic_meantemp_data.csv"  # Replace with your actual file path
    column_name = "meantemp"       # Replace with the actual column name to plot
    window_size = 10

    # 2. Load and preprocess CSV data
    segments, mean, std = load_csv_data(csv_file, column_name, window_size)

    # Generate dummy target data for demonstration (replace with actual target if available)
    target_data = np.sin(np.linspace(0, len(segments), len(segments)))  # Example target

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(segments, dtype=torch.float32)
    target_tensor = torch.tensor(target_data[:len(segments)], dtype=torch.float32).view(-1, 1)

    # 3. Define Model and Hyperparameters
    input_size = window_size
    latent_dim = 8
    hidden_size = 16
    output_size = 1
    epochs = 50
    learning_rate = 0.001
    lambda_ = 0.1

    model = SeqVL(input_size=input_size, latent_dim=latent_dim, hidden_size=hidden_size, output_size=output_size)

    # 4. Train the Model
    train_seqvl(model, data_tensor, target_tensor, epochs=epochs, learning_rate=learning_rate, lambda_=lambda_)

    # 5. Detect Anomalies
    model.eval()
    with torch.no_grad():
        x_recon, mu, log_var, predictions = model(data_tensor)

        # Define thresholds for anomalies
        reconstruction_threshold = 0.5  # Adjust based on dataset
        prediction_threshold = 0.5  # Adjust based on dataset

        anomalies = detect_anomalies(data_tensor, x_recon, predictions, target_tensor,
                                     recon_threshold=reconstruction_threshold,
                                     pred_threshold=prediction_threshold)

        # Print only anomalous rows
        anomalous_data = pd.DataFrame(data_tensor[anomalies].numpy(), columns=[f"Feature_{i}" for i in range(window_size)])
        print("\nAnomalies Detected:")
        print(anomalous_data)
