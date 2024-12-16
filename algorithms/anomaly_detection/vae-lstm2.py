import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Parameters
MISSING_DURATION_THRESHOLD = 3  # Threshold for missing data duration (M)
WINDOW_SIZE = 24                # Sliding window width (w0)
SEQUENCE_LENGTH = 10            # Segment sequence length (L)
PERIOD = 24                     # Period length (e.g., one day)

# Hyperparameters
INPUT_SIZE = WINDOW_SIZE        # Input size for VAE
HIDDEN_SIZE = 64
LATENT_SIZE = 16
LSTM_HIDDEN_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
LAMBDA_PARAM = 1.0              # Weight for balancing the loss functions
BATCH_SIZE = 32
K_THRESHOLD = 3                 # Number of standard deviations for anomaly detection

def generate_time_series_data():
    """
    Generates example time series data with missing values.
    """
    date_range = pd.date_range(start='2021-01-01', periods=500, freq='H')
    data = pd.DataFrame({'timestamp': date_range, 'value': np.sin(np.linspace(0, 50, 500))})
    data.set_index('timestamp', inplace=True)

    # Introduce missing values
    data.iloc[100:105] = np.nan  # Missing duration <= M
    data.iloc[200:210] = np.nan  # Missing duration > M

    return data

def fill_missing_values(data, M, period):
    """
    Fills missing values in the time series data according to specified rules.
    """
    data = data.copy()
    is_nan = data['value'].isna()
    nan_indices = np.where(is_nan)[0]

    i = 0
    while i < len(nan_indices):
        start = nan_indices[i]
        end = start
        while i + 1 < len(nan_indices) and nan_indices[i + 1] == nan_indices[i] + 1:
            i += 1
            end = nan_indices[i]
        duration = end - start + 1

        if duration <= M:
            # Linear interpolation with adjacent points
            x0 = data['value'].iloc[start - 1] if start > 0 else data['value'].iloc[end + 1]
            x1 = data['value'].iloc[end + 1] if end + 1 < len(data) else data['value'].iloc[start - 1]
            interpolated_values = np.linspace(x0, x1, duration + 2)[1:-1]
            data['value'].iloc[start:end + 1] = interpolated_values
        else:
            # Interpolation with the same time slot from adjacent periods
            for idx in range(start, end + 1):
                period_offset = period
                filled = False
                while not filled:
                    prev_idx = idx - period_offset
                    next_idx = idx + period_offset
                    if prev_idx >= 0 and not np.isnan(data['value'].iloc[prev_idx]):
                        data['value'].iloc[idx] = data['value'].iloc[prev_idx]
                        filled = True
                    elif next_idx < len(data) and not np.isnan(data['value'].iloc[next_idx]):
                        data['value'].iloc[idx] = data['value'].iloc[next_idx]
                        filled = True
                    else:
                        period_offset += period
                        if period_offset > len(data):
                            # Fallback if no data is found in adjacent periods
                            data['value'].iloc[idx] = data['value'].mean()
                            filled = True
        i += 1

    return data

def z_score_normalization(data, train_size):
    """
    Applies z-score normalization to the time series data.
    """
    train_mean = data['value'].iloc[:train_size].mean()  # Use training data for mean and std
    train_std = data['value'].iloc[:train_size].std()   # to avoid data leakage

    data['value_norm'] = (data['value'] - train_mean) / train_std # Normalize the data such that the mean is 0 and the standard deviation is 1
    return data, train_mean, train_std # Return the normalized data, mean, and standard deviation

def create_segments(data, window_size, sequence_length):
    """
    Creates segments and sequences from the time series data.
    """
    values = data['value_norm'].values # Use normalized values
    segments = [] # Initialize list to store segments
    for i in range(len(values) - window_size - 1):
        segment = values[i:i + window_size]
        target = values[i + window_size]
        segments.append((segment, target))

    # Organize segments into sequences
    sequences = []
    for i in range(len(segments) - sequence_length + 1):
        seq_segments = [segments[j][0] for j in range(i, i + sequence_length)]
        seq_targets = [segments[j][1] for j in range(i, i + sequence_length)]
        sequences.append((np.array(seq_segments), np.array(seq_targets)))

    return sequences

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data.
    """
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_segments, seq_targets = self.sequences[idx]
        return torch.tensor(seq_segments, dtype=torch.float32), torch.tensor(seq_targets, dtype=torch.float32)

class VAE(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    """
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor for trend prediction.
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predicting a single value

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last output
        out = out[:, -1, :]
        y_pred = self.fc(out)
        return y_pred.squeeze()

class SeqVL(nn.Module):
    """
    Sequential VAE-LSTM model integrating anomaly detection and trend prediction.
    """
    def __init__(self, input_size, hidden_size, latent_size, lstm_hidden_size):
        super(SeqVL, self).__init__()
        self.vae = VAE(input_size, hidden_size, latent_size)
        self.lstm = LSTMPredictor(input_size, lstm_hidden_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        x_recon_seq = []
        mu_seq = []
        logvar_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            x_recon_t, mu_t, logvar_t = self.vae(x_t)
            x_recon_seq.append(x_recon_t.unsqueeze(1))
            mu_seq.append(mu_t.unsqueeze(1))
            logvar_seq.append(logvar_t.unsqueeze(1))
        x_recon_seq = torch.cat(x_recon_seq, dim=1)
        mu_seq = torch.cat(mu_seq, dim=1)
        logvar_seq = torch.cat(logvar_seq, dim=1)

        # Trend prediction
        y_pred = self.lstm(x_recon_seq)

        return x_recon_seq, mu_seq, logvar_seq, y_pred

def vae_loss(x, x_recon, mu, logvar):
    """
    Computes the loss for the VAE component.
    """
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def lstm_loss(y_pred, y_true):
    """
    Computes the loss for the LSTM predictor.
    """
    return nn.MSELoss()(y_pred, y_true)

def train_model(model, train_loader, optimizer, num_epochs, lambda_param, sequence_length):
    """
    Trains the SeqVL model.
    """
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for seq_segments, seq_targets in train_loader:
            optimizer.zero_grad()
            x = seq_segments  # Shape: (batch_size, L, w0)
            y_true = seq_targets[:, -1]  # Predict the last target in the sequence

            x_recon_seq, mu_seq, logvar_seq, y_pred = model(x)

            # Compute losses
            vae_losses = []
            for t in range(sequence_length):
                vae_l = vae_loss(x[:, t, :], x_recon_seq[:, t, :], mu_seq[:, t, :], logvar_seq[:, t, :])
                vae_losses.append(vae_l)
            vae_loss_avg = torch.mean(torch.stack(vae_losses))
            lstm_l = lstm_loss(y_pred, y_true)

            loss = vae_loss_avg + lambda_param * lstm_l
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

def compute_reconstruction_errors(model, data_loader, sequence_length):
    """
    Computes reconstruction errors from the VAE component.
    """
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for seq_segments, _ in data_loader:
            x = seq_segments
            x_recon_seq, _, _, _ = model(x)
            # Compute the deviation of the last status in each segment
            errors = (x[:, -1, :] - x_recon_seq[:, -1, :]).pow(2).mean(dim=1)
            reconstruction_errors.extend(errors.numpy())
    return np.array(reconstruction_errors)

def detect_anomalies(model, data_loader, threshold):
    """
    Detects anomalies in the data based on reconstruction errors.
    """
    model.eval()
    anomalies = []
    with torch.no_grad():
        for seq_segments, _ in data_loader:
            x = seq_segments
            x_recon_seq, _, _, _ = model(x)
            errors = (x[:, -1, :] - x_recon_seq[:, -1, :]).pow(2).mean(dim=1)
            anomaly_flags = errors > threshold
            anomalies.extend(anomaly_flags.numpy())
    return anomalies

def evaluate_trend_prediction(model, data_loader):
    """
    Evaluates the trend prediction performance of the model.
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for seq_segments, seq_targets in data_loader:
            x = seq_segments
            y_true = seq_targets[:, -1]
            _, _, _, y_pred = model(x)
            all_y_true.extend(y_true.numpy())
            all_y_pred.extend(y_pred.numpy())

    mse = mean_squared_error(all_y_true, all_y_pred)
    print(f'Test MSE for Trend Prediction: {mse:.4f}')
    return mse

def main():
    # Generate and preprocess data
    data = generate_time_series_data()
    data_filled = fill_missing_values(data, MISSING_DURATION_THRESHOLD, PERIOD)
    train_size = int(0.8 * len(data_filled))
    data_filled, train_mean, train_std = z_score_normalization(data_filled, train_size)

    # Create sequences
    sequences = create_segments(data_filled, WINDOW_SIZE, SEQUENCE_LENGTH)

    # Split into training and testing datasets
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    train_dataset = TimeSeriesDataset(train_sequences)
    test_dataset = TimeSeriesDataset(test_sequences)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model and optimizer
    model = SeqVL(INPUT_SIZE, HIDDEN_SIZE, LATENT_SIZE, LSTM_HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, train_loader, optimizer, NUM_EPOCHS, LAMBDA_PARAM, SEQUENCE_LENGTH)

    # Compute reconstruction errors and threshold
    reconstruction_errors = compute_reconstruction_errors(model, train_loader, SEQUENCE_LENGTH)
    sigma_r = np.std(reconstruction_errors)
    threshold = K_THRESHOLD * sigma_r

    # Detect anomalies in test data
    anomalies = detect_anomalies(model, test_loader, threshold)
    print(f'Anomalies detected: {np.sum(anomalies)} out of {len(anomalies)} samples.')

    # Evaluate trend prediction performance
    evaluate_trend_prediction(model, test_loader)

if __name__ == "__main__":
    main()
