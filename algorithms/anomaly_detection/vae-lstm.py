import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# let's write explanation comments aside each line of code

# Step 1: Load and Preprocess Data
def load_data(file_path, window_size): # file_path is the path to the CSV file containing the data, and window_size is the size of the sliding window
    """Load data from CSV and preprocess into sliding windows."""
    data = pd.read_csv(file_path) # Load data from CSV file
    sensor_values = data.drop(columns=['timestamp']).values # Extract sensor values
    mean = np.mean(sensor_values, axis=0) # Calculate mean of sensor values
    std = np.std(sensor_values, axis=0) # Calculate standard deviation of sensor values
    normalized_data = (sensor_values - mean) / std # Normalize sensor values

    windows = [] # Initialize list to store sliding windows
    
     # the loop below creates sliding windows of size window_size from the normalized data
    for i in range(len(normalized_data) - window_size): 
        windows.append(normalized_data[i:i + window_size])
   
    return np.array(windows)

class VAE(Model):
    def __init__(self, window_size, latent_dim): # window_size is the size of the sliding window, and latent_dim is the dimension of the latent space
        super(VAE, self).__init__() # Initialize the VAE model  what is super here? 
                                    # super() is a built-in Python function that returns a temporary object of the superclass that allows you to call its methods.
        self.latent_dim = latent_dim # Set the latent dimension
        
        # Encoder
        self.encoder_hidden1 = Dense(100, activation='relu') # Define the first hidden layer of the encoder with 100 units and ReLU activation function each unit is a neuron in the neural network
        self.encoder_hidden2 = Dense(100, activation='relu') # Define the second hidden layer of the encoder with 100 units and ReLU activation function each unit is a neuron in the neural network
        self.mu_layer = Dense(latent_dim) # Define the mean layer of the encoder with the latent dimension as the output size .. what is mu_layer?
        #answer : mu_layer is the layer that outputs the mean of the latent space why
        self.log_var_layer = Dense(latent_dim)
        
        # Decoder
        self.decoder_hidden1 = Dense(100, activation='relu')
        self.decoder_hidden2 = Dense(100, activation='relu')
        self.decoder_output = Dense(window_size, activation='linear')

    def encode(self, x):
        """Encoder network."""
        h = self.encoder_hidden1(x)
        h = self.encoder_hidden2(h)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    def decode(self, z):
        """Decoder network."""
        h = self.decoder_hidden1(z)
        h = self.decoder_hidden2(h)
        return self.decoder_output(h)

    def call(self, x):
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)

        # KL Divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss))

        return reconstruction

def build_vae(window_size, latent_dim):
    """Build the VAE model."""
    vae = VAE(window_size, latent_dim)
    vae.compile(optimizer=Adam(1e-3), loss='mse')
    return vae

# Step 3: Define LSTM Block
def build_lstm(window_size):
    """Build the LSTM block."""
    lstm_input = Input(shape=(window_size, 1))  # Reshape to 3D for LSTM
    lstm = LSTM(100, activation='tanh', return_sequences=False)(lstm_input)
    output = Dense(1, activation='linear')(lstm)

    lstm_model = Model(lstm_input, output, name='LSTM')
    lstm_model.compile(optimizer=Adam(1e-3), loss='mse')
    return lstm_model

# Step 4: Training and Integration
def train_seqvl(vae, lstm, data, window_size):
    """Train the SeqVL model."""
    # Split data
    train_size = int(0.7 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]

    # Train VAE
    x_train = train_data.reshape(-1, window_size)
    vae.fit(x_train, x_train, epochs=50, batch_size=64, validation_split=0.2)

    # Generate reconstructed data
    reconstructed_data = vae.predict(x_train)
    reconstructed_data = reconstructed_data.reshape(-1, window_size, 1)  # Reshape for LSTM

    # Ensure target alignment
    y_train = train_data[window_size:, -1]
    reconstructed_data = reconstructed_data[:len(y_train)]  # Trim if needed

    # Train LSTM on reconstructed data
    lstm.fit(reconstructed_data, y_train, epochs=50, batch_size=64, validation_split=0.2)

    return vae, lstm


# Step 5: Evaluation
def evaluate_seqvl(vae, lstm, test_data, window_size):
    """Evaluate the SeqVL model."""
    # Create sliding windows for test data
    windows = []
    targets = []
    for i in range(len(test_data) - window_size):
        windows.append(test_data[i:i + window_size])
        targets.append(test_data[i + window_size, -1])  # Extract target from the last feature

    windows = np.array(windows)
    targets = np.array(targets)

    # Reshape windows for VAE
    windows = windows.reshape(-1, window_size)

    # Reconstruct data using VAE
    reconstructed_test = vae.predict(windows)
    reconstructed_test = reconstructed_test.reshape(-1, window_size, 1)  # Reshape for LSTM

    # Predict using LSTM
    predictions = lstm.predict(reconstructed_test)

    # Calculate MSE between true targets and predictions
    mse = np.mean(np.square(targets - predictions.squeeze()))
    print(f"Test MSE: {mse}")
    print("Test data shape:", test_data.shape)
    print("Windows shape:", windows.shape)
    print("Targets shape:", targets.shape)
    print("Reconstructed test shape:", reconstructed_test.shape)
    print("Predictions shape:", predictions.shape)
    return mse


# Example Usage
if __name__ == "__main__":
    FILE_PATH = 'yahoo_sub_5.csv'
    WINDOW_SIZE = 120
    LATENT_DIM = 5

    # Load data
    data = load_data(FILE_PATH, WINDOW_SIZE)

    # Build models
    vae = build_vae(WINDOW_SIZE, LATENT_DIM)
    lstm = build_lstm(WINDOW_SIZE)

    # Train models
    vae, lstm = train_seqvl(vae, lstm, data, WINDOW_SIZE)
    



    # Evaluate
    evaluate_seqvl(vae, lstm, data, WINDOW_SIZE)
