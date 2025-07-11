# =============================================================================
# This script file provides fundamental computational functionality for the 
# implementation of Case Study I: Supervised Learning Models for Stock Market
# Prediction.
# =============================================================================

# =============================================================================
# Problem Setup: Multi-Output Time Series Forecasting with Encoder-Decoder LSTM
# =============================================================================

# =============================================================================
#                             Objective:
# =============================================================================
# Train an LSTM-based encoder-decoder model to predict multiple future values 
# of a univariate financial time series based on a fixed-length input sequence 
# of past values.

# Input:
# - A time series of real-valued observations (e.g., asset prices or returns).
# - At each training step, we provide the model with a sequence of length 
#   past_window_size.

# Output:
# - The model is trained to predict the next future_window_size values that follow 
#   the input window.

# Model:
# - An encoder LSTM compresses the input sequence into a hidden state.
# - A decoder LSTM then unfolds this representation to generate multiple future values, 
#   one step at a time, using its previous output as input (autoregressive decoding).

# Use Case:
# - This setup is useful in financial forecasting tasks such as:
#   - Predicting asset prices for multiple future timesteps
#   - Anticipating return curves or volatility trends
#   - Multi-step ahead prediction for decision-making (e.g., portfolio rebalancing)

# Assumptions:
# - We assume a univariate time series with some predictable structure.
# - The time series is regular (uniform time intervals).
# - Future values depend on the recent past but may involve nonlinear dynamics.


# Import required Pythob libraries.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from auxiliary.utils import get_execution_device, plot_training_history

# =============================================================================
# Custom PyTorch Dataset for windowed time series forecasting
# =============================================================================

# -----------------------------------------------------------------------------
# Produces input-output pairs of form:
#   X: [past_window_size] → input sequence
#   Y: [future_window_size] → output sequence to forecast
# -----------------------------------------------------------------------------

class TimeSeriesMultiOutputDataset(Dataset):
    def __init__(self, series, past_window_size, future_window_size, device):
        
        # Input Arguments:
        #    series (np.ndarray): Raw time series data (scaled), shape = (T,) 
        #                         or (T, d)
        #    past_window_size (int): Number of past time steps as input
        #    future_window_size (int): Number of future time steps as target
        #    device (torch.device): Target device to store the dataset
        
        # Store config
        self.device = device
        self.past = past_window_size
        self.future = future_window_size

        # Convert the full time series to tensor on the target device
        self.series = torch.tensor(series, dtype=torch.float32, device=device)

        # Generate all (X, Y) pairs
        self.X, self.Y = self.generate_datapoints()

    def generate_datapoints(self):
        # Create a list of fixed-length input-output pairs
        X, Y = [], []
        for i in range(len(self.series) - self.past - self.future + 1):
            x = self.series[i : i + self.past]
            y = self.series[i + self.past : i + self.past + self.future]
            X.append(x)
            Y.append(y)

        # Stack into tensors (on the device already)
        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# =============================================================================
# Encoder-Decoder LSTM Model (Seq2Seq)
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)  # Return only the final states
        return h_n, c_n

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, y_init, h_n, c_n, steps):
        outputs = []
        input_t = y_init
        for _ in range(steps):
            out, (h_n, c_n) = self.lstm(input_t, (h_n, c_n))
            pred = self.fc(out)
            outputs.append(pred.squeeze(1))
            input_t = pred  # Use previous output as next input
        return torch.stack(outputs, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, future_window_size):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(input_dim, hidden_dim, num_layers)
        self.future_window_size = future_window_size

    def forward(self, x):
        h_n, c_n = self.encoder(x)
        y_init = x[:, -1:, :]  # Use last encoder input as decoder's first input
        return self.decoder(y_init, h_n, c_n, self.future_window_size)


# ============================================================================= 
#                 TRAINING AND TESTING FUNCTIONS SECTION:
# =============================================================================

# This function evaluates the model on a given dataloader.
def test_model(model, dataloader, loss_fn, device):
    
    # Output Arguments:
    #    avg_loss: Average loss across batches
    #    avg_mae:  Average MAE across all predictions
    #    y_true:   Flattened target values (NumPy)
    #    y_pred:   Flattened predicted values (NumPy)
    
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, Y in dataloader:
            output = model(X)
            loss = loss_fn(output, Y)

            total_loss += loss.item()
            all_preds.append(output.cpu().squeeze(-1).numpy())
            all_targets.append(Y.cpu().squeeze(-1).numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    avg_loss = total_loss / len(dataloader)
    avg_mae = mean_absolute_error(y_true, y_pred)

    return avg_loss, avg_mae, y_pred, y_true

# This function trains and evaluates the model after each epoch.
def train_model(model, train_loader, test_loader, optimizer, loss_fn, device, epochs=10):
    
    # Output Arguments:
    #    TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE: lists of per-epoch metrics
    
    model.to(device)
    
    # Metric storage
    TRAIN_LOSS, TRAIN_MAE = [], []
    TEST_LOSS, TEST_MAE = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        for X, Y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(output.detach().cpu().squeeze(-1).numpy())
            all_targets.append(Y.detach().cpu().squeeze(-1).numpy())

        # Compute training metrics
        train_loss_avg = total_loss / len(train_loader)
        train_preds_flat = np.concatenate(all_preds)
        train_targets_flat = np.concatenate(all_targets)
        train_mae = mean_absolute_error(train_targets_flat, train_preds_flat)

        TRAIN_LOSS.append(train_loss_avg)
        TRAIN_MAE.append(train_mae)

        # Evaluate on test set
        test_loss_avg, test_mae, _, _ = test_model(model, test_loader, loss_fn, device)
        TEST_LOSS.append(test_loss_avg)
        TEST_MAE.append(test_mae)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss_avg:.4f}, MAE: {train_mae:.4f} | "
              f"Test Loss: {test_loss_avg:.4f}, MAE: {test_mae:.4f}")

    return TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE


# ============================================================================= 
#                 PERFORMANCE VISUALIZATION FUNCTIONS SECTION:
# =============================================================================

# This function accumulates all true and predicted values for each forecast 
# step t+1 to t+future_window_size for both training and testing sets.
def accumulate_forecast_targets_preds(model, train_loader, test_loader, loss_fn, device):
    

    # Output Arguments:
    #    results: Dictionary with structure:
    #        {
    #            'train': {
    #                k: (y_true_step_k, y_pred_step_k)  # for forecast step t+(k+1)
    #                for k in range(future_window_size)
    #            },
    #            'test': {
    #                k: (y_true_step_k, y_pred_step_k)
    #                for k in range(future_window_size)
    #            }
    #        }
    
    model.to(device)
    model.eval()

    def collect(loader):
        y_preds, y_trues = [], []
        with torch.no_grad():
            for X, Y in loader:
                output = model(X)  # (batch, future_window_size, 1)
                y_preds.append(output.cpu().squeeze(-1).numpy())
                y_trues.append(Y.cpu().squeeze(-1).numpy())
        y_preds = np.concatenate(y_preds)  # shape: (N, future_window_size)
        y_trues = np.concatenate(y_trues)
        return y_trues, y_preds

    train_true, train_pred = collect(train_loader)
    test_true, test_pred = collect(test_loader)

    future_window_size = train_true.shape[1]
    results = {'train': {}, 'test': {}}

    for k in range(future_window_size):
        results['train'][k] = (train_true[:, k], train_pred[:, k])
        results['test'][k] = (test_true[:, k], test_pred[:, k])

    return results

# This function generates one scatter plot figure per forecast step, showing 
# true vs predicted values for both train and test sets.
def visualize_model_performance(results):
    
    # Input Arguments:
    #    results (dict): Output from accumulate_forecast_targets_preds(...)
    #                    Structure:
    #                    {
    #                      'train': {k: (true_vals, pred_vals)},
    #                      'test':  {k: (true_vals, pred_vals)}
    #                    }
    
    future_window_size = len(results['train'])

    for k in range(future_window_size):
        y_true_train, y_pred_train = results['train'][k]
        y_true_test, y_pred_test = results['test'][k]

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_train, y_pred_train, alpha=0.5, label='Train', color='blue')
        plt.scatter(y_true_test, y_pred_test, alpha=0.5, label='Test', color='red')

        # Diagonal line for perfect prediction
        all_y = np.concatenate([y_true_train, y_true_test])
        min_y, max_y = np.min(all_y), np.max(all_y)
        plt.plot([min_y, max_y], [min_y, max_y], '--', color='gray')

        plt.title(f"Forecast Step t+{k+1}")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ============================================================================= 
#                   MAIN CODE SECTION:
# =============================================================================

# ============================================================================= 
# Step 1: Load and preprocess data
# =============================================================================

# Set the name of the data directory.
data_directory = './data'
# The files that can be used are either the XXXX_time_series_data.csv which 
# contain as features lagged versions of only the target regression variable
# or XXXX_multi_time_series_data.csv which contain as features lagged versions
# of all the available data variables.
data_file = 'MSFT_time_series_data.csv'

# Construct the full data path.
data_path = os.path.join(data_directory,data_file)

# Load the DataFrame from the CSV file.
df = pd.read_csv(data_path)

# Drop Date column
df = df.drop(columns=['Date'])

# Acquire all dataframe series objects that store the value of a financial 
# instrument at t = 0.
columns_t0 = [col for col in df.columns if col.endswith('_t-0')]
time_series_df = df[columns_t0]

# ============================================================================= 
# Step 2 : Partition original time series into training and testing sets
# ============================================================================= 

# -----------------------------------------------------------------------------
# IMPORTANT: Do NOT shuffle time series data. Temporal order must be preserved
# to avoid mixing past and future — a common source of data leakage in forecasting.
# -----------------------------------------------------------------------------

test_ratio = 0.2  # Reserve 20% of the sequence for testing

# Assuming time_series_df is a pandas DataFrame with one or more features
train_time_series_df, test_time_series_df = train_test_split(
    time_series_df, 
    test_size=test_ratio, 
    shuffle=False
)

# ============================================================================= 
# Step 3: Scale features using StandardScaler
# =============================================================================

# -----------------------------------------------------------------------------
# Fit the scaler only on training data to avoid leaking test information.
# Then apply the transformation to both train and test sets.
# -----------------------------------------------------------------------------

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_time_series_df.values)
test_scaled = scaler.transform(test_time_series_df.values)

# ============================================================================= 
# Step 4: Configure dataset-related parameters.
# =============================================================================

# Set the past and future windows sizes as well as the batch size to be used 
# both during training and testing.
past_window_size = 30
future_window_size = 10
batch_size = 32

# Get the execution device.
# Get the execution device to be used for training and testing the model.
device = get_execution_device()

# ============================================================================= 
# Step 5: Create training and testing datasets from scaled data
# ============================================================================= 

train_dataset = TimeSeriesMultiOutputDataset(train_scaled, past_window_size, 
                                             future_window_size, device)
test_dataset = TimeSeriesMultiOutputDataset(test_scaled, past_window_size, 
                                            future_window_size, device)

# Note: DataLoader will operate on tensors that are already on the device
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================= 
# Step 6: Configure model-related parameters.
# =============================================================================

hidden_dim = 50
num_layers = 2
learning_rate = 0.001
epochs = 100

# ============================================================================= 
# Step 7: Initialize model, optimizer, loss
# =============================================================================

input_dim = train_scaled.shape[1]  # e.g., 1 for univariate
model = Seq2Seq(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, 
                future_window_size=future_window_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# ============================================================================= 
# Step 8: Train and evaluate the model.
# =============================================================================

TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    epochs=epochs
)

# Plot the training and testing history.
plot_training_history(TRAIN_LOSS, TRAIN_MAE, TEST_LOSS, TEST_MAE)

# ============================================================================= 
# Step 9: Visualize model performance for each future time step.
# =============================================================================
results = accumulate_forecast_targets_preds(model, train_loader, test_loader, 
                                            loss_fn, device)
visualize_model_performance(results)