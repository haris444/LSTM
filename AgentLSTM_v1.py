# Import required Python modules.
import requests
import datetime
import pandas as pd
import time
import pickle
import os

# Set the Twelve Data and Alpha Vantage API Keys
TWELVE_DATA_API_KEY = "05b22eb6c3c449379f45ac6571f5a4b3"
ALPHA_VANTAGE_API_KEY = "T5JRS9Q6QUY44LHK"


def check_rate_limit(request_timestamps, max_requests=8, window_sec=60):
    now = time.time()
    pruned_timestamps = [t for t in request_timestamps if (now - t) < window_sec]
    used = len(pruned_timestamps)
    left = max_requests - used

    if used >= max_requests:
        oldest_in_window = pruned_timestamps[0]
        wait_time = int(window_sec - (now - oldest_in_window)) + 1
        raise RuntimeError(
            f"Rate limit exceeded: {max_requests} requests in the last {window_sec}s.\n"
            f"Please wait ~{wait_time} seconds before trying again."
        )

    return pruned_timestamps, used, left


def fetch_twelvedata(symbol, start_dt, end_dt, interval='1day', apikey=TWELVE_DATA_API_KEY):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": interval,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "apikey": apikey, "outputsize": 5000
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if "status" in data and data["status"] == "error":
        raise ValueError(f"Error from Twelve Data API for '{symbol}': {data.get('message', 'Unknown error')}")
    if "values" not in data:
        raise ValueError(f"No 'values' found for symbol '{symbol}'. Response: {data}")

    df = pd.DataFrame(data["values"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' in response for symbol '{symbol}'.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    return df


def fetch_earnings_data(symbol, apikey=ALPHA_VANTAGE_API_KEY):
    """
    Fetches historical quarterly earnings data from Alpha Vantage.
    """
    print(f"Fetching earnings data for {symbol} from Alpha Vantage...")
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={apikey}'
    r = requests.get(url)
    data = r.json()

    if "quarterlyEarnings" not in data:
        print(f"Warning: Could not find earnings data for {symbol}. Response: {data}")
        return pd.DataFrame()

    earnings_df = pd.DataFrame(data['quarterlyEarnings'])
    earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])
    earnings_df.rename(columns={'fiscalDateEnding': 'date', 'reportedEPS': 'eps'}, inplace=True)

    # Convert EPS to numeric, coercing errors to NaN
    earnings_df['eps'] = pd.to_numeric(earnings_df['eps'], errors='coerce')

    # Drop rows with NaN EPS values and sort by date
    earnings_df.dropna(subset=['eps'], inplace=True)
    earnings_df.sort_values('date', inplace=True)

    print(f"Successfully fetched {len(earnings_df)} earnings reports for {symbol}.")
    return earnings_df[['date', 'eps']]


def download_dataset(td_symbol_groups, apikey=TWELVE_DATA_API_KEY):
    try:
        collected_data = {}
        request_timestamps = []

        for group_name, symbols in td_symbol_groups.items():
            print(f"\n--- Downloading {group_name} ---")
            group_data = {}
            for symbol in symbols:
                request_timestamps, used, left = check_rate_limit(request_timestamps)
                print(f"Request usage in last 60s: used={used}, left={left}")
                try:
                    df = fetch_twelvedata(symbol, start_date, end_date, interval, apikey)
                    group_data[symbol] = df
                    print(f"Downloaded {symbol}, rows={len(df)}")
                except Exception as ex:
                    print(f"Error downloading {symbol}: {ex}")
                request_timestamps.append(time.time())
            collected_data[group_name] = group_data

        # ----> ADDED: FETCH AND STORE EARNINGS DATA FOR MSFT <----
        msft_earnings = fetch_earnings_data("MSFT")
        if not msft_earnings.empty:
            collected_data['fundamental'] = {'MSFT_EARNINGS': msft_earnings}

        print("\nAll downloads finished.\n")
        return collected_data
    except RuntimeError as e:
        print(f"\n** Rate Limit Error **\n{e}")

start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2024, 1, 1)
interval = '1day'

td_symbol_groups = {
    "stocks": ["MSFT", "IBM", "GOOGL"],
    "currencies": ["USD/JPY", "GBP/USD"],
    "indices": ["SPY", "DIA", "QQQ"]
}

collected_data = download_dataset(td_symbol_groups, apikey=TWELVE_DATA_API_KEY)

# Save the collected data
data_directory = './data'
data_file = 'collected_data.pkl'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
save_path = os.path.join(data_directory, data_file)

with open(save_path, "wb") as f:
    pickle.dump(collected_data, f)
print(f"Data successfully saved to '{save_path}'")


import pandas as pd
import pickle
import os

def prepare_daily_data(pickle_path='data/collected_data.pkl', output_csv='data/daily_data.csv'):
    """
    Loads raw data, extracts daily close and volume for all symbols,
    synchronizes them to a common index, and saves to a CSV file.

    Args:
        pickle_path (str): Path to the input pickle file.
        output_csv (str): Path to save the final CSV file.
    """
    print("Starting data preparation...")
    # Load the collected data from the pickle file
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Pickle file not found at '{pickle_path}'. Please run the data collection script first.")

    with open(pickle_path, "rb") as f:
        collected_data = pickle.load(f)

    # Dictionary to hold the relevant series (close and volume) for each symbol
    all_series = {}

    # Extract 'close' and 'volume' for each symbol
    for group, group_data in collected_data.items():
        for symbol, df in group_data.items():
            if 'close' in df.columns:
                all_series[f'{symbol}_close'] = df['close']
            if 'volume' in df.columns:
                all_series[f'{symbol}_volume'] = df['volume']

    # Create a unified DataFrame from all the extracted series
    main_df = pd.DataFrame(all_series)

    # The index is already a DatetimeIndex. Let's ensure it's sorted.
    main_df.sort_index(inplace=True)

    # Forward-fill missing values to handle non-trading days
    main_df.ffill(inplace=True)
    # Backward-fill any remaining NaNs at the beginning
    main_df.bfill(inplace=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save the prepared data to a new CSV file
    main_df.to_csv(output_csv)
    print(f"Data preparation complete. Clean data saved to '{output_csv}'")


prepare_daily_data()

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- From indicators.py ---

def calculate_sma(price_series, window):
    return price_series.rolling(window=window, min_periods=1).mean()

def calculate_ema(price_series, window):
    return price_series.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rsi(price_series, window=14):
    delta = price_series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(price_series, window_slow=26, window_fast=12, window_signal=9):
    ema_fast = calculate_ema(price_series, window=window_fast)
    ema_slow = calculate_ema(price_series, window=window_slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, window=window_signal)
    return macd_line, signal_line

def calculate_bollinger_bands(price_series, window=20, num_std_dev=2):
    sma = calculate_sma(price_series, window)
    std_dev = price_series.rolling(window=window, min_periods=1).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

def calculate_obv(price_series, volume_series):
    obv = (np.sign(price_series.diff()) * volume_series).fillna(0).cumsum()
    return obv

# --- From crossover_indicator_generator.py ---

def generate_sma_crossover_indicator(data, price_col, short_window, long_window):
    sma_short = calculate_sma(data[price_col], short_window)
    sma_long = calculate_sma(data[price_col], long_window)
    crossover_state = pd.Series(0, index=data.index)
    crossover_state[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))] = 1
    crossover_state[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))] = -1
    return crossover_state

def generate_ema_crossover_indicator(data, price_col, short_window, long_window):
    ema_short = calculate_ema(data[price_col], short_window)
    ema_long = calculate_ema(data[price_col], long_window)
    crossover_state = pd.Series(0, index=data.index)
    crossover_state[(ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))] = 1
    crossover_state[(ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))] = -1
    return crossover_state

def generate_macd_crossover_indicator(data, price_col, fast, slow, signal):
    macd_line, signal_line = calculate_macd(data[price_col], slow, fast, signal)
    crossover_state = pd.Series(0, index=data.index)
    crossover_state[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
    crossover_state[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
    return crossover_state

def add_all_crossover_indicators(data, params):
    price_col = params['price_column']
    data['sma_crossover'] = generate_sma_crossover_indicator(
        data, price_col, params['sma_short_window'], params['sma_long_window']
    )
    data['ema_crossover'] = generate_ema_crossover_indicator(
        data, price_col, params['ema_short_window'], params['ema_long_window']
    )
    data['macd_crossover'] = generate_macd_crossover_indicator(
        data, price_col, params['macd_fast'], params['macd_slow'], params['macd_signal']
    )
    return data

# --- From fundamental_indicators.py ---

def calculate_pe_ratio(price_series, earnings_data):
    pe_ratios = pd.Series(index=price_series.index, dtype=float)
    reporting_dates = earnings_data['date'].tolist()
    eps_values = earnings_data['eps'].tolist()

    for i, date in enumerate(price_series.index):
        prior_reports_dates = [d for d in reporting_dates if d <= date]
        if prior_reports_dates:
            latest_report_date = max(prior_reports_dates)
            latest_eps = eps_values[reporting_dates.index(latest_report_date)]
            if latest_eps > 0:
                pe_ratios.iloc[i] = price_series.iloc[i] / latest_eps
            else:
                pe_ratios.iloc[i] = np.nan
        else:
            pe_ratios.iloc[i] = np.nan
    pe_ratios.ffill(inplace=True)
    return pe_ratios

def estimate_earnings_with_model(historical_earnings, forecast_periods=1):
    if len(historical_earnings) < 4:
        return [historical_earnings[-1]] * forecast_periods if historical_earnings else [0]
    X = np.arange(len(historical_earnings)).reshape(-1, 1)
    y = np.array(historical_earnings)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(historical_earnings), len(historical_earnings) + forecast_periods).reshape(-1, 1)
    return model.predict(future_X)

def calculate_earnings_surprise(actual_earnings, expected_earnings):
    if expected_earnings == 0:
        return 0.0
    return ((actual_earnings - expected_earnings) / abs(expected_earnings)) * 100

def add_fundamental_indicators_to_data(df, earnings_df, price_column='MSFT_close'):
    if earnings_df.empty:
        print("Warning: Earnings data is empty. Skipping fundamental indicators.")
        df['PE_Ratio'] = np.nan
        df['Earnings_Surprise'] = 0
        return df

    df['PE_Ratio'] = calculate_pe_ratio(df[price_column], earnings_df)
    historical_eps = earnings_df['eps'].tolist()
    reporting_dates = earnings_df['date'].tolist()
    surprise_series = pd.Series(index=df.index, dtype=float, data=0.0)

    for i in range(len(historical_eps)):
        if i >= 4:
            expected_eps = estimate_earnings_with_model(historical_eps[:i])[0]
            actual_eps = historical_eps[i]
            surprise = calculate_earnings_surprise(actual_eps, expected_eps)
            report_date = reporting_dates[i]
            if report_date in df.index:
                surprise_series.loc[report_date] = surprise
            else:
                closest_idx = df.index.get_indexer([report_date], method='nearest')[0]
                closest_date = df.index[closest_idx]
                surprise_series.loc[closest_date] = surprise
    df['Earnings_Surprise'] = surprise_series
    print("Successfully added P/E Ratio and truly discrete Earnings Surprise.")
    return df

def add_fundamental_indicators_to_data_q2(df, earnings_df, price_column='MSFT_close'):
    """
    Version for Question 2 where forward-filling earnings surprise is acceptable.
    """
    df = add_fundamental_indicators_to_data(df, earnings_df, price_column)
    # For Q2: Forward-fill the earnings surprise signal
    df['Earnings_Surprise'] = df['Earnings_Surprise'].replace(0, np.nan).ffill().fillna(0)
    print("Applied forward-filling for Question 2 implementation.")
    return df


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def get_execution_device():
    if hasattr(torch.backends, "mps"):
        is_mps = torch.backends.mps.is_available()
    else:
        is_mps = False
    is_cuda = torch.cuda.is_available()
    if is_mps:
        device = torch.device("mps")
        print("MPS GPU is available!")
    elif is_cuda:
        device = torch.device("cuda")
        print("CUDA GPU is available!")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_execution_device()

# --- Model Definition with Regularization ---
# This class is included here for completeness. It includes Dropout to prevent overfitting.
class LSTMTradingAgent(nn.Module):
    def __init__(self, d_input, d_hidden, num_indicators, window_sizes, dropout_prob=0.3, tau=1.0):
        super().__init__()
        self.num_indicators = num_indicators
        self.window_sizes = window_sizes
        self.M = len(window_sizes)
        self.tau = tau

        # A deeper, 2-layer LSTM with Dropout is more robust
        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True, dropout=dropout_prob, num_layers=2)
        self.dropout = nn.Dropout(dropout_prob)

        self.window_heads = nn.ModuleList([
            nn.Linear(d_hidden, self.M) for _ in range(num_indicators)
        ])
        self.threshold_heads = nn.ModuleList([
            nn.Linear(d_hidden, 2) for _ in range(num_indicators)
        ])

        self.beta = nn.Parameter(torch.randn(num_indicators))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, indicator_bank, debug=False, return_params=False):
        _, (h_n, _) = self.lstm(x)
        h_t = self.dropout(h_n[-1])

        signals = []
        all_weights = []
        all_thresholds = []
        all_indicator_values = []

        for i in range(self.num_indicators):
            alpha_logits = self.window_heads[i](h_t)
            # Gumbel-Softmax Sampling [cite: 242]
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(alpha_logits)))
            weights = F.softmax((alpha_logits + gumbel_noise) / self.tau, dim=-1)

            # Soft Indicator Computation [cite: 245]
            I_i = indicator_bank[i]
            indicator_value = (weights * I_i).sum(dim=-1)

            # Threshold Prediction [cite: 247]
            thresholds = self.threshold_heads[i](h_t)
            theta_plus, theta_minus = thresholds[:, 0], thresholds[:, 1]

            s_i = torch.sigmoid((indicator_value - theta_plus) / 0.1) - \
                  torch.sigmoid((theta_minus - indicator_value) / 0.1)
            signals.append(s_i)

            if return_params:
                all_weights.append(weights)
                all_thresholds.append(thresholds)
                all_indicator_values.append(indicator_value)

        S = torch.stack(signals, dim=-1)
        decision = torch.tanh(S @ self.beta + self.bias)

        if return_params:
            # Return decision and stacked parameters for each indicator
            return decision, torch.stack(all_weights, dim=1), torch.stack(all_thresholds, dim=1), torch.stack(all_indicator_values, dim=1)
        else:
            return decision

class TradingDataset(Dataset):
    def __init__(self, X, indicator_bank, returns):
        self.X = X.to(device)
        self.indicator_bank = [ib.to(device) for ib in indicator_bank]
        self.returns = returns.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        indicators = [bank[idx] for bank in self.indicator_bank]
        y = self.returns[idx]
        return x, indicators, y

def sharpe_loss(returns):
    mean = returns.mean()
    std = returns.std(unbiased=False) + 1e-6
    return -mean / std

def return_loss(returns):
    return -returns.sum()

def hybrid_loss(returns, mu_sharpe=0.5, mu_ret=0.5):
    return mu_sharpe * sharpe_loss(returns) + mu_ret * return_loss(returns)


def get_dynamic_parameters(model, loader, device):
    """
    Runs the model on a dataset and collects the learned dynamic parameters.
    """
    model.eval()
    all_w, all_theta, all_I_hat = [], [], []
    with torch.no_grad():
        for x, banks, y in loader:
            x, banks = x.to(device), banks.to(device)
            banks_list = [banks[:, i, :] for i in range(model.num_indicators)]

            # Use the modified forward pass to get parameters
            _, weights, thresholds, ind_values = model(x, banks_list, return_params=True)

            all_w.append(weights.cpu().numpy())
            all_theta.append(thresholds.cpu().numpy())
            all_I_hat.append(ind_values.cpu().numpy())

    # Concatenate results from all batches
    return np.concatenate(all_w, axis=0), np.concatenate(all_theta, axis=0), np.concatenate(all_I_hat, axis=0)


def plot_dynamic_parameters(dates, weights, thresholds, indicator_values, indicator_names, window_sizes, indicator_to_plot_idx=0):
    """
    Visualizes the learned dynamic window sizes and thresholds over time for a selected indicator.
    """
    indicator_name = indicator_names[indicator_to_plot_idx]

    # Extract data for the selected indicator
    w = weights[:, indicator_to_plot_idx, :]
    theta_plus = thresholds[:, indicator_to_plot_idx, 0]
    theta_minus = thresholds[:, indicator_to_plot_idx, 1]
    i_hat = indicator_values[:, indicator_to_plot_idx]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plt.style.use('dark_background')

    # --- Plot 1: Dynamic Window Weights ---
    axes[0].stackplot(dates, w.T, labels=[f'Win {size}' for size in window_sizes], alpha=0.8)
    axes[0].set_title(f'Dynamic Window Weights for {indicator_name}', fontsize=14)
    axes[0].set_ylabel('Attention Weight')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # --- Plot 2: Dynamic Thresholds vs. Indicator Value ---
    axes[1].plot(dates, i_hat, label=f'Soft Indicator Value ($\hat{{I}}(t)$)', color='white', linewidth=2)
    axes[1].plot(dates, theta_plus, label=f'Upper Threshold ($\Theta^+(t)$)', color='lime', linestyle='--')
    axes[1].plot(dates, theta_minus, label=f'Lower Threshold ($\Theta^-(t)$)', color='red', linestyle='--')
    axes[1].fill_between(dates, theta_minus, theta_plus, color='gray', alpha=0.2, label='Neutral Zone')

    axes[1].set_title(f'Dynamic Thresholds for {indicator_name}', fontsize=14)
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Date')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt



# --- 1. Load Data and Engineer Features ---
print("\nLoading and Preparing Data for Full LSTM Agent...")
try:
    df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
    with open('data/collected_data.pkl', 'rb') as f:
        collected_data = pickle.load(f)
    msft_earnings_df = collected_data.get('fundamental', {}).get('MSFT_EARNINGS', pd.DataFrame())
    df = add_fundamental_indicators_to_data_q2(df, msft_earnings_df, price_column='MSFT_close')
    price = df['MSFT_close']
    volume = df['MSFT_volume']
    df['returns'] = price.pct_change().fillna(0)
    upper_bb, lower_bb = calculate_bollinger_bands(price, window=20)
    df['volatility'] = (upper_bb - lower_bb).div(price).fillna(0)
    raw_features = ['MSFT_close', 'MSFT_volume', 'returns', 'volatility', 'PE_Ratio', 'Earnings_Surprise']
    df['target_returns'] = df['returns'].shift(-1).fillna(0)
    for col in raw_features:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    print("Feature engineering complete.")
except FileNotFoundError:
    print("ERROR: Data files not found.")
    assert False

# --- 2. Create Indicator Bank for ALL Indicators ---
print("Creating comprehensive indicator bank for all 7 indicators...")
window_sizes = [5, 10, 20, 50, 100, 200]
indicator_dfs = []
epsilon = 1e-9

# Indicator 1: SMA Difference
def sma_diff(price_series, window):
    sma_short = calculate_sma(price_series, window)
    sma_long = calculate_sma(price_series, int(window * 2.5))
    return (sma_short - sma_long).fillna(0)
sma_df = pd.DataFrame({f'sma_{w}': sma_diff(price, w) for w in window_sizes})
indicator_dfs.append(sma_df)

# Indicator 2: EMA Difference
def ema_diff(price_series, window):
    ema_short = calculate_ema(price_series, window)
    ema_long = calculate_ema(price_series, int(window * 2.5))
    return (ema_short - ema_long).fillna(0)
ema_df = pd.DataFrame({f'ema_{w}': ema_diff(price, w) for w in window_sizes})
indicator_dfs.append(ema_df)

# Indicator 3: RSI
rsi_df = pd.DataFrame({f'rsi_{w}': calculate_rsi(price, w) for w in window_sizes}).fillna(50)
indicator_dfs.append(rsi_df)

# Indicator 4: Bollinger Band Position
def bb_position(price_series, window):
    upper, lower = calculate_bollinger_bands(price_series, window)
    return ((price_series - lower) / (upper - lower + epsilon)).fillna(0.5)
bb_df = pd.DataFrame({f'bb_{w}': bb_position(price, w) for w in window_sizes})
indicator_dfs.append(bb_df)

# Indicator 5: On-Balance Volume
obv = calculate_obv(price, volume)
obv_df = pd.DataFrame({f'obv_{w}': (obv - obv.rolling(w).mean()) / (obv.rolling(w).std() + epsilon) for w in window_sizes}).fillna(0)
indicator_dfs.append(obv_df)

# Indicator 6: MACD Signal Difference
def macd_diff(price_series, window):
    macd_line, signal_line = calculate_macd(price, window_fast=window, window_slow=int(window*2.5), window_signal=int(window*0.9))
    return (macd_line - signal_line).fillna(0)
macd_df = pd.DataFrame({f'macd_{w}': macd_diff(price, w) for w in window_sizes})
indicator_dfs.append(macd_df)

# Indicator 7: P/E Ratio (as a signal)
# The bank will contain the same PE value repeated, letting the LSTM learn its importance
pe_df = pd.DataFrame({f'pe_{w}': df['PE_Ratio'] for w in window_sizes}).fillna(0)
indicator_dfs.append(pe_df)

print(f"Created banks for {len(indicator_dfs)} indicators.")

# --- 3. Create Sequences and Sanitize Data ---
seq_len = 256
X_list, y_list, all_indicator_banks_list = [], [], []
for ind_df in indicator_dfs:
    ind_df = ind_df.reindex(df.index).fillna(0)
for i in range(len(df) - seq_len):
    X_list.append(df[raw_features].iloc[i:i+seq_len].values)
    y_list.append(df['target_returns'].iloc[i+seq_len-1])
    banks_for_step = [ind_df.iloc[i+seq_len-1].values for ind_df in indicator_dfs]
    all_indicator_banks_list.append(np.array(banks_for_step))
X_np = np.nan_to_num(np.array(X_list))
y_np = np.nan_to_num(np.array(y_list))
indicator_banks_np = np.nan_to_num(np.array(all_indicator_banks_list))
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)
indicator_banks = torch.tensor(indicator_banks_np, dtype=torch.float32)

# --- 4. Split Data and Create DataLoaders ---
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]
banks_train, banks_val = indicator_banks[:train_size], indicator_banks[train_size:]
train_dataset = TensorDataset(X_train, banks_train, y_train)
val_dataset = TensorDataset(X_val, banks_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
print(f"Data prepared. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

# --- 5. Initialize Model, Optimizer, and Run Training ---
d_input = len(raw_features)
d_hidden = 32
num_indicators = len(indicator_dfs)
epochs = 150

model = LSTMTradingAgent(d_input, d_hidden, num_indicators, window_sizes, dropout_prob=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# --- 6. Training Loop with Validation ---
def stable_sharpe_loss(returns, epsilon=1e-9):
    return -(returns.mean() / (returns.std() + epsilon))

def get_performance(loader):
    model.eval()
    all_returns = []
    with torch.no_grad():
        for x, banks, y in loader:
            x, banks, y = x.to(device), banks.to(device), y.to(device)
            banks_list = [banks[:, i, :] for i in range(num_indicators)]
            decisions = model(x, banks_list)
            returns = decisions.flatten() * y.flatten()
            all_returns.append(returns)
    all_returns = torch.cat(all_returns)
    return (all_returns.mean() / (all_returns.std() + epsilon)).item()

history = {'train_loss': [], 'train_sharpe': [], 'val_sharpe': []}
print("\nStarting model training with all indicators and regularization...")
for epoch in range(epochs):
    model.train()
    batch_losses = []
    for x_batch, banks_batch, y_batch in train_loader:
        x_batch, banks_batch, y_batch = x_batch.to(device), banks_batch.to(device), y_batch.to(device)
        banks_list_batch = [banks_batch[:, i, :] for i in range(num_indicators)]
        optimizer.zero_grad()
        decisions = model(x_batch, banks_list_batch)
        returns = decisions.flatten() * y_batch.flatten()
        loss = stable_sharpe_loss(returns)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_loss = np.mean(batch_losses)
    train_sharpe = get_performance(train_loader)
    val_sharpe = get_performance(val_loader)
    history['train_loss'].append(epoch_loss)
    history['train_sharpe'].append(train_sharpe)
    history['val_sharpe'].append(val_sharpe)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Sharpe: {train_sharpe:.4f} | Val Sharpe: {val_sharpe:.4f}")

print("\nTraining complete.")
final_val_sharpe = get_performance(val_loader) * np.sqrt(252)
print(f"\nFinal Annualized Validation Sharpe Ratio: {final_val_sharpe:.4f}")



# --- Set up for final visualization in the next cell ---
test_loader = val_loader
original_df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
test_dates = df.index[-len(X_val):]
original_test_df = original_df.loc[test_dates]



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_lstm_performance(model, test_loader, original_test_df, decision_dates, company_ticker="GOOGL"):
    """
    Evaluates LSTM model with proper portfolio value tracking (cash + unrealized P&L)
    """
    model.eval()
    all_decisions = []

    # Get model decisions
    with torch.no_grad():
        for x_batch, banks_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            banks_batch = banks_batch.to(device)
            banks_list_batch = [banks_batch[:, i, :] for i in range(model.num_indicators)]
            decisions = model(x_batch, banks_list_batch)
            all_decisions.extend(decisions.cpu().numpy().flatten())

    decisions_series = pd.Series(all_decisions, index=decision_dates)

    # Portfolio simulation with proper total value tracking
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    shares_held = 0
    portfolio_values = []
    portfolio_dates = []
    trade_log = []

    trade_threshold = 0.7
    transaction_fee = 5.0  # Use float for consistency
    lambda_worst_case = 1.2  # Worst-case multiplier for short selling

    for i in range(len(original_test_df)):
        date = original_test_df.index[i]
        current_price = original_test_df[f'{company_ticker}_close'].iloc[i]

        if date in decisions_series.index:
            decision = decisions_series[date]

            # Close existing position if signal changes or weakens
            if position != 0 and (abs(decision) < trade_threshold or np.sign(decision) != position):
                if position == 1:  # Close long
                    # MODIFIED: Subtract exit fee from sale proceeds
                    proceeds = (shares_held * current_price) - transaction_fee
                    capital += proceeds
                    trade_log.append(
                        {'date': date, 'action': 'close_long', 'price': current_price, 'shares': shares_held,
                         'proceeds': proceeds})
                else:  # Close short
                    # MODIFIED: Add exit fee to buy-back cost
                    cost = (shares_held * current_price) + transaction_fee
                    capital -= cost
                    trade_log.append(
                        {'date': date, 'action': 'close_short', 'price': current_price, 'shares': shares_held,
                         'cost': cost})

                position = 0
                shares_held = 0

            # Open new position if no current position
            if position == 0:
                if decision > trade_threshold:  # Go long
                    # MODIFIED: Account for entry fee when calculating shares to buy
                    shares_to_buy = int((capital - transaction_fee) / current_price)
                    if shares_to_buy > 0:
                        # MODIFIED: Add entry fee to purchase cost
                        cost = (shares_to_buy * current_price) + transaction_fee
                        capital -= cost
                        shares_held = shares_to_buy
                        position = 1
                        trade_log.append(
                            {'date': date, 'action': 'open_long', 'price': current_price, 'shares': shares_held,
                             'cost': cost})

                elif decision < -trade_threshold:  # Go short
                    # MODIFIED: Account for entry fee when calculating shares to short
                    worst_case_price = lambda_worst_case * current_price
                    if capital > transaction_fee:  # Ensure we can at least pay the fee
                        shares_to_short = int((capital - transaction_fee) / worst_case_price)
                    else:
                        shares_to_short = 0
                    if shares_to_short > 0:
                        # MODIFIED: Subtract entry fee from short sale proceeds
                        proceeds = (shares_to_short * current_price) - transaction_fee
                        capital += proceeds
                        shares_held = shares_to_short
                        position = -1
                        trade_log.append(
                            {'date': date, 'action': 'open_short', 'price': current_price, 'shares': shares_held,
                             'proceeds': proceeds})

        # Calculate TOTAL portfolio value = cash + unrealized P&L
        current_position_value = 0
        if position == 1:  # Long position
            current_position_value = shares_held * current_price
        elif position == -1:  # Short position
            current_position_value = -1 * shares_held * current_price

        portfolio_value = capital + current_position_value
        portfolio_values.append(portfolio_value)
        portfolio_dates.append(date)

    # Convert to pandas series for easier plotting
    portfolio_series = pd.Series(portfolio_values, index=portfolio_dates)

    # Calculate performance metrics
    if len(portfolio_values) > 1:
        cumulative_return = (portfolio_values[-1] / initial_capital - 1) * 100
        returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = 0
        if returns.std() > 1e-9:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        annualized_volatility = returns.std() * np.sqrt(252) * 100
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        print(f"\n--- LSTM Performance Analysis ---")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
        print(f"Cumulative Return: {cumulative_return:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {len([t for t in trade_log if 'open' in t['action']])}")
        print(f"Final Position: {'Long' if position == 1 else 'Short' if position == -1 else 'None'}")
        print(f"Final Cash: ${capital:,.2f}")
        if shares_held > 0:
            print(f"Shares Held: {shares_held:,}")

        # Analysis of decision patterns
        decision_stats = decisions_series.describe()
        print(f"\nDecision Signal Stats:")
        print(f"Mean: {decision_stats['mean']:.4f}")
        print(f"Std: {decision_stats['std']:.4f}")
        print(f"Above threshold (+{trade_threshold}): {(decisions_series > trade_threshold).sum()}")
        print(f"Below threshold (-{trade_threshold}): {(decisions_series < -trade_threshold).sum()}")
        print(f"In neutral zone: {(abs(decisions_series) <= trade_threshold).sum()}")

    # Create visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Portfolio value plot (including unrealized P&L)
    axes[0].plot(portfolio_series.index, portfolio_series, label='Total Portfolio Value', color='cyan', linewidth=2)
    axes[0].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[0].set_title(f'LSTM Portfolio Performance - {company_ticker} (Cash + Unrealized P&L)', fontsize=14)
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)

    # Mark trades on portfolio chart
    for trade in trade_log:
        trade_value = portfolio_series.loc[trade['date']]
        if 'open_long' in trade['action']:
            axes[0].scatter(trade['date'], trade_value, color='lime', s=80, marker='^', alpha=0.8, zorder=5)
        elif 'open_short' in trade['action']:
            axes[0].scatter(trade['date'], trade_value, color='red', s=80, marker='v', alpha=0.8, zorder=5)
        elif 'close' in trade['action']:
            axes[0].scatter(trade['date'], trade_value, color='yellow', s=60, marker='o', alpha=0.8, zorder=5)

    axes[0].legend()

    # Stock price with trades
    price_col = f'{company_ticker}_close'
    axes[1].plot(original_test_df.index, original_test_df[price_col],
                label=f'{company_ticker} Price', color='white', alpha=0.9, linewidth=1.5)

    for trade in trade_log:
        if 'open_long' in trade['action']:
            axes[1].scatter(trade['date'], trade['price'], color='lime', s=100, marker='^', alpha=0.8, zorder=5)
        elif 'open_short' in trade['action']:
            axes[1].scatter(trade['date'], trade['price'], color='red', s=100, marker='v', alpha=0.8, zorder=5)
        elif 'close' in trade['action']:
            axes[1].scatter(trade['date'], trade['price'], color='yellow', s=80, marker='o', alpha=0.8, zorder=5)

    axes[1].set_title('Stock Price with Trade Signals')
    axes[1].set_ylabel('Price ($)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Decision signals over time
    axes[2].plot(decisions_series.index, decisions_series, label='LSTM Decision', color='orange', alpha=0.8, linewidth=1.5)
    axes[2].axhline(y=trade_threshold, color='lime', linestyle=':', alpha=0.7, label=f'Long Threshold (+{trade_threshold})')
    axes[2].axhline(y=-trade_threshold, color='red', linestyle=':', alpha=0.7, label=f'Short Threshold (-{trade_threshold})')
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].fill_between(decisions_series.index, -trade_threshold, trade_threshold, alpha=0.1, color='gray', label='Neutral Zone')
    axes[2].set_title('LSTM Decision Signals')
    axes[2].set_ylabel('Decision Value')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return trade_log, portfolio_series

print("\n--- Generating Dynamic Parameter Visualizations ---")

# Define the names of your indicators in the order they were created in the indicator bank
indicator_names = [
    "SMA Difference", "EMA Difference", "RSI", "Bollinger Band Position",
    "On-Balance Volume (OBV)", "MACD Signal Difference", "P/E Ratio"
]

print("\n--- Generating Dynamic Parameter Visualizations for All Indicators ---")



# 1. Extract the parameters from the test (validation) set once
weights, thresholds, ind_values = get_dynamic_parameters(model, test_loader, device)

# 2. Loop through and plot the parameters for each of the 7 indicators
for i in range(len(indicator_names)):
    print(f"\nPlotting parameters for: {indicator_names[i]} (Indicator {i+1}/{len(indicator_names)})...")
    plot_dynamic_parameters(
        dates=test_dates,
        weights=weights,
        thresholds=thresholds,
        indicator_values=ind_values,
        indicator_names=indicator_names,
        window_sizes=window_sizes,
        indicator_to_plot_idx=i
    )

# Usage
trade_log, portfolio_series = visualize_lstm_performance(model, test_loader, original_test_df, test_dates, company_ticker="MSFT")