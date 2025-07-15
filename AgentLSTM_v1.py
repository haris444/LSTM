# Import required modules
import requests
import datetime
import pandas as pd
import time
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuration
TWELVE_DATA_API_KEY = "05b22eb6c3c449379f45ac6571f5a4b3"
ALPHA_VANTAGE_API_KEY = "T5JRS9Q6QUY44LHK"

# Global parameters
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2024, 1, 1)
interval = '1day'


# --- Utility Functions ---
def get_execution_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS GPU available")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU available")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def check_rate_limit(request_timestamps, max_requests=8, window_sec=60):
    now = time.time()
    pruned_timestamps = [t for t in request_timestamps if (now - t) < window_sec]
    used = len(pruned_timestamps)

    if used >= max_requests:
        oldest_in_window = pruned_timestamps[0]
        wait_time = int(window_sec - (now - oldest_in_window)) + 1
        raise RuntimeError(f"Rate limit exceeded. Wait ~{wait_time} seconds.")

    return pruned_timestamps, used, max_requests - used


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# --- Data Collection Functions ---
def fetch_stock_data(symbol, start_dt, end_dt, interval='1day', apikey=TWELVE_DATA_API_KEY):
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
        raise ValueError(f"Error from API for '{symbol}': {data.get('message', 'Unknown error')}")
    if "values" not in data:
        raise ValueError(f"No 'values' found for symbol '{symbol}'")

    df = pd.DataFrame(data["values"])

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' in response for symbol '{symbol}'")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    return df


def fetch_earnings_data(symbol, apikey=ALPHA_VANTAGE_API_KEY):
    print(f"Fetching earnings data for {symbol}")
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={apikey}'
    r = requests.get(url)
    data = r.json()

    if "quarterlyEarnings" not in data:
        print(f"Warning: Could not find earnings data for {symbol}")
        return pd.DataFrame()

    earnings_df = pd.DataFrame(data['quarterlyEarnings'])
    earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])
    earnings_df.rename(columns={'fiscalDateEnding': 'date', 'reportedEPS': 'eps'}, inplace=True)
    earnings_df['eps'] = pd.to_numeric(earnings_df['eps'], errors='coerce')
    earnings_df.dropna(subset=['eps'], inplace=True)
    earnings_df.sort_values('date', inplace=True)

    print(f"Successfully fetched {len(earnings_df)} earnings reports for {symbol}")
    return earnings_df[['date', 'eps']]


def collect_stock_data(symbol):
    print(f"Collecting data for {symbol}")

    # Fetch stock price data
    stock_df = fetch_stock_data(symbol, start_date, end_date, interval)
    print(f"Downloaded {symbol} stock data, rows={len(stock_df)}")

    # Fetch earnings data
    earnings_df = fetch_earnings_data(symbol)

    # Create combined dataset
    data = {
        'stock_data': stock_df,
        'earnings_data': earnings_df
    }

    return data


# --- Technical Indicators ---
def calculate_sma(price_series, window):
    return price_series.rolling(window=window, min_periods=1).mean()


def calculate_ema(price_series, window):
    return price_series.ewm(span=window, adjust=False, min_periods=1).mean()


def calculate_rsi(price_series, window=14):
    delta = price_series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


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
    return (np.sign(price_series.diff()) * volume_series).fillna(0).cumsum()


# --- Fundamental Indicators ---
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


def add_fundamental_indicators(df, earnings_df, symbol):
    price_col = f'{symbol}_close'

    if earnings_df.empty:
        print("Warning: Earnings data is empty. Skipping fundamental indicators")
        df['PE_Ratio'] = np.nan
        df['Earnings_Surprise'] = 0
        return df

    df['PE_Ratio'] = calculate_pe_ratio(df[price_col], earnings_df)

    # Calculate earnings surprise
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
    df['Earnings_Surprise'] = df['Earnings_Surprise'].replace(0, np.nan).ffill().fillna(0)

    print("Successfully added P/E Ratio and Earnings Surprise")
    return df


# --- Data Preparation ---
def prepare_dataframe(stock_data, earnings_data, symbol):
    print("Preparing dataframe with features")

    # Create main dataframe
    df = stock_data.copy()
    df.columns = [f'{symbol}_{col}' for col in df.columns]

    # Add fundamental indicators
    df = add_fundamental_indicators(df, earnings_data, symbol)

    # Feature engineering
    price_col = f'{symbol}_close'
    volume_col = f'{symbol}_volume'

    price = df[price_col]
    df['returns'] = price.pct_change().fillna(0)
    upper_bb, lower_bb = calculate_bollinger_bands(price, window=20)
    df['volatility'] = (upper_bb - lower_bb).div(price).fillna(0)

    # Target returns
    df['target_returns'] = df['returns'].shift(-1).fillna(0)

    # Normalize features
    feature_cols = [price_col, volume_col, 'returns']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    print("Feature engineering complete")
    return df, feature_cols


def create_indicator_banks(df, symbol, window_sizes):
    print(f"Creating indicator banks for {symbol}")

    price_col = f'{symbol}_close'
    volume_col = f'{symbol}_volume'
    price = df[price_col]
    volume = df[volume_col]
    epsilon = 1e-9
    indicator_dfs = []

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
    obv_df = pd.DataFrame(
        {f'obv_{w}': (obv - obv.rolling(w).mean()) / (obv.rolling(w).std() + epsilon) for w in window_sizes}).fillna(0)
    indicator_dfs.append(obv_df)

    # Indicator 6: MACD Signal Difference
    def macd_diff(price_series, window):
        macd_line, signal_line = calculate_macd(price, window_fast=window, window_slow=int(window * 2.5),
                                                window_signal=int(window * 0.9))
        return (macd_line - signal_line).fillna(0)

    macd_df = pd.DataFrame({f'macd_{w}': macd_diff(price, w) for w in window_sizes})
    indicator_dfs.append(macd_df)

    # Indicator 7: P/E Ratio
    pe_df = pd.DataFrame({f'pe_{w}': df['PE_Ratio'] for w in window_sizes}).fillna(0)
    indicator_dfs.append(pe_df)

    print(f"Created banks for {len(indicator_dfs)} indicators")
    return indicator_dfs


def prepare_training_data(df, indicator_dfs, feature_cols, seq_len=256):
    print("Creating sequences and sanitizing data")

    X_list, y_list, all_indicator_banks_list = [], [], []

    # Ensure indicator dataframes are aligned with main dataframe
    for ind_df in indicator_dfs:
        ind_df = ind_df.reindex(df.index).fillna(0)

    for i in range(len(df) - seq_len):
        X_list.append(df[feature_cols].iloc[i:i + seq_len].values)
        y_list.append(df['target_returns'].iloc[i + seq_len - 1])
        banks_for_step = [ind_df.iloc[i + seq_len - 1].values for ind_df in indicator_dfs]
        all_indicator_banks_list.append(np.array(banks_for_step))

    # Convert to tensors and handle NaN values
    X_np = np.nan_to_num(np.array(X_list))
    y_np = np.nan_to_num(np.array(y_list))
    indicator_banks_np = np.nan_to_num(np.array(all_indicator_banks_list))

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    indicator_banks = torch.tensor(indicator_banks_np, dtype=torch.float32)

    return X, y, indicator_banks


# --- LSTM Model ---
class LSTMTradingAgent(nn.Module):
    def __init__(self, d_input, d_hidden, num_indicators, window_sizes, dropout_prob=0.3, tau=1):
        super().__init__()
        self.num_indicators = num_indicators
        self.window_sizes = window_sizes
        self.M = len(window_sizes)
        self.tau = tau

        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, batch_first=True,
                            dropout=dropout_prob, num_layers=4)
        self.dropout = nn.Dropout(dropout_prob)

        self.window_heads = nn.ModuleList([nn.Linear(d_hidden, self.M) for _ in range(num_indicators)])
        self.threshold_heads = nn.ModuleList([nn.Linear(d_hidden, 2) for _ in range(num_indicators)])

        self.beta = nn.Parameter(torch.randn(num_indicators))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, indicator_bank, return_params=False):
        _, (h_n, _) = self.lstm(x)
        h_t = self.dropout(h_n[-1])

        signals = []
        all_weights = []
        all_thresholds = []
        all_indicator_values = []

        for i in range(self.num_indicators):
            alpha_logits = self.window_heads[i](h_t)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(alpha_logits)))
            weights = F.softmax((alpha_logits + gumbel_noise) / self.tau, dim=-1)

            I_i = indicator_bank[i]
            indicator_value = (weights * I_i).sum(dim=-1)

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
            return decision, torch.stack(all_weights, dim=1), torch.stack(all_thresholds, dim=1), torch.stack(
                all_indicator_values, dim=1)
        else:
            return decision


# --- Loss Functions ---
def stable_sharpe_loss(returns, epsilon=1e-9):
    return -(returns.mean() / (returns.std() + epsilon))


def get_performance(model, loader, device, num_indicators):
    model.eval()
    all_returns = []
    epsilon = 1e-9

    with torch.no_grad():
        for x, banks, y in loader:
            x, banks, y = x.to(device), banks.to(device), y.to(device)
            banks_list = [banks[:, i, :] for i in range(num_indicators)]
            decisions = model(x, banks_list)
            returns = decisions.flatten() * y.flatten()
            all_returns.append(returns)

    all_returns = torch.cat(all_returns)
    return (all_returns.mean() / (all_returns.std() + epsilon)).item()


def train_model(model, train_loader, val_loader, device, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = {'train_loss': [], 'train_sharpe': [], 'val_sharpe': []}

    print("\nStarting model training")

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for x_batch, banks_batch, y_batch in train_loader:
            x_batch, banks_batch, y_batch = x_batch.to(device), banks_batch.to(device), y_batch.to(device)
            banks_list_batch = [banks_batch[:, i, :] for i in range(model.num_indicators)]

            optimizer.zero_grad()
            decisions = model(x_batch, banks_list_batch)
            returns = decisions.flatten() * y_batch.flatten()
            loss = stable_sharpe_loss(returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        train_sharpe = get_performance(model, train_loader, device, model.num_indicators)
        val_sharpe = get_performance(model, val_loader, device, model.num_indicators)

        history['train_loss'].append(epoch_loss)
        history['train_sharpe'].append(train_sharpe)
        history['val_sharpe'].append(val_sharpe)

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Train Sharpe: {train_sharpe:.4f} | Val Sharpe: {val_sharpe:.4f}")

    print("\nTraining complete")
    final_val_sharpe = get_performance(model, val_loader, device, model.num_indicators) * np.sqrt(252)
    print(f"Final Annualized Validation Sharpe Ratio: {final_val_sharpe:.4f}")

    return history


# --- Visualization Functions ---
def plot_dynamic_parameters(dates, weights, thresholds, indicator_values, indicator_names, window_sizes,
                            indicator_to_plot_idx=0):
    indicator_name = indicator_names[indicator_to_plot_idx]
    w = weights[:, indicator_to_plot_idx, :]
    theta_plus = thresholds[:, indicator_to_plot_idx, 0]
    theta_minus = thresholds[:, indicator_to_plot_idx, 1]
    i_hat = indicator_values[:, indicator_to_plot_idx]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plt.style.use('dark_background')

    # Dynamic Window Weights
    axes[0].stackplot(dates, w.T, labels=[f'Win {size}' for size in window_sizes], alpha=0.8)
    axes[0].set_title(f'Dynamic Window Weights for {indicator_name}', fontsize=14)
    axes[0].set_ylabel('Attention Weight')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Dynamic Thresholds vs Indicator Value
    axes[1].plot(dates, i_hat, label=f'Soft Indicator Value', color='white', linewidth=2)
    axes[1].plot(dates, theta_plus, label=f'Upper Threshold', color='lime', linestyle='--')
    axes[1].plot(dates, theta_minus, label=f'Lower Threshold', color='red', linestyle='--')
    axes[1].fill_between(dates, theta_minus, theta_plus, color='gray', alpha=0.2, label='Neutral Zone')

    axes[1].set_title(f'Dynamic Thresholds for {indicator_name}', fontsize=14)
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Date')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_dynamic_parameters(model, loader, device):
    model.eval()
    all_w, all_theta, all_I_hat = [], [], []
    with torch.no_grad():
        for x, banks, y in loader:
            x, banks = x.to(device), banks.to(device)
            banks_list = [banks[:, i, :] for i in range(model.num_indicators)]
            _, weights, thresholds, ind_values = model(x, banks_list, return_params=True)
            all_w.append(weights.cpu().numpy())
            all_theta.append(thresholds.cpu().numpy())
            all_I_hat.append(ind_values.cpu().numpy())

    return np.concatenate(all_w, axis=0), np.concatenate(all_theta, axis=0), np.concatenate(all_I_hat, axis=0)


def visualize_lstm_performance(model, test_loader, original_test_df, decision_dates, symbol, device=None):
    if device is None:
        device = get_execution_device()

    model.eval()
    all_decisions = []

    with torch.no_grad():
        for x_batch, banks_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            banks_batch = banks_batch.to(device)
            banks_list_batch = [banks_batch[:, i, :] for i in range(model.num_indicators)]
            decisions = model(x_batch, banks_list_batch)
            all_decisions.extend(decisions.cpu().numpy().flatten())

    decisions_series = pd.Series(all_decisions, index=decision_dates)

    # Portfolio simulation
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    shares_held = 0
    portfolio_values = []
    portfolio_dates = []
    trade_log = []

    trade_threshold = 0.7
    transaction_fee = 5.0
    lambda_worst_case = 1.2

    price_col = f'{symbol}_close'

    for i in range(len(original_test_df)):
        date = original_test_df.index[i]
        current_price = original_test_df[price_col].iloc[i]

        if date in decisions_series.index:
            decision = decisions_series[date]

            # Close existing position if signal changes
            if position != 0 and (abs(decision) < trade_threshold or np.sign(decision) != position):
                if position == 1:  # Close long
                    proceeds = (shares_held * current_price) - transaction_fee
                    capital += proceeds
                    trade_log.append({'date': date, 'action': 'close_long', 'price': current_price,
                                      'shares': shares_held, 'proceeds': proceeds})
                else:  # Close short
                    cost = (shares_held * current_price) + transaction_fee
                    capital -= cost
                    trade_log.append({'date': date, 'action': 'close_short', 'price': current_price,
                                      'shares': shares_held, 'cost': cost})
                position = 0
                shares_held = 0

            # Open new position
            if position == 0:
                if decision > trade_threshold:  # Go long
                    shares_to_buy = int((capital - transaction_fee) / current_price)
                    if shares_to_buy > 0:
                        cost = (shares_to_buy * current_price) + transaction_fee
                        capital -= cost
                        shares_held = shares_to_buy
                        position = 1
                        trade_log.append({'date': date, 'action': 'open_long', 'price': current_price,
                                          'shares': shares_held, 'cost': cost})

                elif decision < -trade_threshold:  # Go short
                    worst_case_price = lambda_worst_case * current_price
                    if capital > transaction_fee:
                        shares_to_short = int((capital - transaction_fee) / worst_case_price)
                    else:
                        shares_to_short = 0
                    if shares_to_short > 0:
                        proceeds = (shares_to_short * current_price) - transaction_fee
                        capital += proceeds
                        shares_held = shares_to_short
                        position = -1
                        trade_log.append({'date': date, 'action': 'open_short', 'price': current_price,
                                          'shares': shares_held, 'proceeds': proceeds})

        # Calculate total portfolio value
        current_position_value = 0
        if position == 1:
            current_position_value = shares_held * current_price
        elif position == -1:
            current_position_value = -1 * shares_held * current_price

        portfolio_value = capital + current_position_value
        portfolio_values.append(portfolio_value)
        portfolio_dates.append(date)

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

        print(f"\n--- LSTM Performance Analysis for {symbol} ---")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
        print(f"Cumulative Return: {cumulative_return:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {len([t for t in trade_log if 'open' in t['action']])}")

        decision_stats = decisions_series.describe()
        print(f"\nDecision Signal Stats:")
        print(f"Above threshold (+{trade_threshold}): {(decisions_series > trade_threshold).sum()}")
        print(f"Below threshold (-{trade_threshold}): {(decisions_series < -trade_threshold).sum()}")

    # Create visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Portfolio value plot
    axes[0].plot(portfolio_series.index, portfolio_series, label='Total Portfolio Value', color='cyan', linewidth=2)
    axes[0].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[0].set_title(f'LSTM Portfolio Performance - {symbol}', fontsize=14)
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)

    # Mark trades
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
    axes[1].plot(original_test_df.index, original_test_df[price_col],
                 label=f'{symbol} Price', color='white', alpha=0.9, linewidth=1.5)
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

    # Decision signals
    axes[2].plot(decisions_series.index, decisions_series, label='LSTM Decision', color='orange', alpha=0.8,
                 linewidth=1.5)
    axes[2].axhline(y=trade_threshold, color='lime', linestyle=':', alpha=0.7, label=f'Long Threshold')
    axes[2].axhline(y=-trade_threshold, color='red', linestyle=':', alpha=0.7, label=f'Short Threshold')
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].fill_between(decisions_series.index, -trade_threshold, trade_threshold, alpha=0.1, color='gray',
                         label='Neutral Zone')
    axes[2].set_title('LSTM Decision Signals')
    axes[2].set_ylabel('Decision Value')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return trade_log, portfolio_series


# --- Main Execution Functions ---
def run_lstm_trading_system(symbol="MSFT"):
    """
    Complete LSTM trading system for a single stock

    Args:
        symbol (str): Stock symbol to trade (default: "MSFT")
    """
    print(f"=== LSTM Trading System for {symbol} ===")

    device = get_execution_device()

    # Step 1: Data Collection
    print(f"\n--- Step 1: Collecting Data for {symbol} ---")
    try:
        data = collect_stock_data(symbol)
        stock_data = data['stock_data']
        earnings_data = data['earnings_data']
    except Exception as e:
        print(f"Error collecting data: {e}")
        return None

    # Step 2: Data Preparation
    print(f"\n--- Step 2: Preparing Data ---")
    df, feature_cols = prepare_dataframe(stock_data, earnings_data, symbol)

    # Step 3: Create Indicator Banks
    print(f"\n--- Step 3: Creating Indicator Banks ---")
    window_sizes = [5, 10, 20, 50, 100, 200]
    indicator_dfs = create_indicator_banks(df, symbol, window_sizes)

    # Step 4: Prepare Training Data
    print(f"\n--- Step 4: Preparing Training Data ---")
    X, y, indicator_banks = prepare_training_data(df, indicator_dfs, feature_cols)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    banks_train, banks_val = indicator_banks[:train_size], indicator_banks[train_size:]

    train_dataset = TensorDataset(X_train, banks_train, y_train)
    val_dataset = TensorDataset(X_val, banks_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    print(f"Data prepared. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Step 5: Initialize and Train Model
    print(f"\n--- Step 5: Training LSTM Model ---")
    d_input = len(feature_cols)
    d_hidden = 32
    num_indicators = len(indicator_dfs)

    model = LSTMTradingAgent(d_input, d_hidden, num_indicators, window_sizes, dropout_prob=0.5).to(device)
    history = train_model(model, train_loader, val_loader, device)

    # Step 6: Visualization and Analysis
    print(f"\n--- Step 6: Analysis and Visualization ---")

    # Prepare data for visualization
    test_dates = df.index[-len(X_val):]
    original_test_df = df.loc[test_dates]

    # Visualize dynamic parameters
    indicator_names = [
        "SMA Difference", "EMA Difference", "RSI", "Bollinger Band Position",
        "On-Balance Volume (OBV)", "MACD Signal Difference", "P/E Ratio"
    ]

    print("Generating dynamic parameter visualizations")
    weights, thresholds, ind_values = get_dynamic_parameters(model, val_loader, device)

    for i in range(len(indicator_names)):
        print(f"Plotting parameters for: {indicator_names[i]} ({i + 1}/{len(indicator_names)})")
        plot_dynamic_parameters(
            dates=test_dates,
            weights=weights,
            thresholds=thresholds,
            indicator_values=ind_values,
            indicator_names=indicator_names,
            window_sizes=window_sizes,
            indicator_to_plot_idx=i
        )

    # Generate performance visualization
    trade_log, portfolio_series = visualize_lstm_performance(
        model, val_loader, original_test_df, test_dates, symbol, device=device
    )

    # Save results
    results = {
        'symbol': symbol,
        'model': model,
        'history': history,
        'trade_log': trade_log,
        'portfolio_series': portfolio_series,
        'test_dates': test_dates,
        'feature_cols': feature_cols
    }

    print(f"\n=== {symbol} Trading System Complete ===")
    return results


# --- Main Execution ---
if __name__ == "__main__":
    # Run for Microsoft by default, but can easily change the symbol
    results = run_lstm_trading_system("MSFT")

    # Example: Run for different stocks
    # results_googl = run_lstm_trading_system("GOOGL")
    #results_aapl = run_lstm_trading_system("AAPL")