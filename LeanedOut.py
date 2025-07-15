# Import required modules
import requests
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Configuration & Globals ---
TWELVE_DATA_API_KEY = "05b22eb6c3c449379f45ac6571f5a4b3"
ALPHA_VANTAGE_API_KEY = "5ALT0MIBRGWUG07N"
START_DATE = datetime.datetime(2010, 1, 1)
END_DATE = datetime.datetime(2025, 1, 1)
INTERVAL = '1day'


# --- Utility Functions ---
def get_execution_device():
    """Gets the best available device (MPS, CUDA, or CPU) for PyTorch."""
    if torch.backends.mps.is_available():
        print("Using MPS GPU")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# --- Data Collection ---
def fetch_stock_data(symbol, start_dt, end_dt, interval='1day', api_key=TWELVE_DATA_API_KEY):
    """Fetches historical stock data from the Twelve Data API."""
    params = {
        "symbol": symbol, "interval": interval,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "apikey": api_key, "outputsize": 5000
    }
    response = requests.get("https://api.twelvedata.com/time_series", params=params)
    response.raise_for_status()  # Raises HTTPError for bad responses
    data = response.json()

    if data.get("status") == "error":
        raise ValueError(f"API Error for '{symbol}': {data.get('message')}")

    df = pd.DataFrame(data["values"]).rename(columns={"datetime": "date"})
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").set_index("date")


def fetch_earnings_data(symbol, api_key=ALPHA_VANTAGE_API_KEY):
    """Fetches quarterly earnings data from the Alpha Vantage API."""
    print(f"Fetching earnings data for {symbol}")
    params = {'function': 'EARNINGS', 'symbol': symbol, 'apikey': api_key}
    response = requests.get('https://www.alphavantage.co/query', params=params)
    response.raise_for_status()
    data = response.json()

    if not data.get('quarterlyEarnings'):
        print(f"Warning: No earnings data found for {symbol}.")
        return pd.DataFrame()

    earnings_df = (
        pd.DataFrame(data['quarterlyEarnings'])
        .rename(columns={'fiscalDateEnding': 'date', 'reportedEPS': 'eps'})
        .assign(
            date=lambda x: pd.to_datetime(x['date']),
            eps=lambda x: pd.to_numeric(x['eps'], errors='coerce')
        )
        .dropna(subset=['eps'])
        .sort_values('date')
    )
    print(f"Fetched {len(earnings_df)} earnings reports for {symbol}.")
    return earnings_df[['date', 'eps']]


def collect_stock_data(symbol):
    """Collects and combines stock prices and earnings data."""
    print(f"Collecting data for {symbol}")
    stock_df = fetch_stock_data(symbol, START_DATE, END_DATE, INTERVAL)
    earnings_df = fetch_earnings_data(symbol)
    return {'stock_data': stock_df, 'earnings_data': earnings_df}


# --- Technical Indicators ---
def calculate_sma(price_series, window):
    return price_series.rolling(window=window, min_periods=1).mean()


def calculate_ema(price_series, window):
    return price_series.ewm(span=window, adjust=False).mean()


def calculate_rsi(price_series, window=14):
    delta = price_series.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
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
    std_dev = price_series.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band


def calculate_obv(price_series, volume_series):
    return (np.sign(price_series.diff()) * volume_series).fillna(0).cumsum()


# --- Fundamental Indicators ---
def calculate_pe_ratio(price_series, earnings_df):
    """Vectorized P/E ratio calculation using a forward-fill merge."""
    if earnings_df.empty:
        return pd.Series(index=price_series.index, dtype=float)

    # Prepare earnings data for merging
    earnings_df = earnings_df.set_index('date').sort_index()
    earnings_df = earnings_df[earnings_df['eps'] > 0]

    # Use merge_asof to find the last known EPS for each trading day
    temp_df = pd.merge_asof(
        left=price_series.to_frame('price').sort_index(),
        right=earnings_df[['eps']],
        left_index=True,
        right_index=True,
        direction='backward'
    )
    return temp_df['price'] / temp_df['eps']


def add_fundamental_indicators(df, earnings_df, symbol):
    """Adds P/E ratio and earnings surprise to the main DataFrame."""
    price_col = f'{symbol}_close'

    if earnings_df.empty:
        print("Warning: Skipping fundamental indicators due to empty earnings data.")
        df['PE_Ratio'] = np.nan
        df['Earnings_Surprise'] = 0
        return df

    # Calculate P/E Ratio
    df['PE_Ratio'] = calculate_pe_ratio(df[price_col], earnings_df)

    # Calculate Earnings Surprise
    if len(earnings_df) >= 5:
        earnings_df['time_idx'] = np.arange(len(earnings_df))
        X = earnings_df[['time_idx']].iloc[:-1]  # Use all but the last report for fitting
        y = earnings_df['eps'].iloc[:-1]

        # Simple rolling window linear regression to estimate next EPS
        # For each report, we predict using a model trained on all *prior* reports
        expected_eps = []
        for i in range(4, len(earnings_df)):
            X_train = np.arange(i).reshape(-1, 1)
            y_train = earnings_df['eps'].iloc[:i].values
            model = LinearRegression().fit(X_train, y_train)
            expected_eps.append(model.predict(np.array([[i]]))[0])

        actual_eps = earnings_df['eps'].iloc[4:].values
        surprise_pct = (actual_eps - expected_eps) / (np.abs(expected_eps) + 1e-9)

        surprise_df = pd.DataFrame({
            'date': earnings_df['date'].iloc[4:],
            'Earnings_Surprise': surprise_pct
        }).set_index('date')

        # Merge surprise into the main df, forward-filling the values
        df = df.merge(surprise_df, left_index=True, right_index=True, how='left')
        df['Earnings_Surprise'].ffill(inplace=True)

    df.fillna({'PE_Ratio': 0, 'Earnings_Surprise': 0}, inplace=True)
    print("Added P/E Ratio and Earnings Surprise.")
    return df


# --- Data Preparation ---
def prepare_dataframe(stock_data, earnings_data, symbol):
    """Prepares the master dataframe with all features and targets."""
    print("Preparing dataframe with features.")
    df = stock_data.copy().add_prefix(f'{symbol}_')
    df = add_fundamental_indicators(df, earnings_data, symbol)

    price_col = f'{symbol}_close'
    price = df[price_col]

    # Feature Engineering
    df['returns'] = price.pct_change()
    upper_bb, lower_bb = calculate_bollinger_bands(price, window=20)
    df['volatility'] = (upper_bb - lower_bb) / (price + 1e-9)
    df['target_returns'] = df['returns'].shift(-1)  # Target is next day's return

    # Normalize features
    feature_cols = [f'{symbol}_close', f'{symbol}_volume', 'returns', 'volatility']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    print("Feature engineering complete.")
    return df, feature_cols


def create_indicator_banks(df, symbol, window_sizes):
    """Creates banks of technical indicators with varying window sizes."""
    print(f"Creating indicator banks for {symbol}")
    price = df[f'{symbol}_close']
    volume = df[f'{symbol}_volume']
    epsilon = 1e-9

    # Define indicator calculations as lambdas
    indicators = {
        'sma': lambda w: calculate_sma(price, w) - calculate_sma(price, int(w * 2.5)),
        'ema': lambda w: calculate_ema(price, w) - calculate_ema(price, int(w * 2.5)),
        'rsi': lambda w: calculate_rsi(price, w),
        'bb': lambda w: (price - calculate_bollinger_bands(price, w)[1]) / (
                    calculate_bollinger_bands(price, w)[0] - calculate_bollinger_bands(price, w)[1] + epsilon),
        'obv': lambda w:
        (obv := calculate_obv(price, volume), (obv - obv.rolling(w).mean()) / (obv.rolling(w).std() + epsilon))[1],
        'macd': lambda w: (macd_line := calculate_macd(price, w, int(w * 2.5), int(w * 0.9))[0],
                           macd_line - calculate_macd(price, w, int(w * 2.5), int(w * 0.9))[1])[1],
        'pe': lambda w: df['PE_Ratio']
    }

    # Generate dataframes for each indicator bank
    indicator_dfs = [pd.DataFrame({f'{name}_{w}': func(w) for w in window_sizes}) for name, func in indicators.items()]

    print(f"Created banks for {len(indicator_dfs)} indicators.")
    return indicator_dfs


def create_sequences(df, indicator_dfs, feature_cols, seq_len=1000):
    """Converts dataframes into sequences for LSTM training."""
    print("Creating sequences and sanitizing data.")

    # Align all dataframes and convert to numpy
    main_features_np = df[feature_cols].values
    target_np = df['target_returns'].values
    indicator_banks_np = np.stack([ind_df.reindex(df.index).fillna(0).values for ind_df in indicator_dfs], axis=1)

    # Use sliding window to create sequences
    X_list, y_list, banks_list = [], [], []
    for i in range(len(df) - seq_len):
        X_list.append(main_features_np[i: i + seq_len])
        y_list.append(target_np[i + seq_len - 1])
        banks_list.append(indicator_banks_np[i + seq_len - 1])

    # Convert to sanitized tensors
    X = torch.from_numpy(np.nan_to_num(np.array(X_list))).float()
    y = torch.from_numpy(np.nan_to_num(np.array(y_list))).float()
    indicator_banks = torch.from_numpy(np.nan_to_num(np.array(banks_list))).float()

    return X, y, indicator_banks


# --- LSTM Model ---
class LSTMTradingAgent(nn.Module):
    def __init__(self, d_input, d_hidden, num_indicators, num_windows, dropout_prob=0.3, tau=1.0):
        super().__init__()
        self.num_indicators = num_indicators
        self.tau = tau

        self.lstm = nn.LSTM(d_input, d_hidden, num_layers=4, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        # Heads for generating dynamic parameters
        self.window_heads = nn.ModuleList([nn.Linear(d_hidden, num_windows) for _ in range(num_indicators)])
        self.threshold_heads = nn.ModuleList(
            [nn.Linear(d_hidden, 2) for _ in range(num_indicators)])  # (theta_plus, theta_minus)

        self.beta = nn.Parameter(torch.randn(num_indicators))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, indicator_bank, return_params=False):
        _, (h_n, _) = self.lstm(x)
        h_t = self.dropout(h_n[-1])  # Use hidden state from the last layer

        signals = []
        all_weights, all_thresholds, all_indicator_values = [], [], []

        for i in range(self.num_indicators):
            # 1. Select window weights using Gumbel-Softmax trick for stochastic selection
            alpha_logits = self.window_heads[i](h_t)
            weights = F.softmax((alpha_logits - torch.log(-torch.log(torch.rand_like(alpha_logits)))) / self.tau,
                                dim=-1)

            # 2. Compute soft indicator value
            indicator_value = (weights * indicator_bank[:, i, :]).sum(dim=-1)

            # 3. Generate dynamic thresholds
            thresholds = self.threshold_heads[i](h_t)
            theta_plus, theta_minus = thresholds[:, 0], thresholds[:, 1]

            # 4. Generate indicator signal using soft thresholding
            s_i = torch.sigmoid((indicator_value - theta_plus) * 10) - torch.sigmoid(
                (theta_minus - indicator_value) * 10)
            signals.append(s_i)

            if return_params:
                all_weights.append(weights)
                all_thresholds.append(thresholds)
                all_indicator_values.append(indicator_value)

        # Combine signals into a final trading decision
        S = torch.stack(signals, dim=-1)
        decision = torch.tanh(S @ self.beta + self.bias)

        if return_params:
            return decision, torch.stack(all_weights, 1), torch.stack(all_thresholds, 1), torch.stack(
                all_indicator_values, 1)
        return decision


# --- Training and Evaluation ---
def sharpe_loss(returns, epsilon=1e-9):
    """Negative Sharpe Ratio, to be minimized."""
    return -(returns.mean() / (returns.std() + epsilon))


def get_sharpe(model, loader, device, num_indicators):
    """Calculates the Sharpe ratio for a given dataset."""
    model.eval()
    all_returns = []
    with torch.no_grad():
        for x, banks, y in loader:
            x, banks, y = x.to(device), banks.to(device), y.to(device)
            decisions = model(x, banks)
            returns = decisions.flatten() * y.flatten()
            all_returns.append(returns)
    all_returns = torch.cat(all_returns)
    return -sharpe_loss(all_returns).item()  # Return positive sharpe


def train_model(model, train_loader, val_loader, device, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    history = {'train_loss': [], 'train_sharpe': [], 'val_sharpe': []}

    # --- Start of Changes ---
    # 1. Initialize variables to track the best model
    best_val_sharpe = -np.inf
    best_model_state = None
    # --- End of Changes ---

    print("\n🚀 Starting model training...")

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x_batch, banks_batch, y_batch in train_loader:
            x_batch, banks_batch, y_batch = x_batch.to(device), banks_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            decisions = model(x_batch, banks_batch)
            returns = decisions.flatten() * y_batch.flatten()
            loss = sharpe_loss(returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        train_sharpe = get_sharpe(model, train_loader, device, model.num_indicators)
        val_sharpe = get_sharpe(model, val_loader, device, model.num_indicators)
        history.update({'train_loss': history['train_loss'] + [epoch_loss],
                        'train_sharpe': history['train_sharpe'] + [train_sharpe],
                        'val_sharpe': history['val_sharpe'] + [val_sharpe]})

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Train Sharpe: {train_sharpe:.4f} | Val Sharpe: {val_sharpe:.4f}")

        # --- Start of Changes ---
        # 2. Check if the current model is the best one and save its state if it is
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_model_state = model.state_dict().copy()  # Use .copy() to save a snapshot
            print(f"✨ New best model found at epoch {epoch + 1} with Val Sharpe: {val_sharpe:.4f}")
        # --- End of Changes ---

    # --- Start of Changes ---
    # 3. After the loop, load the best model state back into the model
    if best_model_state:
        print(f"\n✅ Loading best model with Val Sharpe: {best_val_sharpe:.4f}")
        model.load_state_dict(best_model_state)
    # --- End of Changes ---

    print("\n✅ Training complete.")
    final_sharpe = get_sharpe(model, val_loader, device, model.num_indicators) * np.sqrt(252)
    print(f"Final Annualized Validation Sharpe Ratio: {final_sharpe:.4f}")

    return history


# --- Visualization ---
def get_dynamic_parameters(model, loader, device):
    """Extracts dynamic parameters (weights, thresholds) from the model for visualization."""
    model.eval()
    all_w, all_theta, all_I_hat = [], [], []
    with torch.no_grad():
        for x, banks, y in loader:
            x, banks = x.to(device), banks.to(device)
            _, weights, thresholds, ind_values = model(x, banks, return_params=True)
            all_w.append(weights.cpu().numpy())
            all_theta.append(thresholds.cpu().numpy())
            all_I_hat.append(ind_values.cpu().numpy())
    return np.concatenate(all_w), np.concatenate(all_theta), np.concatenate(all_I_hat)


def plot_dynamic_parameters(dates, weights, thresholds, ind_values, indicator_names, window_sizes, idx):
    """Plots the model's dynamic weights and thresholds for a specific indicator."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, facecolor='#1E1E1E')
    plt.style.use('dark_background')

    # Plot 1: Dynamic Window Weights
    indicator_weights = weights[:, idx, :]
    for i, s in enumerate(window_sizes):
        axes[0].plot(dates, indicator_weights[:, i], label=f'Weight for Window {s}', alpha=0.9)
    axes[0].set_title(f'Dynamic Window Weights for {indicator_names[idx]}', fontsize=14)
    axes[0].set_ylabel('Attention Weight');
    axes[0].legend(loc='upper left');
    axes[0].grid(True, alpha=0.2);
    axes[0].set_ylim(0, 1)

    # Plot 2: Dynamic Thresholds vs. Indicator Value
    axes[1].plot(dates, ind_values[:, idx], label='Soft Indicator Value', color='cyan', lw=2)
    axes[1].plot(dates, thresholds[:, idx, 0], label='Upper Threshold (Buy)', color='lime', ls='--')
    axes[1].plot(dates, thresholds[:, idx, 1], label='Lower Threshold (Sell)', color='red', ls='--')
    axes[1].fill_between(dates, thresholds[:, idx, 1], thresholds[:, idx, 0], color='gray', alpha=0.2,
                         label='Neutral Zone')
    axes[1].set_title(f'Dynamic Thresholds for {indicator_names[idx]}', fontsize=14)
    axes[1].set_ylabel('Value');
    axes[1].set_xlabel('Date');
    axes[1].legend(loc='upper left');
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout();
    plt.show()


def visualize_detailed_backtest(model, test_loader, test_df, test_dates, symbol, device):
    """Runs a detailed, event-driven backtest and plots the performance."""
    model.eval()
    all_decisions = []
    with torch.no_grad():
        for x, banks, y in test_loader:
            decisions = model(x.to(device), banks.to(device))
            all_decisions.extend(decisions.cpu().numpy().flatten())

    decisions_series = pd.Series(all_decisions, index=test_dates)

    # --- Detailed Portfolio Simulation Logic (Restored) ---
    initial_capital = 100000.0
    capital = initial_capital
    position = 0  # 0: out, 1: long, -1: short
    shares_held = 0
    portfolio_values = []
    trade_log = []

    L_short = 1.2
    trade_threshold = 0.7
    transaction_fee = 5.0
    price_col = f'{symbol}_close_orig'  # Use original, un-normalized price

    for date, row in test_df.iterrows():
        current_price = row[price_col]
        decision = decisions_series.get(date)

        if decision is not None:
            # Close existing position if signal flips or weakens
            if (position == 1 and decision < trade_threshold) or \
                    (position == -1 and decision > -trade_threshold):
                capital += (shares_held * current_price) - transaction_fee
                log_action = 'close_long' if position == 1 else 'close_short'
                trade_log.append({'date': date, 'action': log_action, 'price': current_price})
                position, shares_held = 0, 0

            # Open new position
            if position == 0:
                if decision > trade_threshold:  # Go Long
                    shares_to_buy = (capital - transaction_fee) // current_price
                    if shares_to_buy > 0:
                        capital -= (shares_to_buy * current_price) + transaction_fee
                        shares_held, position = shares_to_buy, 1
                        trade_log.append({'date': date, 'action': 'open_long', 'price': current_price})
                elif decision < -trade_threshold:  # Go Short (simple version)
                    shares_to_short = (capital - transaction_fee) // current_price * L_short
                    if shares_to_short > 0:
                        capital -= (shares_to_short * current_price) + transaction_fee
                        shares_held, position = shares_to_short, -1
                        trade_log.append({'date': date, 'action': 'open_short', 'price': current_price})

        # Calculate current portfolio value
        current_position_value = shares_held * current_price if position != 0 else 0
        portfolio_values.append(capital + current_position_value)

    portfolio_series = pd.Series(portfolio_values, index=test_df.index)

    # --- Performance Metrics & Plotting ---
    returns = portfolio_series.pct_change().dropna()
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if returns.std() > 0 else 0
    cum_return = (portfolio_series.iloc[-1] / initial_capital - 1) * 100

    print(f"\n--- Detailed Backtest Performance: {symbol} ---")
    print(f"Final Portfolio Value: ${portfolio_series.iloc[-1]:,.2f}")
    print(f"Cumulative Return: {cum_return:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
    print(f"Total Trades: {len([t for t in trade_log if 'open' in t['action']])}")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot 1: Portfolio Value
    axes[0].plot(portfolio_series.index, portfolio_series, label='Portfolio Value', color='cyan', lw=2)
    axes[0].set_title(f'Detailed Backtest Performance - {symbol}', fontsize=14)
    axes[0].set_ylabel('Portfolio Value ($)');
    axes[0].grid(True, alpha=0.2);
    axes[0].legend()

    # Plot 2: Stock Price with Trade Markers
    axes[1].plot(test_df.index, test_df[price_col], label=f'{symbol} Price', color='white', alpha=0.9)
    for trade in trade_log:
        marker, color = ('^', 'lime') if 'long' in trade['action'] else ('v', 'red')
        if 'close' in trade['action']: marker, color = ('o', 'yellow')
        axes[1].scatter(trade['date'], trade['price'], color=color, s=80, marker=marker, alpha=0.9, zorder=5)
    axes[1].set_title('Stock Price with Trade Signals');
    axes[1].set_ylabel('Price ($)');
    axes[1].grid(True, alpha=0.2);
    axes[1].legend()

    # Plot 3: Decision Signals
    axes[2].plot(decisions_series.index, decisions_series, label='LSTM Decision', color='orange')
    axes[2].axhline(y=trade_threshold, color='lime', ls=':', alpha=0.7)
    axes[2].axhline(y=-trade_threshold, color='red', ls=':', alpha=0.7)
    axes[2].fill_between(decisions_series.index, -trade_threshold, trade_threshold, alpha=0.15, color='gray')
    axes[2].set_title('LSTM Decision Signals');
    axes[2].set_ylabel('Decision Value');
    axes[2].grid(True, alpha=0.2);
    axes[2].legend()

    plt.tight_layout();
    plt.show()


# --- Main Execution ---
def run_trading_system(symbol="MSFT"):
    """Main function to run the entire data collection, training, and analysis pipeline."""
    print(f"=== 📈 Running Trading System for {symbol} ===")
    device = get_execution_device()

    # 1. Data Collection & Preparation
    data = collect_stock_data(symbol)
    df, feature_cols = prepare_dataframe(data['stock_data'], data['earnings_data'], symbol)
    # Keep original price for realistic backtesting
    df[f'{symbol}_close_orig'] = data['stock_data']['close']

    # 2. Feature & Sequence Creation
    window_sizes = [5, 10, 20, 50, 100]
    indicator_dfs = create_indicator_banks(df, symbol, window_sizes)
    X, y, indicator_banks = create_sequences(df, indicator_dfs, feature_cols)

    # 3. Data Splitting & Loading
    train_size = int(len(X) * 0.8)
    X_train, X_val, y_train, y_val, banks_train, banks_val = \
        X[:train_size], X[train_size:], y[:train_size], y[train_size:], \
            indicator_banks[:train_size], indicator_banks[train_size:]

    train_loader = DataLoader(TensorDataset(X_train, banks_train, y_train), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, banks_val, y_val), batch_size=512)
    print(f"Data prepared. Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # 4. Model Training
    model = LSTMTradingAgent(len(feature_cols), 32, len(indicator_dfs), len(window_sizes)).to(device)
    history = train_model(model, train_loader, val_loader, device, epochs=500)

    # 5. Analysis & Visualization
    print("\n---  Analysis and Visualization ---")
    val_dates = df.index[-len(X_val):]
    original_val_df = df.loc[val_dates]

    # Visualize dynamic parameters
    indicator_names = ["SMA", "EMA", "RSI", "BB Position", "OBV", "MACD", "P/E Ratio"]
    weights, thresholds, ind_values = get_dynamic_parameters(model, val_loader, device)
    for i in range(len(indicator_names)):
        plot_dynamic_parameters(val_dates, weights, thresholds, ind_values, indicator_names, window_sizes, i)

    # Visualize backtest performance using the detailed, realistic simulation
    visualize_detailed_backtest(model, val_loader, original_val_df, val_dates, symbol, device)

    print(f"\n===  {symbol} System Run Complete ===")
    return {'model': model, 'history': history}


if __name__ == "__main__":
    results = run_trading_system(symbol="MSFT")