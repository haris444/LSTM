import pandas as pd
import numpy as np
from indicators import calculate_sma, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_obv


def run_simulation(data, params):
    """
    Runs a trading simulation for a given set of parameters based on the
    assignment's signal generation and aggregation rules.

    Args:
        data (pd.DataFrame): DataFrame with price and volume data.
        params (dict): A dictionary of parameters for the indicators.

    Returns:
        float: The annualized Sharpe ratio for the simulation.
        pd.Series: The portfolio value over time.
    """
    # --- Parameters ---
    initial_capital = 100000.0
    transaction_fee = 5.0
    price_col = 'MSFT_close'
    volume_col = 'MSFT_volume'
    lambda_worst = params.get('lambda_worst', 1.5)  # For short position constraint

    # --- Indicator Calculation ---
    sma_short = calculate_sma(data[price_col], window=params['sma_short'])
    sma_long = calculate_sma(data[price_col], window=params['sma_long'])
    rsi = calculate_rsi(data[price_col], window=params['rsi_window'])
    macd_line, signal_line = calculate_macd(data[price_col], window_slow=params['macd_slow'],
                                            window_fast=params['macd_fast'], window_signal=params['macd_signal'])
    upper_band, lower_band = calculate_bollinger_bands(data[price_col], window=params['bb_window'],
                                                       num_std_dev=params['bb_std_dev'])
    obv = calculate_obv(data[price_col], data[volume_col])
    obv_sma = calculate_sma(obv, window=params['obv_window'])  # Use a smoothed OBV for signals

    # --- Simulation Setup ---
    capital = initial_capital
    position = 0
    portfolio_values = pd.Series(index=data.index, dtype=float)
    portfolio_values.iloc[0] = initial_capital

    # --- Simulation Loop ---
    for i in range(1, len(data)):
        current_price = data[price_col].iloc[i]

        # --- 1. Generate Individual Signals ---
        signals = []

        # SMA Crossover Signal
        if (sma_short.iloc[i] > sma_long.iloc[i]) and (sma_short.iloc[i - 1] <= sma_long.iloc[i - 1]):
            signals.append(1)
        elif (sma_short.iloc[i] < sma_long.iloc[i]) and (sma_short.iloc[i - 1] >= sma_long.iloc[i - 1]):
            signals.append(-1)

        # RSI Signal
        if rsi.iloc[i] < params['rsi_buy']:
            signals.append(1)
        elif rsi.iloc[i] > params['rsi_sell']:
            signals.append(-1)

        # MACD Crossover Signal
        if (macd_line.iloc[i] > signal_line.iloc[i]) and (macd_line.iloc[i - 1] <= signal_line.iloc[i - 1]):
            signals.append(1)
        elif (macd_line.iloc[i] < signal_line.iloc[i]) and (macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]):
            signals.append(-1)

        # Bollinger Bands Signal
        if current_price < lower_band.iloc[i]:
            signals.append(1)  # Price broke below lower band (potential reversal)
        elif current_price > upper_band.iloc[i]:
            signals.append(-1)  # Price broke above upper band

        # OBV Signal
        if (obv.iloc[i] > obv_sma.iloc[i]) and (obv.iloc[i - 1] <= obv_sma.iloc[i - 1]):
            signals.append(1)  # OBV crosses above its moving average
        elif (obv.iloc[i] < obv_sma.iloc[i]) and (obv.iloc[i - 1] >= obv_sma.iloc[i - 1]):
            signals.append(-1)

        # --- 2. Aggregate Signals (Majority Vote) ---
        final_decision = np.sign(np.sum(signals))

        # --- 3. Execution Logic ---
        if final_decision > 0:  # GO LONG
            if position < 0:
                capital += position * current_price - transaction_fee
                position = 0
            if position == 0:
                shares_to_buy = (capital - transaction_fee) // current_price
                if shares_to_buy > 0:
                    position = shares_to_buy
                    capital -= position * current_price + transaction_fee

        elif final_decision < 0:  # GO SHORT
            if position > 0:
                capital += position * current_price - transaction_fee
                position = 0
            if position == 0:
                # Assignment's short position constraint
                p_worst = lambda_worst * current_price
                max_shares_short = (capital - transaction_fee) // p_worst
                if max_shares_short > 0:
                    position = -max_shares_short
                    capital -= position * current_price - transaction_fee  # Capital increases

        elif final_decision == 0:  # NEUTRAL
            if position != 0:
                capital += position * current_price - transaction_fee
                position = 0

        portfolio_values.iloc[i] = capital + (position * current_price)

    # --- Performance Calculation ---
    returns = portfolio_values.pct_change().dropna()

    if returns.std() == 0 or len(returns) == 0:
        return 0.0, portfolio_values

    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    return sharpe_ratio, portfolio_values