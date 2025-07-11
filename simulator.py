import pandas as pd
import numpy as np
from indicators import calculate_sma, calculate_rsi


def run_simulation(data, params):
    """
    Runs a trading simulation for a given set of parameters based on the
    assignment's signal generation and aggregation rules.

    Args:
        data (pd.DataFrame): DataFrame with price data, e.g., 'MSFT_close'.
        params (dict): A dictionary of parameters for the indicators.

    Returns:
        float: The annualized Sharpe ratio for the simulation.
        pd.Series: The portfolio value over time.
    """
    print("Simulator V2:")
    # --- Parameters ---
    initial_capital = 100000.0
    transaction_fee = 5.0
    price_col = 'MSFT_close'

    # --- Indicator Calculation ---
    sma_short = calculate_sma(data[price_col], window=params['sma_short'])
    sma_long = calculate_sma(data[price_col], window=params['sma_long'])
    rsi = calculate_rsi(data[price_col], window=params['rsi_window'])

    # --- Simulation Setup ---
    capital = initial_capital
    position = 0  # Can be positive (long), negative (short), or zero.
    portfolio_values = pd.Series(index=data.index, dtype=float)
    portfolio_values.iloc[0] = initial_capital

    # --- Simulation Loop ---
    for i in range(1, len(data)):
        current_price = data[price_col].iloc[i]

        # --- 1. Generate Individual Signals ---
        signals = []

        # SMA Crossover Signal (Event-based)
        if (sma_short.iloc[i] > sma_long.iloc[i]) and (sma_short.iloc[i - 1] <= sma_long.iloc[i - 1]):
            signals.append(1)  # Golden Cross
        elif (sma_short.iloc[i] < sma_long.iloc[i]) and (sma_short.iloc[i - 1] >= sma_long.iloc[i - 1]):
            signals.append(-1)  # Death Cross

        # RSI Signal (State-based)
        if rsi.iloc[i] < params['rsi_buy']:
            signals.append(1)  # Oversold
        elif rsi.iloc[i] > params['rsi_sell']:
            signals.append(-1)  # Overbought

        # --- 2. Aggregate Signals (Majority Vote) ---
        final_decision = np.sign(np.sum(signals))  # +1 for Buy, -1 for Sell, 0 for Hold/Close

        # --- 3. Execution Logic ---
        # Case 1: Signal is to GO LONG
        if final_decision > 0:
            if position < 0:  # If short, close position first
                capital += position * current_price - transaction_fee
                position = 0
            if position == 0:  # If no position, open long
                shares_to_buy = (capital - transaction_fee) // current_price
                if shares_to_buy > 0:
                    position = shares_to_buy
                    capital -= position * current_price + transaction_fee

        # Case 2: Signal is to GO SHORT
        elif final_decision < 0:
            if position > 0:  # If long, close position first
                capital += position * current_price - transaction_fee
                position = 0
            if position == 0:  # If no position, open short
                shares_to_sell = (capital - transaction_fee) // current_price
                if shares_to_sell > 0:
                    position = -shares_to_sell  # Negative position for short
                    capital -= position * current_price - transaction_fee  # capital increases

        # Case 3: Signal is NEUTRAL (Close any open position)
        elif final_decision == 0:
            if position != 0:
                capital += position * current_price - transaction_fee
                position = 0

        # Update portfolio value for the day
        portfolio_values.iloc[i] = capital + (position * current_price)

    # --- Performance Calculation ---
    returns = portfolio_values.pct_change().dropna()

    if returns.std() == 0 or len(returns) == 0:
        return 0.0, portfolio_values

    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    return sharpe_ratio, portfolio_values
