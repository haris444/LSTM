# improved_simulator.py
import pandas as pd
import numpy as np
from indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands, \
    calculate_obv
from fundamental_indicators import add_fundamental_indicators_to_data


def generate_signal(indicator_value, theta_plus, theta_minus):
    """
    Generate trading signal based on assignment specification.

    Args:
        indicator_value: Current value of the indicator
        theta_plus: Upper threshold for long signal
        theta_minus: Lower threshold for short signal

    Returns:
        Signal in {-1, 0, +1}
    """
    if pd.isna(indicator_value):
        return 0

    if indicator_value > theta_plus:
        return 1  # Long signal
    elif indicator_value < theta_minus:
        return -1  # Short signal
    else:
        return 0  # Neutral signal


def calculate_position_size(capital, price, transaction_fee, position_type='long', lambda_worst=1.5):
    """
    Calculate maximum position size based on assignment constraints.

    Args:
        capital: Available capital
        price: Current stock price
        transaction_fee: Fixed transaction cost
        position_type: 'long' or 'short'
        lambda_worst: Worst-case price multiplier for short positions

    Returns:
        Maximum number of shares that can be traded
    """
    if position_type == 'long':
        # Long position constraint: q <= (C(t0) - f) / P(t0)
        max_shares = int((capital - transaction_fee) / price)
    else:  # short position
        # Short position constraint: q <= (C(t0) - f) / (lambda * P(t0))
        worst_case_price = lambda_worst * price
        max_shares = int((capital - transaction_fee) / worst_case_price)

    return max(0, max_shares)


def calculate_all_indicators(data, params):
    """
    Calculate all technical and fundamental indicators with given parameters.

    Args:
        data: DataFrame with price and volume data
        params: Dictionary of parameters for indicators

    Returns:
        Dictionary of calculated indicators
    """
    price_col = params.get('price_column', 'MSFT_close')
    volume_col = params.get('volume_column', 'MSFT_volume')

    indicators = {}

    # Technical Indicators
    # 1. Moving Averages
    indicators['sma_short'] = calculate_sma(data[price_col], params['sma_short_window'])
    indicators['sma_long'] = calculate_sma(data[price_col], params['sma_long_window'])
    indicators['ema_short'] = calculate_ema(data[price_col], params['ema_short_window'])
    indicators['ema_long'] = calculate_ema(data[price_col], params['ema_long_window'])

    # 2. RSI
    indicators['rsi'] = calculate_rsi(data[price_col], params['rsi_window'])

    # 3. MACD
    macd_line, signal_line = calculate_macd(
        data[price_col],
        window_slow=params['macd_slow'],
        window_fast=params['macd_fast'],
        window_signal=params['macd_signal']
    )
    indicators['macd_line'] = macd_line
    indicators['macd_signal'] = signal_line

    # 4. Bollinger Bands
    upper_band, lower_band = calculate_bollinger_bands(
        data[price_col],
        window=params['bb_window'],
        num_std_dev=params['bb_std_dev']
    )
    indicators['bb_upper'] = upper_band
    indicators['bb_lower'] = lower_band
    indicators['bb_middle'] = calculate_sma(data[price_col], params['bb_window'])

    # 5. OBV
    indicators['obv'] = calculate_obv(data[price_col], data[volume_col])
    indicators['obv_sma'] = calculate_sma(indicators['obv'], params['obv_window'])

    # Fundamental Indicators (if available in data)
    if 'PE_Ratio' in data.columns:
        indicators['pe_ratio'] = data['PE_Ratio']
    if 'Earnings_Surprise' in data.columns:
        indicators['earnings_surprise'] = data['Earnings_Surprise']

    return indicators


# In improved_simulator.py

def generate_all_signals(indicators, params, current_idx, previous_idx=None):
    """
    Generate signals for all indicators based on assignment specification.
    Can also be used to isolate a single indicator's signal for debugging.
    """
    all_signals = []

    # Calculate all potential signals
    # 1. SMA
    sma_diff = indicators['sma_short'].iloc[current_idx] - indicators['sma_long'].iloc[current_idx]
    all_signals.append(generate_signal(sma_diff, params['sma_theta_plus'], params['sma_theta_minus']))
    # 2. EMA
    ema_diff = indicators['ema_short'].iloc[current_idx] - indicators['ema_long'].iloc[current_idx]
    all_signals.append(generate_signal(ema_diff, params['ema_theta_plus'], params['ema_theta_minus']))
    # 3. RSI
    all_signals.append(
        generate_signal(indicators['rsi'].iloc[current_idx], params['rsi_theta_plus'], params['rsi_theta_minus']))
    # 4. MACD
    macd_diff = indicators['macd_line'].iloc[current_idx] - indicators['macd_signal'].iloc[current_idx]
    all_signals.append(generate_signal(macd_diff, params['macd_theta_plus'], params['macd_theta_minus']))
    # 5. Bollinger Bands
    price_curr = indicators['bb_middle'].iloc[current_idx]
    if price_curr < indicators['bb_lower'].iloc[current_idx]:
        all_signals.append(1)
    elif price_curr > indicators['bb_upper'].iloc[current_idx]:
        all_signals.append(-1)
    else:
        all_signals.append(0)
    # 6. OBV
    obv_diff = indicators['obv'].iloc[current_idx] - indicators['obv_sma'].iloc[current_idx]
    all_signals.append(generate_signal(obv_diff, params['obv_theta_plus'], params['obv_theta_minus']))
    # 7. P/E Ratio
    pe_signal = generate_signal(indicators.get('pe_ratio', pd.Series(0)).iloc[current_idx], params['pe_theta_plus'],
                                params['pe_theta_minus'])
    all_signals.append(-pe_signal)  # Inverted signal
    # 8. Earnings Surprise
    surprise = indicators.get('earnings_surprise', pd.Series(0)).iloc[current_idx]
    all_signals.append(generate_signal(surprise, params['surprise_theta_plus'],
                                       params['surprise_theta_minus']) if surprise != 0 else 0)

    # --- Isolation Logic for Debugging ---
    if 'use_single_indicator' in params:
        indicator_index = params['use_single_indicator']
        isolated_signals = [0] * len(all_signals)
        isolated_signals[indicator_index] = all_signals[indicator_index]
        return isolated_signals

    return all_signals


def aggregate_signals(signals, method='majority_vote', weights=None):
    """
    Aggregate individual indicator signals into a final decision.

    Args:
        signals: List of individual signals
        method: Aggregation method ('majority_vote', 'weighted_sum')
        weights: Weights for weighted aggregation

    Returns:
        Final trading decision {-1, 0, 1}
    """
    if not signals:
        return 0

    if method == 'majority_vote':
        return int(np.sign(np.sum(signals)))
    elif method == 'weighted_sum':
        if weights is None:
            weights = [1.0] * len(signals)
        weighted_sum = np.sum([s * w for s, w in zip(signals, weights)])
        return int(np.sign(weighted_sum))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def run_improved_simulation(data, params):
    """
    Run trading simulation with proper signal generation per assignment.

    Args:
        data: DataFrame with price and volume data
        params: Dictionary of all parameters

    Returns:
        tuple: (sharpe_ratio, portfolio_values, trade_log)
    """
    # Add fundamental indicators if not present
    if 'PE_Ratio' not in data.columns:
        data = add_fundamental_indicators_to_data(data, params.get('price_column', 'MSFT_close'))

    # Calculate all indicators
    indicators = calculate_all_indicators(data, params)

    # Simulation setup
    initial_capital = params.get('initial_capital', 100000.0)
    transaction_fee = params.get('transaction_fee', 5.0)
    price_col = params.get('price_column', 'MSFT_close')
    lambda_worst = params.get('lambda_worst', 1.5)

    capital = initial_capital
    position = 0  # Number of shares held (positive = long, negative = short)
    portfolio_values = pd.Series(index=data.index, dtype=float)
    portfolio_values.iloc[0] = initial_capital

    trade_log = []  # Track all trades for analysis

    # Simulation loop
    for i in range(1, len(data)):
        current_price = data[price_col].iloc[i]
        previous_idx = i - 1

        # Generate signals for all indicators
        signals = generate_all_signals(indicators, params, i, previous_idx)

        # Aggregate signals into final decision
        final_decision = aggregate_signals(
            signals,
            method=params.get('aggregation_method', 'majority_vote'),
            weights=params.get('signal_weights', None)
        )

        # Execute trading logic based on assignment rules
        # Close position if signal is neutral or opposite
        if position != 0 and (final_decision == 0 or final_decision == -np.sign(position)):
            action = 'close_long' if position > 0 else 'close_short'
            capital += position * current_price - transaction_fee
            trade_log.append({
                'date': data.index[i], 'action': action,
                'shares': abs(position), 'price': current_price, 'capital': capital
            })
            position = 0

        # Open a new position if there's a signal and no current position
        if position == 0 and final_decision != 0:
            if final_decision > 0:  # Open long
                max_shares = calculate_position_size(capital, current_price, transaction_fee, 'long')
                if max_shares > 0:
                    position = max_shares
                    capital -= position * current_price + transaction_fee
                    trade_log.append({
                        'date': data.index[i], 'action': 'open_long',
                        'shares': position, 'price': current_price, 'capital': capital
                    })
            else:  # Open short
                max_shares = calculate_position_size(capital, current_price, transaction_fee, 'short', lambda_worst)
                if max_shares > 0:
                    position = -max_shares
                    capital -= position * current_price - transaction_fee  # Position is negative
                    trade_log.append({
                        'date': data.index[i], 'action': 'open_short',
                        'shares': -position, 'price': current_price, 'capital': capital
                    })

        # Update portfolio value
        portfolio_values.iloc[i] = capital + (position * current_price)

    # Calculate performance metrics
    returns = portfolio_values.pct_change().dropna()

    if returns.std() == 0 or len(returns) == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    return sharpe_ratio, portfolio_values, trade_log