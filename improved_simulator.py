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


def generate_all_signals(indicators, params, current_idx, previous_idx=None):
    """
    Generate signals for all indicators based on assignment specification.

    Args:
        indicators: Dictionary of calculated indicators
        params: Parameters including thresholds
        current_idx: Current time index
        previous_idx: Previous time index (for crossover signals)

    Returns:
        List of signals for aggregation
    """
    signals = []

    # 1. SMA Crossover Signal
    if previous_idx is not None:
        sma_short_curr = indicators['sma_short'].iloc[current_idx]
        sma_long_curr = indicators['sma_long'].iloc[current_idx]
        sma_short_prev = indicators['sma_short'].iloc[previous_idx]
        sma_long_prev = indicators['sma_long'].iloc[previous_idx]

        # Golden Cross (short > long) and Death Cross (short < long)
        if (sma_short_curr > sma_long_curr) and (sma_short_prev <= sma_long_prev):
            signals.append(1)  # Golden cross - long signal
        elif (sma_short_curr < sma_long_curr) and (sma_short_prev >= sma_long_prev):
            signals.append(-1)  # Death cross - short signal
        else:
            signals.append(0)

    # 2. EMA Crossover Signal
    if previous_idx is not None:
        ema_short_curr = indicators['ema_short'].iloc[current_idx]
        ema_long_curr = indicators['ema_long'].iloc[current_idx]
        ema_short_prev = indicators['ema_short'].iloc[previous_idx]
        ema_long_prev = indicators['ema_long'].iloc[previous_idx]

        if (ema_short_curr > ema_long_curr) and (ema_short_prev <= ema_long_prev):
            signals.append(1)
        elif (ema_short_curr < ema_long_curr) and (ema_short_prev >= ema_long_prev):
            signals.append(-1)
        else:
            signals.append(0)

    # 3. RSI Signal with thresholds
    rsi_signal = generate_signal(
        indicators['rsi'].iloc[current_idx],
        params['rsi_theta_plus'],
        params['rsi_theta_minus']
    )
    signals.append(rsi_signal)

    # 4. MACD Crossover Signal
    if previous_idx is not None:
        macd_curr = indicators['macd_line'].iloc[current_idx]
        signal_curr = indicators['macd_signal'].iloc[current_idx]
        macd_prev = indicators['macd_line'].iloc[previous_idx]
        signal_prev = indicators['macd_signal'].iloc[previous_idx]

        if (macd_curr > signal_curr) and (macd_prev <= signal_prev):
            signals.append(1)
        elif (macd_curr < signal_curr) and (macd_prev >= signal_prev):
            signals.append(-1)
        else:
            signals.append(0)

    # 5. Bollinger Bands Signal
    price_curr = indicators['bb_middle'].iloc[current_idx] if 'bb_middle' in indicators else 0
    bb_upper = indicators['bb_upper'].iloc[current_idx]
    bb_lower = indicators['bb_lower'].iloc[current_idx]

    if price_curr < bb_lower:
        signals.append(1)  # Price below lower band - potential reversal up
    elif price_curr > bb_upper:
        signals.append(-1)  # Price above upper band - potential reversal down
    else:
        signals.append(0)

    # 6. OBV Signal
    if previous_idx is not None:
        obv_curr = indicators['obv'].iloc[current_idx]
        obv_sma_curr = indicators['obv_sma'].iloc[current_idx]
        obv_prev = indicators['obv'].iloc[previous_idx]
        obv_sma_prev = indicators['obv_sma'].iloc[previous_idx]

        if (obv_curr > obv_sma_curr) and (obv_prev <= obv_sma_prev):
            signals.append(1)
        elif (obv_curr < obv_sma_curr) and (obv_prev >= obv_sma_prev):
            signals.append(-1)
        else:
            signals.append(0)

    # 7. P/E Ratio Signal (if available)
    if 'pe_ratio' in indicators:
        pe_signal = generate_signal(
            indicators['pe_ratio'].iloc[current_idx],
            params.get('pe_theta_plus', 25),  # High P/E threshold
            params.get('pe_theta_minus', 10)  # Low P/E threshold
        )
        # Invert P/E signal (low P/E = undervalued = long signal)
        signals.append(-pe_signal)

    # 8. Earnings Surprise Signal (if available)
    if 'earnings_surprise' in indicators:
        surprise_signal = generate_signal(
            indicators['earnings_surprise'].iloc[current_idx],
            params.get('surprise_theta_plus', 5),  # Positive surprise threshold
            params.get('surprise_theta_minus', -5)  # Negative surprise threshold
        )
        signals.append(surprise_signal)

    return signals


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

        # Execute trading logic
        if final_decision > 0:  # Long signal
            if position < 0:  # Close short position
                capital += position * current_price - transaction_fee
                trade_log.append({
                    'date': data.index[i],
                    'action': 'close_short',
                    'shares': -position,
                    'price': current_price,
                    'capital': capital
                })
                position = 0

            if position == 0:  # Open long position
                max_shares = calculate_position_size(capital, current_price, transaction_fee, 'long')
                if max_shares > 0:
                    position = max_shares
                    capital -= position * current_price + transaction_fee
                    trade_log.append({
                        'date': data.index[i],
                        'action': 'open_long',
                        'shares': position,
                        'price': current_price,
                        'capital': capital
                    })

        elif final_decision < 0:  # Short signal
            if position > 0:  # Close long position
                capital += position * current_price - transaction_fee
                trade_log.append({
                    'date': data.index[i],
                    'action': 'close_long',
                    'shares': position,
                    'price': current_price,
                    'capital': capital
                })
                position = 0

            if position == 0:  # Open short position
                max_shares = calculate_position_size(capital, current_price, transaction_fee, 'short', lambda_worst)
                if max_shares > 0:
                    position = -max_shares
                    capital -= position * current_price - transaction_fee  # Note: position is negative
                    trade_log.append({
                        'date': data.index[i],
                        'action': 'open_short',
                        'shares': -position,
                        'price': current_price,
                        'capital': capital
                    })

        else:  # Neutral signal - close any open position
            if position != 0:
                action = 'close_long' if position > 0 else 'close_short'
                capital += position * current_price - transaction_fee
                trade_log.append({
                    'date': data.index[i],
                    'action': action,
                    'shares': abs(position),
                    'price': current_price,
                    'capital': capital
                })
                position = 0

        # Update portfolio value
        portfolio_values.iloc[i] = capital + (position * current_price)

    # Calculate performance metrics
    returns = portfolio_values.pct_change().dropna()

    if returns.std() == 0 or len(returns) == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    return sharpe_ratio, portfolio_values, trade_log