# improved_simulator.py
import pandas as pd
import numpy as np
from indicators import calculate_rsi, calculate_bollinger_bands, calculate_sma, calculate_obv
from fundamental_indicators import add_fundamental_indicators_to_data


def generate_signal(indicator_value, theta_plus, theta_minus):
    """
    Generate trading signal based on assignment specification.
    """
    if pd.isna(indicator_value):
        return 0
    if indicator_value > theta_plus:
        return 1
    elif indicator_value < theta_minus:
        return -1
    else:
        return 0


def calculate_position_size(capital, price, transaction_fee, position_type='long', lambda_worst=1.5):
    """
    Calculate maximum position size based on assignment constraints.
    """
    if position_type == 'long':
        max_shares = int((capital - transaction_fee) / price)
    else:  # short position
        worst_case_price = lambda_worst * price
        max_shares = int((capital - transaction_fee) / worst_case_price)
    return max(0, max_shares)


def calculate_trend_slope(series, window):
    """
    Calculate trend slope using linear regression over a rolling window.
    Returns the slope coefficient normalized by the mean value to get percentage slope.
    """
    if len(series) < window:
        return 0.0

    # Get the last 'window' values
    y_values = series.iloc[-window:].values
    x_values = np.arange(len(y_values))

    # Handle case where all values are the same
    if np.std(y_values) == 0:
        return 0.0

    # Simple linear regression: slope = covariance(x,y) / variance(x)
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)

    numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    # Normalize by mean to get percentage slope per period
    if y_mean != 0:
        normalized_slope = slope / abs(y_mean)
    else:
        normalized_slope = 0.0

    return normalized_slope


def calculate_obv_signal(obv_series, price_series, window, theta_plus, theta_minus, current_idx):
    """
    Calculate OBV trend confirmation signal based on assignment specification.

    Args:
        obv_series: On-Balance Volume time series
        price_series: Price time series
        window: Lookback window for trend calculation
        theta_plus: Minimum slope threshold for uptrend
        theta_minus: Maximum slope threshold for downtrend
        current_idx: Current index in the series

    Returns:
        Signal: +1 (buy), -1 (sell), 0 (hold)
    """
    if current_idx < window:
        return 0

    # Get recent data
    obv_recent = obv_series.iloc[current_idx - window + 1:current_idx + 1]
    price_recent = price_series.iloc[current_idx - window + 1:current_idx + 1]

    # Calculate trend slopes
    obv_slope = calculate_trend_slope(obv_recent, window)
    price_slope = calculate_trend_slope(price_recent, window)

    # Determine trend directions
    obv_trending_up = obv_slope > theta_plus
    obv_trending_down = obv_slope < theta_minus
    price_trending_up = price_slope > theta_plus
    price_trending_down = price_slope < theta_minus

    # Trend confirmation signals (OBV confirms price trend)
    if obv_trending_up and price_trending_up:
        return 1  # Both trending up - strong buy signal
    elif obv_trending_down and price_trending_down:
        return -1  # Both trending down - strong sell signal

    # Divergence signals (OBV contradicts price trend - potential reversal)
    elif obv_trending_up and price_trending_down:
        return 1  # OBV suggests underlying strength despite price decline
    elif obv_trending_down and price_trending_up:
        return -1  # OBV suggests underlying weakness despite price rise

    return 0  # No clear signal


def calculate_runtime_indicators(data, params):
    """
    Calculate indicators that don't depend on crossover logic.
    These are calculated at runtime during the simulation.
    """
    price_col = params.get('price_column', 'MSFT_close')
    volume_col = params.get('volume_column', 'MSFT_volume')

    indicators = {}
    indicators['rsi'] = calculate_rsi(data[price_col], params['rsi_window'])
    upper, lower = calculate_bollinger_bands(data[price_col], params['bb_window'], params['bb_std_dev'])
    indicators['bb_upper'] = upper
    indicators['bb_lower'] = lower
    indicators['bb_middle'] = calculate_sma(data[price_col], params['bb_window'])

    # Calculate OBV
    indicators['obv'] = calculate_obv(data[price_col], data[volume_col])
    indicators['price'] = data[price_col]  # Store price for OBV trend analysis

    # Add pre-calculated crossover states and fundamental indicators from the input data
    for col in ['sma_crossover', 'ema_crossover', 'macd_crossover', 'PE_Ratio', 'Earnings_Surprise']:
        if col in data.columns:
            indicators[col] = data[col]

    return indicators


def generate_all_signals(indicators, params, current_idx):
    """
    Generate signals for all indicators by reading their pre-computed states
    or calculating their values at runtime.
    """
    signals = []

    # Check if we're testing a single indicator
    use_single = params.get('use_single_indicator', None)

    if use_single is not None:
        # Only use the specified indicator
        if use_single == 0:  # SMA
            signals.append(generate_signal(indicators['sma_crossover'].iloc[current_idx],
                                           params['sma_theta_plus'], params['sma_theta_minus']))
        elif use_single == 1:  # EMA
            signals.append(generate_signal(indicators['ema_crossover'].iloc[current_idx],
                                           params['ema_theta_plus'], params['ema_theta_minus']))
        elif use_single == 2:  # RSI
            signals.append(generate_signal(indicators['rsi'].iloc[current_idx],
                                           params['rsi_theta_plus'], params['rsi_theta_minus']))
        elif use_single == 3:  # MACD
            signals.append(generate_signal(indicators['macd_crossover'].iloc[current_idx],
                                           params['macd_theta_plus'], params['macd_theta_minus']))
        elif use_single == 4:  # Bollinger Bands
            price_curr = indicators['bb_middle'].iloc[current_idx]
            if price_curr < indicators['bb_lower'].iloc[current_idx]:
                signals.append(1)
            elif price_curr > indicators['bb_upper'].iloc[current_idx]:
                signals.append(-1)
            else:
                signals.append(0)
        elif use_single == 5:  # OBV
            obv_signal = calculate_obv_signal(
                indicators['obv'], indicators['price'],
                params['obv_window'], params['obv_theta_plus'], params['obv_theta_minus'],
                current_idx
            )
            signals.append(obv_signal)
        elif use_single == 6:  # P/E Ratio
            if 'PE_Ratio' in indicators:
                pe_signal = generate_signal(indicators['PE_Ratio'].iloc[current_idx],
                                            params['pe_theta_plus'], params['pe_theta_minus'])
                signals.append(-pe_signal)  # Inverted: Low P/E is a buy signal
        elif use_single == 7:  # Earnings Surprise
            if 'Earnings_Surprise' in indicators:
                surprise = indicators['Earnings_Surprise'].iloc[current_idx]
                signals.append(generate_signal(surprise, params['surprise_theta_plus'],
                                               params['surprise_theta_minus']))
        return signals

    # Use all indicators (normal mode)
    # 1. SMA Crossover Signal
    signals.append(generate_signal(indicators['sma_crossover'].iloc[current_idx],
                                   params['sma_theta_plus'], params['sma_theta_minus']))

    # 2. EMA Crossover Signal
    signals.append(generate_signal(indicators['ema_crossover'].iloc[current_idx],
                                   params['ema_theta_plus'], params['ema_theta_minus']))

    # 3. RSI Signal
    signals.append(generate_signal(indicators['rsi'].iloc[current_idx],
                                   params['rsi_theta_plus'], params['rsi_theta_minus']))

    # 4. MACD Crossover Signal
    signals.append(generate_signal(indicators['macd_crossover'].iloc[current_idx],
                                   params['macd_theta_plus'], params['macd_theta_minus']))

    # 5. Bollinger Bands Signal
    price_curr = indicators['bb_middle'].iloc[current_idx]
    if price_curr < indicators['bb_lower'].iloc[current_idx]:
        signals.append(1)
    elif price_curr > indicators['bb_upper'].iloc[current_idx]:
        signals.append(-1)
    else:
        signals.append(0)

    # 6. OBV Trend Confirmation Signal
    obv_signal = calculate_obv_signal(
        indicators['obv'], indicators['price'],
        params['obv_window'], params['obv_theta_plus'], params['obv_theta_minus'],
        current_idx
    )
    signals.append(obv_signal)

    # 7. P/E Ratio Signal
    if 'PE_Ratio' in indicators:
        pe_signal = generate_signal(indicators['PE_Ratio'].iloc[current_idx],
                                    params['pe_theta_plus'], params['pe_theta_minus'])
        signals.append(-pe_signal)  # Inverted: Low P/E is a buy signal

    # 8. Earnings Surprise Signal
    if 'Earnings_Surprise' in indicators:
        surprise = indicators['Earnings_Surprise'].iloc[current_idx]
        signals.append(generate_signal(surprise, params['surprise_theta_plus'],
                                       params['surprise_theta_minus']))

    return signals


def aggregate_signals(signals, method='majority_vote', weights=None):
    """
    Aggregate individual indicator signals into a final decision.
    """
    if not signals:
        return 0
    if method == 'majority_vote':
        return int(np.sign(np.sum(signals)))
    elif method == 'weighted_sum':
        if weights and len(weights) >= len(signals):
            weighted_sum = np.sum([s * w for s, w in zip(signals, weights)])
            return int(np.sign(weighted_sum))
        else:
            return int(np.sign(np.sum(signals)))
    return 0


def run_improved_simulation(data, params):
    """
    Run trading simulation using a DataFrame that includes pre-computed crossover states.
    """
    # Calculate runtime indicators like RSI, Bollinger Bands, and OBV
    indicators = calculate_runtime_indicators(data, params)

    # Simulation setup
    initial_capital = params.get('initial_capital', 100000.0)
    transaction_fee = params.get('transaction_fee', 5.0)
    price_col = params.get('price_column', 'MSFT_close')

    capital = initial_capital
    position = 0
    portfolio_values = pd.Series(index=data.index, dtype=float)
    trade_log = []

    # Simulation loop
    for i in range(1, len(data)):
        current_price = data[price_col].iloc[i]
        portfolio_values.iloc[i - 1] = capital + (position * data[price_col].iloc[i - 1])

        signals = generate_all_signals(indicators, params, i)
        final_decision = aggregate_signals(signals, method=params.get('aggregation_method', 'majority_vote'),
                                           weights=params.get('signal_weights'))

        # Close position if signal is neutral or opposite
        if position != 0 and (final_decision == 0 or final_decision == -np.sign(position)):
            action = 'close_long' if position > 0 else 'close_short'
            capital += position * current_price - transaction_fee
            trade_log.append({'date': data.index[i], 'action': action, 'shares': abs(position), 'price': current_price})
            position = 0

        # Open a new position
        if position == 0 and final_decision != 0:
            if final_decision > 0:  # Open long
                max_shares = calculate_position_size(capital, current_price, transaction_fee, 'long')
                if max_shares > 0:
                    position = max_shares
                    capital -= position * current_price + transaction_fee
                    trade_log.append(
                        {'date': data.index[i], 'action': 'open_long', 'shares': position, 'price': current_price})
            else:  # Open short
                max_shares = calculate_position_size(capital, current_price, transaction_fee, 'short',
                                                     params['lambda_worst'])
                if max_shares > 0:
                    position = -max_shares
                    capital += abs(position) * current_price - transaction_fee
                    trade_log.append({'date': data.index[i], 'action': 'open_short', 'shares': abs(position),
                                      'price': current_price})

    portfolio_values.iloc[-1] = capital + (position * data[price_col].iloc[-1])

    # Calculate performance metrics
    returns = portfolio_values.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0

    return sharpe_ratio, portfolio_values, trade_log