# improved_simulator.py
import pandas as pd
import numpy as np
from indicators import calculate_rsi, calculate_bollinger_bands, calculate_sma
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


def calculate_runtime_indicators(data, params):
    """
    Calculate indicators that don't depend on crossover logic.
    These are calculated at runtime during the simulation.
    """
    price_col = params.get('price_column', 'MSFT_close')

    # We only need to calculate non-crossover indicators here
    indicators = {}
    indicators['rsi'] = calculate_rsi(data[price_col], params['rsi_window'])
    upper, lower = calculate_bollinger_bands(data[price_col], params['bb_window'], params['bb_std_dev'])
    indicators['bb_upper'] = upper
    indicators['bb_lower'] = lower
    indicators['bb_middle'] = calculate_sma(data[price_col], params['bb_window'])

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

    # 1. SMA Crossover Signal (reading pre-computed state)
    signals.append(generate_signal(indicators['sma_crossover'].iloc[current_idx], params['sma_theta_plus'],
                                   params['sma_theta_minus']))

    # 2. EMA Crossover Signal (reading pre-computed state)
    signals.append(generate_signal(indicators['ema_crossover'].iloc[current_idx], params['ema_theta_plus'],
                                   params['ema_theta_minus']))

    # 3. MACD Crossover Signal (reading pre-computed state)
    signals.append(generate_signal(indicators['macd_crossover'].iloc[current_idx], params['macd_theta_plus'],
                                   params['macd_theta_minus']))

    # 4. RSI Signal (runtime calculation)
    signals.append(
        generate_signal(indicators['rsi'].iloc[current_idx], params['rsi_theta_plus'], params['rsi_theta_minus']))

    # 5. Bollinger Bands Signal (runtime calculation)
    price_curr = indicators['bb_middle'].iloc[current_idx]
    if price_curr < indicators['bb_lower'].iloc[current_idx]:
        signals.append(1)
    elif price_curr > indicators['bb_upper'].iloc[current_idx]:
        signals.append(-1)
    else:
        signals.append(0)

    # 6. Fundamental Indicators (reading pre-computed state)
    if 'PE_Ratio' in indicators:
        pe_signal = generate_signal(indicators['PE_Ratio'].iloc[current_idx], params['pe_theta_plus'],
                                    params['pe_theta_minus'])
        signals.append(-pe_signal)  # Inverted: Low P/E is a buy signal

    if 'Earnings_Surprise' in indicators:
        surprise = indicators['Earnings_Surprise'].iloc[current_idx]
        signals.append(generate_signal(surprise, params['surprise_theta_plus'], params['surprise_theta_minus']))

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
        weighted_sum = np.sum([s * w for s, w in zip(signals, weights or [])])
        return int(np.sign(weighted_sum))
    return 0


def run_improved_simulation(data, params):
    """
    Run trading simulation using a DataFrame that includes pre-computed crossover states.
    """
    # Calculate runtime indicators like RSI and Bollinger Bands
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
        final_decision = aggregate_signals(signals, method=params.get('aggregation_method'))

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