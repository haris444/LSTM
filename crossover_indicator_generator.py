# crossover_indicator_generator.py
import pandas as pd
from indicators import calculate_sma, calculate_ema, calculate_macd, calculate_obv


def generate_sma_crossover_indicator(data, price_col, short_window, long_window):
    """Generates a +1/-1/0 signal for SMA crossovers."""
    sma_short = calculate_sma(data[price_col], short_window)
    sma_long = calculate_sma(data[price_col], long_window)

    crossover_state = pd.Series(0, index=data.index)
    # Golden Cross: +1
    crossover_state[(sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))] = 1
    # Death Cross: -1
    crossover_state[(sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))] = -1
    return crossover_state


def generate_ema_crossover_indicator(data, price_col, short_window, long_window):
    """Generates a +1/-1/0 signal for EMA crossovers."""
    ema_short = calculate_ema(data[price_col], short_window)
    ema_long = calculate_ema(data[price_col], long_window)

    crossover_state = pd.Series(0, index=data.index)
    crossover_state[(ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))] = 1
    crossover_state[(ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))] = -1
    return crossover_state


def generate_macd_crossover_indicator(data, price_col, fast, slow, signal):
    """Generates a +1/-1/0 signal for MACD crossovers."""
    macd_line, signal_line = calculate_macd(data[price_col], slow, fast, signal)

    crossover_state = pd.Series(0, index=data.index)
    crossover_state[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
    crossover_state[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
    return crossover_state


def add_all_crossover_indicators(data, params):
    """
    Adds all pre-computed crossover indicator states to the dataframe.
    """
    price_col = params['price_column']

    # Add SMA Crossover State
    data['sma_crossover'] = generate_sma_crossover_indicator(
        data, price_col, params['sma_short_window'], params['sma_long_window']
    )

    # Add EMA Crossover State
    data['ema_crossover'] = generate_ema_crossover_indicator(
        data, price_col, params['ema_short_window'], params['ema_long_window']
    )

    # Add MACD Crossover State
    data['macd_crossover'] = generate_macd_crossover_indicator(
        data, price_col, params['macd_fast'], params['macd_slow'], params['macd_signal']
    )

    return data