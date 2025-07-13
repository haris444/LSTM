# indicators.py - Updated with Bollinger Bands signal generation
import pandas as pd
import numpy as np


def calculate_sma(price_series, window):
    """Calculates Simple Moving Average."""
    return price_series.rolling(window=window, min_periods=1).mean()


def calculate_ema(price_series, window):
    """Calculates Exponential Moving Average."""
    return price_series.ewm(span=window, adjust=False, min_periods=1).mean()


def calculate_rsi(price_series, window=14):
    """Calculates the Relative Strength Index."""
    delta = price_series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    # Add a small epsilon to the denominator to prevent division by zero
    rs = gain / (loss + 1e-9)

    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(price_series, window_slow=26, window_fast=12, window_signal=9):
    """Calculates MACD and Signal Line."""
    ema_fast = calculate_ema(price_series, window=window_fast)
    ema_slow = calculate_ema(price_series, window=window_slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, window=window_signal)

    return macd_line, signal_line


def calculate_bollinger_bands(price_series, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    sma = calculate_sma(price_series, window)
    std_dev = price_series.rolling(window=window, min_periods=1).std()

    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)

    return upper_band, lower_band


def calculate_obv(price_series, volume_series):
    """Calculates On-Balance Volume."""
    obv = (np.sign(price_series.diff()) * volume_series).fillna(0).cumsum()
    return obv


# NEW BOLLINGER BANDS SIGNAL FUNCTIONS
def generate_bollinger_band_signals(price_series, bb_upper, bb_lower):
    """
    Generate mean-reversion signals based on Bollinger Band breaches.

    Args:
        price_series: Stock price time series
        bb_upper: Upper Bollinger Band
        bb_lower: Lower Bollinger Band

    Returns:
        pd.Series with:
        +1: Price crosses below lower band (oversold, buy signal)
        -1: Price crosses above upper band (overbought, sell signal)
        0: Price within bands or no clear signal
    """
    signals = pd.Series(0, index=price_series.index)

    # Oversold condition: Price crosses below lower band
    oversold_cross = (price_series < bb_lower) & (price_series.shift(1) >= bb_lower.shift(1))
    signals[oversold_cross] = 1

    # Overbought condition: Price crosses above upper band
    overbought_cross = (price_series > bb_upper) & (price_series.shift(1) <= bb_upper.shift(1))
    signals[overbought_cross] = -1

    return signals