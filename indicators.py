import pandas as pd
import numpy as np


def calculate_sma(price_series, window):
    """Calculates Simple Moving Average."""
    return price_series.rolling(window=window, min_periods=1).mean()


def calculate_ema(price_series, window):
    """Calculates Exponential Moving Average."""
    return price_series.ewm(span=window, adjust=False).mean()


def calculate_rsi(price_series, window=14):
    """Calculates the Relative Strength Index."""
    delta = price_series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # FIX: Prevent division by zero
    rs = gain / (loss + 1e-9)  # Add a small epsilon to the denominator

    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(price_series, window_slow=26, window_fast=12, window_signal=9):
    """Calculates MACD, Signal Line, and Histogram."""
    ema_fast = calculate_ema(price_series, window=window_fast)
    ema_slow = calculate_ema(price_series, window=window_slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, window=window_signal)

    return macd_line, signal_line


def calculate_bollinger_bands(price_series, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    sma = calculate_sma(price_series, window)
    std_dev = price_series.rolling(window=window).std()

    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)

    return upper_band, lower_band


def calculate_obv(price_series, volume_series):
    """Calculates On-Balance Volume."""
    obv = (np.sign(price_series.diff()) * volume_series).fillna(0).cumsum()
    return obv
