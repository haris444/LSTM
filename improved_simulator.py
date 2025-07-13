# improved_simulator.py (Optimized with Numba and Hybrid Logging)
import pandas as pd
import numpy as np
from numba import jit

# Assuming these functions exist in your project and are correctly defined.
from indicators import calculate_rsi, calculate_bollinger_bands, calculate_sma, calculate_obv
from fundamental_indicators import add_fundamental_indicators_to_data


# =============================================================================
# NUMBA-OPTIMIZED CORE SIMULATION LOOP (FOR SPEED)
# =============================================================================
@jit(nopython=True, cache=True)
def fast_simulation_loop(prices, final_decisions, initial_capital, transaction_fee, lambda_worst):
    """
    This is the core simulation loop, compiled to fast machine code by Numba.
    It does NOT generate a trade log, prioritizing speed for grid search.
    """
    capital = initial_capital
    position = 0
    portfolio_values = np.full(prices.shape, np.nan, dtype=np.float64)

    for i in range(1, len(prices)):
        portfolio_values[i - 1] = capital + (position * prices[i - 1])
        current_price = prices[i]
        final_decision = final_decisions[i]

        if position != 0 and (final_decision == 0 or final_decision == -np.sign(position)):
            capital += position * current_price - transaction_fee
            position = 0

        if position == 0 and final_decision != 0:
            if final_decision > 0:
                if current_price > 0:
                    max_shares = int((capital - transaction_fee) / current_price)
                    if max_shares > 0:
                        position = max_shares
                        capital -= position * current_price + transaction_fee
            else:
                worst_case_price = lambda_worst * current_price
                if worst_case_price > 0:
                    max_shares = int((capital - transaction_fee) / worst_case_price)
                    if max_shares > 0:
                        position = -max_shares
                        capital += abs(position) * current_price - transaction_fee

    if len(prices) > 0:
        portfolio_values[-1] = capital + (position * prices[-1])
    return portfolio_values


# =============================================================================
# PYTHON-BASED SIMULATION LOOP (FOR ACCURATE LOGGING)
# =============================================================================
def slow_simulation_loop_with_logging(data, prices, final_decisions, initial_capital, transaction_fee, lambda_worst):
    """
    A pure Python version of the simulation loop that correctly generates a trade log.
    Used for analysis where details are more important than maximum speed.
    """
    capital = initial_capital
    position = 0
    portfolio_values = pd.Series(index=data.index, dtype=float)
    trade_log = []

    for i in range(1, len(data)):
        portfolio_values.iloc[i - 1] = capital + (position * prices[i - 1])
        current_price = prices[i]
        final_decision = final_decisions[i]

        if position != 0 and (final_decision == 0 or final_decision == -np.sign(position)):
            action = 'close_long' if position > 0 else 'close_short'
            capital += position * current_price - transaction_fee
            trade_log.append({'date': data.index[i], 'action': action, 'shares': abs(position), 'price': current_price})
            position = 0

        if position == 0 and final_decision != 0:
            if final_decision > 0:
                if current_price > 0:
                    max_shares = int((capital - transaction_fee) / current_price)
                    if max_shares > 0:
                        position = max_shares
                        capital -= position * current_price + transaction_fee
                        trade_log.append(
                            {'date': data.index[i], 'action': 'open_long', 'shares': position, 'price': current_price})
            else:
                worst_case_price = lambda_worst * current_price
                if worst_case_price > 0:
                    max_shares = int((capital - transaction_fee) / worst_case_price)
                    if max_shares > 0:
                        position = -max_shares
                        capital += abs(position) * current_price - transaction_fee
                        trade_log.append({'date': data.index[i], 'action': 'open_short', 'shares': abs(position),
                                          'price': current_price})

    if len(data) > 0:
        portfolio_values.iloc[-1] = capital + (position * prices[-1])
    return portfolio_values, trade_log


# =============================================================================
# PANDAS-BASED HELPER FUNCTIONS (UNCHANGED)
# =============================================================================

def vectorized_generate_signal(series, theta_plus, theta_minus):
    series = pd.to_numeric(series, errors='coerce')
    conditions = [series > theta_plus, series < theta_minus]
    choices = [1, -1]
    return np.select(conditions, choices, default=0)


def vectorized_calculate_trend_slope(series, window):
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = x.var()
    if x_var == 0: return pd.Series(0.0, index=series.index)
    y_mean_rolling = series.rolling(window).mean()
    xy_cov_rolling = (series.rolling(window).apply(lambda y: np.mean(x * y), raw=True) - y_mean_rolling * x_mean)
    slope = xy_cov_rolling / x_var
    return slope.div(y_mean_rolling.abs() + 1e-9).replace([np.inf, -np.inf], 0).fillna(0)


def calculate_all_indicators(data, params):
    price_col, volume_col = params.get('price_column', 'MSFT_close'), params.get('volume_column', 'MSFT_volume')
    indicators = pd.DataFrame(index=data.index)
    indicators['price'] = data[price_col]
    if 'rsi_window' in params: indicators['rsi'] = calculate_rsi(data[price_col], params['rsi_window'])
    if 'bb_window' in params:
        indicators['bb_middle'] = calculate_sma(data[price_col], params['bb_window'])
        upper, lower = calculate_bollinger_bands(data[price_col], params['bb_window'], params['bb_std_dev'])
        indicators['bb_upper'], indicators['bb_lower'] = upper, lower
    if 'obv_window' in params: indicators['obv'] = calculate_obv(data[price_col], data[volume_col])
    for col in ['sma_crossover', 'ema_crossover', 'macd_crossover', 'PE_Ratio', 'Earnings_Surprise']:
        if col in data.columns: indicators[col] = data[col]
    return indicators.bfill().ffill().fillna(0)


def generate_all_signals_vectorized(indicators, params):
    signals = pd.DataFrame(index=indicators.index)

    # SMA and EMA crossover signals (unchanged)
    if 'sma_crossover' in indicators:
        signals['sma'] = indicators['sma_crossover']
    if 'ema_crossover' in indicators:
        signals['ema'] = indicators['ema_crossover']

    # RSI signal (inverted - overbought is sell signal)
    if 'rsi' in indicators:
        signals['rsi'] = -vectorized_generate_signal(
            indicators['rsi'], params['rsi_theta_plus'], params['rsi_theta_minus']
        )

    # MACD crossover signal (unchanged)
    if 'macd_crossover' in indicators:
        signals['macd'] = indicators['macd_crossover']

    # Bollinger Bands signal (unchanged)
    if all(k in indicators for k in ['bb_upper', 'bb_lower', 'price']):
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        bb_position = (indicators['price'] - indicators['bb_lower']).div(
            bb_range.where(bb_range != 0)
        ).replace([np.inf, -np.inf], 0.5).fillna(0.5)
        signals['bb'] = -vectorized_generate_signal(
            bb_position, params['bb_theta_plus'], params['bb_theta_minus']
        )

    # FIXED OBV Implementation - Based on Price-OBV Divergence/Convergence
    if all(k in indicators for k in ['obv', 'price']):
        obv_slope = vectorized_calculate_trend_slope(indicators['obv'], params['obv_window'])
        price_slope = vectorized_calculate_trend_slope(indicators['price'], params['obv_window'])

        # Define trend directions
        obv_rising = obv_slope > params['obv_theta_plus']
        obv_falling = obv_slope < params['obv_theta_minus']
        price_rising = price_slope > 0
        price_falling = price_slope < 0

        # Signal generation based on convergence and divergence
        condlist = [
            # Bullish signals: OBV rising with price rising (confirmation) OR
            # OBV rising with price falling (bullish divergence)
            obv_rising & price_rising,  # Uptrend confirmation
            obv_rising & price_falling,  # Bullish divergence

            # Bearish signals: OBV falling with price falling (confirmation) OR
            # OBV falling with price rising (bearish divergence)
            obv_falling & price_falling,  # Downtrend confirmation
            obv_falling & price_rising  # Bearish divergence
        ]

        choicelist = [1, 1, -1, -1]  # Both bullish conditions = +1, both bearish = -1
        signals['obv'] = np.select(condlist, choicelist, default=0)

    # Fundamental indicators (unchanged)
    if 'PE_Ratio' in indicators:
        signals['pe'] = -vectorized_generate_signal(
            indicators['PE_Ratio'], params['pe_theta_plus'], params['pe_theta_minus']
        )
    if 'Earnings_Surprise' in indicators:
        signals['surprise'] = vectorized_generate_signal(
            indicators['Earnings_Surprise'], params['surprise_theta_plus'], params['surprise_theta_minus']
        )

    # Single indicator testing logic (unchanged)
    use_single = params.get('use_single_indicator', None)
    if use_single is not None:
        indicator_map = {0: 'sma', 1: 'ema', 2: 'rsi', 3: 'macd', 4: 'bb', 5: 'obv', 6: 'pe', 7: 'surprise'}
        indicator_to_keep = indicator_map.get(use_single)
        if indicator_to_keep in signals.columns:
            return signals[[indicator_to_keep]].reindex(indicators.index).fillna(0).astype(int)
        else:
            return pd.DataFrame(0, index=indicators.index, columns=['empty'])

    return signals.reindex(indicators.index).fillna(0).astype(int)


def aggregate_signals_vectorized(signals_df, method='majority_vote', weights=None):
    if signals_df.empty or 'empty' in signals_df.columns: return pd.Series(0, index=signals_df.index)
    if method == 'weighted_sum' and weights:
        signal_order = ['sma', 'ema', 'rsi', 'macd', 'bb', 'obv', 'pe', 'surprise']
        weight_map = dict(zip(signal_order, weights))
        weighted_sum = sum(signals_df[col] * weight_map[col] for col in signals_df.columns if col in weight_map)
        return np.sign(weighted_sum).astype(int)
    return np.sign(signals_df.sum(axis=1)).astype(int)


# =============================================================================
# MAIN SIMULATOR FUNCTION (NOW WITH HYBRID LOGIC)
# =============================================================================
def run_improved_simulation(data, params, log_trades=False):
    """
    Run trading simulation with a choice between a fast Numba loop (no logs)
    and a slower Python loop (with logs).
    """
    try:
        indicators = calculate_all_indicators(data.copy(), params)
        signals_df = generate_all_signals_vectorized(indicators, params)
        final_decision_series = aggregate_signals_vectorized(
            signals_df, method=params.get('aggregation_method', 'majority_vote'), weights=params.get('signal_weights')
        )
    except KeyError as e:
        print(f"Parameter error during data prep: {e}")
        return -np.inf, pd.Series(dtype=float), []

    prices_np = data[params.get('price_column', 'MSFT_close')].to_numpy(dtype=np.float64)
    final_decisions_np = final_decision_series.to_numpy(dtype=np.int64)

    # --- HYBRID LOOP SELECTION ---
    if log_trades:
        # Use the slower loop that generates a detailed trade log for analysis.
        portfolio_values, trade_log = slow_simulation_loop_with_logging(
            data, prices_np, final_decisions_np, params.get('initial_capital', 100000.0),
            params.get('transaction_fee', 5.0), params.get('lambda_worst', 1.5)
        )
    else:
        # Use the fast Numba loop for grid search; no trade log is generated.
        portfolio_values_np = fast_simulation_loop(
            prices_np, final_decisions_np, params.get('initial_capital', 100000.0),
            params.get('transaction_fee', 5.0), params.get('lambda_worst', 1.5)
        )
        portfolio_values = pd.Series(portfolio_values_np, index=data.index)
        trade_log = []

    # --- Performance Metric Calculation ---
    if len(portfolio_values.dropna()) > 1:
        returns = portfolio_values.pct_change().dropna()
        if not returns.empty and returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    return sharpe_ratio, portfolio_values, trade_log
