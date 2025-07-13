import os
import pandas as pd
import numpy as np
import itertools
import sys
import pickle
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Import necessary functions from other files
# Ensure these files and functions exist and are correctly defined.
from improved_simulator import run_improved_simulation
from fundamental_indicators import add_fundamental_indicators_to_data
from indicators import calculate_sma
from crossover_indicator_generator import add_all_crossover_indicators


def analyze_parameter_sensitivity(results):
    """
    Analyze which parameters have the most impact on performance by correlating
    them with the Sharpe ratio.
    """
    if not results or 'params' not in results[0]:
        print("Not enough data for sensitivity analysis.")
        return

    df_results = pd.DataFrame(results)
    param_cols = {}
    for key in results[0]['params'].keys():
        param_cols[key] = [r['params'][key] for r in results]

    df_params = pd.DataFrame(param_cols)
    df_combined = pd.concat([df_params, df_results[['sharpe_ratio']]], axis=1)

    correlations = {}
    for col in df_params.columns:
        if df_params[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            corr = df_combined[col].corr(df_combined['sharpe_ratio'])
            correlations[col] = corr

    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nParameter Sensitivity Analysis (Correlation with Sharpe Ratio):")
    print("=" * 60)
    for param, corr in sorted_correlations:
        print(f"{param:25s}: {corr:8.4f}")
    print("=" * 60)


def run_simulation_worker(base_df, params):
    """
    Worker function that prepares data and runs the simulation for one parameter set.
    For the grid search, we do NOT log trades to maximize speed.
    """
    df_for_run = add_all_crossover_indicators(base_df.copy(), params)
    # The main grid search uses the fast Numba loop by default (log_trades=False)
    return run_improved_simulation(df_for_run, params)


def main():
    """
    Main execution function that runs the trading strategy optimization.
    """
    print("=" * 60)
    print("COMPUTATIONAL FINANCE - QUESTION 1 IMPLEMENTATION (OPTIMIZED & FIXED)")
    print("=" * 60)

    # --- 1. Load Data ---
    print("\nLoading and Preparing Data...")
    try:
        df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
        with open('data/collected_data.pkl', 'rb') as f:
            collected_data = pickle.load(f)
        msft_earnings_df = collected_data.get('fundamental', {}).get('MSFT_EARNINGS', pd.DataFrame())
        df = add_fundamental_indicators_to_data(df, msft_earnings_df)

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    except FileNotFoundError:
        print("ERROR: Data files not found. Please run collection and preparation scripts.")
        sys.exit()

    # --- 2. Split Data ---
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    print(f"\nTraining data: {len(train_df)} samples | Testing data: {len(test_df)} samples")

    # --- PART 1: TEST INDICATORS IN ISOLATION ---
    print("\n" + "=" * 60)
    print("PART 1: TESTING INDICATORS IN ISOLATION")
    print("=" * 60)

    single_indicator_params = {
        'initial_capital': 100000.0, 'transaction_fee': 5.0, 'lambda_worst': 1.5,
        'price_column': 'MSFT_close', 'volume_column': 'MSFT_volume',
        'sma_short_window': 20, 'sma_long_window': 100, 'ema_short_window': 12, 'ema_long_window': 26,
        'rsi_window': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'bb_window': 20, 'bb_std_dev': 2.0, 'obv_window': 20, 'sma_theta_plus': 2.0, 'sma_theta_minus': -2.0,
        'ema_theta_plus': 0.5, 'ema_theta_minus': -0.5, 'rsi_theta_plus': 70, 'rsi_theta_minus': 30,
        'macd_theta_plus': 0.15, 'macd_theta_minus': -0.15, 'obv_theta_plus': 0.003, 'obv_theta_minus': -0.003,
        'bb_theta_plus': 0.8, 'bb_theta_minus': 0.2, 'pe_theta_plus': 30, 'pe_theta_minus': 15,
        'surprise_theta_plus': 5, 'surprise_theta_minus': -5,
    }

    train_with_indicators = add_all_crossover_indicators(train_df.copy(), single_indicator_params)
    indicator_names = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'OBV', 'PE', 'Surprise']
    for i, name in enumerate(indicator_names):
        params = single_indicator_params.copy()
        params['use_single_indicator'] = i
        print(f"\n--- Testing Strategy: {name} ONLY ---")

        # FIX: Call with log_trades=True to get the detailed trade log for analysis.
        sharpe, portfolio, trade_log = run_improved_simulation(train_with_indicators.copy(), params, log_trades=True)

        if len(trade_log) > 0:
            print(f"  SUCCESS: Strategy made {len(trade_log)} trades.")
            print(f"  Sharpe Ratio: {sharpe:.4f}")
            print(f"  Final Portfolio Value: ${portfolio.iloc[-1]:,.2f}")
        else:
            print(f"  FAILURE: Strategy made 0 trades.")

    # --- PART 2: FULL GRID SEARCH WITH WEIGHTED SUM ---
    print("\n" + "=" * 60)
    print("PART 2: FULL GRID SEARCH WITH WEIGHTED AGGREGATION")
    print("=" * 60)

    param_grid = {
        # --- Core simulation parameters (fixed) ---
        'initial_capital': [100000.0],
        'transaction_fee': [5.0],
        'lambda_worst': [1.5],
        'price_column': ['MSFT_close'],
        'volume_column': ['MSFT_volume'],
        'aggregation_method': ['weighted_sum'],

        # --- STRATEGIC CHANGE: Test hypotheses based on isolated indicator performance ---
        # Order: [SMA, EMA, RSI, MACD, BB, OBV, PE, Surprise]
        'signal_weights': [
            # Hypothesis 1: Trust the winners. High weight on SMA, RSI, BB. Low/zero on others.
            [2.0, 0.5, 2.5, 0.5, 2.0, 0.5, 0.25, 0.25],

            # Hypothesis 2: Winners only. Turn off the historically losing indicators.
            [1.0, 0.0, 1.5, 0.0, 1.0, 0.0, 0.5, 0.5],

            # Hypothesis 3: Mean-reversion focus. RSI and BB are the primary drivers.
            [0.5, 0.0, 2.5, 0.0, 2.5, 0.0, 0.5, 0.0],
        ],

        # --- Focus search on the best indicators (SMA, RSI, BB) ---
        'sma_short_window': [10, 20],
        'sma_long_window': [50, 100, 150],
        'rsi_window': [14, 21],  # RSI was the star performer
        'rsi_theta_plus': [65, 70, 75],
        'rsi_theta_minus': [25, 30, 35],
        'bb_window': [20, 30],
        'bb_std_dev': [2.0, 2.5],
        'bb_theta_plus': [0.9, 0.95],
        'bb_theta_minus': [0.05, 0.1],

        # --- Keep a minimal search for weaker/riskier indicators ---
        'ema_short_window': [12],
        'ema_long_window': [26],
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'obv_window': [20, 40],
        'obv_theta_plus': [0.003],
        'obv_theta_minus': [-0.003],

        # --- CRITICAL FIX: Make the PE Ratio much more conservative to avoid blowing up the account ---
        'pe_theta_plus': [40, 50],  # Sell signal (overvalued) only when PE is very high
        'pe_theta_minus': [10],  # Buy signal (undervalued) only when PE is very low
        'surprise_theta_plus': [10],  # Only act on significant surprises
        'surprise_theta_minus': [-10],
    }

    keys, values = zip(*param_grid.items())
    valid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combinations = len(valid_combinations)
    print(f"Generated {total_combinations} combinations for grid search.")

    print("\nStarting parallel optimization...")
    best_sharpe = -np.inf
    best_params = None
    results = []
    start_time = time.time()
    completed_count = 0

    total_cores = psutil.cpu_count(logical=False)
    print(f"Using {total_cores} workers for parallel processing...")

    with ProcessPoolExecutor(max_workers=total_cores) as executor:
        future_to_params = {
            executor.submit(run_simulation_worker, train_df, params): params
            for params in valid_combinations
        }
        print(f"{len(future_to_params)} simulation jobs successfully submitted.")

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            completed_count += 1

            try:
                # The worker returns (sharpe, portfolio, EMPTY_log)
                sharpe, portfolio_values, _ = future.result()

                results.append({
                    'params': params, 'sharpe_ratio': sharpe,
                    'final_value': portfolio_values.iloc[-1] if not portfolio_values.empty else 0,
                    'total_trades': 0  # We don't know the trade count here, and that's okay
                })

                # FIX: Check the sharpe ratio directly, ignoring the empty trade log.
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    print(f"NEW BEST | Completed {completed_count}/{total_combinations} | Sharpe: {best_sharpe:.4f}")

                if completed_count % 50 == 0 or completed_count == 1:
                    elapsed_time = time.time() - start_time
                    if completed_count > 0:
                        avg_time_per_combo = elapsed_time / completed_count
                        estimated_remaining = (total_combinations - completed_count) * avg_time_per_combo
                        completion_rate = completed_count / elapsed_time * 60
                        time_str = f"{estimated_remaining / 60:.1f} min"
                        if estimated_remaining > 3600:
                            time_str = f"{estimated_remaining / 3600:.1f} hours"
                        print(
                            f"Progress: {completed_count}/{total_combinations} ({completed_count / total_combinations * 100:.2f}%) | Best Sharpe: {best_sharpe:.4f} | Rate: {completion_rate:.0f}/min | Est. Left: {time_str}")

            except Exception as exc:
                print(f'A job failed with an exception: {exc}')

    total_time = time.time() - start_time
    print(f"\nOptimization complete! Processed {completed_count}/{total_combinations} combinations.")
    print(f"Total time: {total_time / 60:.2f} minutes")

    # --- Analysis and Final Evaluation ---
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS ON TRAINING DATA")
    print("=" * 60)
    if best_params:
        print(f"Best Sharpe Ratio Found: {best_sharpe:.4f}")
        print("\nOptimal Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        analyze_parameter_sensitivity(results)
    else:
        print("No profitable strategy found.")

    if best_params:
        print("\n" + "=" * 60)
        print("PART 3: EVALUATING BEST STRATEGY ON TEST DATA")
        print("=" * 60)

        test_df_with_indicators = add_all_crossover_indicators(test_df.copy(), best_params)

        # FIX: Call with log_trades=True to get the detailed trade log for final analysis.
        test_sharpe, test_portfolio, test_log = run_improved_simulation(test_df_with_indicators, best_params,
                                                                        log_trades=True)

        print("\n--- Test Set Performance Metrics ---")
        cumulative_return = (test_portfolio.iloc[-1] / test_portfolio.iloc[0] - 1) * 100
        returns = test_portfolio.pct_change().dropna()
        annualized_volatility = returns.std() * np.sqrt(252) * 100
        rolling_max = test_portfolio.cummax()
        drawdown = (test_portfolio - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        print(f"Cumulative Return: {cumulative_return:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility:.2f}%")
        print(f"Sharpe Ratio: {test_sharpe:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {len(test_log)}")

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle('Strategy Performance on Test Set', fontsize=16)

        ax1.plot(test_portfolio.index, test_portfolio, label='Portfolio Value', color='cyan')
        ax1.set_title(
            f'Portfolio Performance | Final Value: ${test_portfolio.iloc[-1]:,.2f} | Sharpe: {test_sharpe:.2f}')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, linestyle='--', alpha=0.5)

        buy_signals = [trade for trade in test_log if 'open_long' in trade['action']]
        sell_signals = [trade for trade in test_log if 'open_short' in trade['action']]

        if buy_signals:
            buy_dates = [trade['date'] for trade in buy_signals]
            ax1.plot(buy_dates, test_portfolio.loc[buy_dates], '^', color='lime', markersize=8, label='Buy Signal')
        if sell_signals:
            sell_dates = [trade['date'] for trade in sell_signals]
            ax1.plot(sell_dates, test_portfolio.loc[sell_dates], 'v', color='red', markersize=8, label='Sell Signal')
        ax1.legend()

        price_col = best_params.get('price_column', 'MSFT_close')
        ax2.plot(test_df.index, test_df[price_col], label='MSFT Price', color='white', alpha=0.9)
        sma_short = calculate_sma(test_df[price_col], best_params['sma_short_window'])
        sma_long = calculate_sma(test_df[price_col], best_params['sma_long_window'])
        ax2.plot(sma_short.index, sma_short, color='orange', label=f"SMA({best_params['sma_short_window']})")
        ax2.plot(sma_long.index, sma_long, color='magenta', label=f"SMA({best_params['sma_long_window']})")

        if buy_signals: ax2.plot([t['date'] for t in buy_signals], [t['price'] for t in buy_signals], '^', color='lime',
                                 markersize=8)
        if sell_signals: ax2.plot([t['date'] for t in sell_signals], [t['price'] for t in sell_signals], 'v',
                                  color='red', markersize=8)

        ax2.set_title('Stock Price with Moving Averages and Trade Entry Points')
        ax2.set_ylabel('Stock Price ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == '__main__':
    main()
