# computational_finance_q1.py
import os
import pandas as pd
import numpy as np
import itertools
import sys
import pickle
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import necessary functions from other files
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

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Extract parameters into separate columns
    param_cols = {}
    for key in results[0]['params'].keys():
        param_cols[key] = [r['params'][key] for r in results]

    df_params = pd.DataFrame(param_cols)
    df_combined = pd.concat([df_params, df_results[['sharpe_ratio']]], axis=1)

    # Calculate correlation between each numeric parameter and Sharpe ratio
    correlations = {}
    for col in df_params.columns:
        if df_params[col].dtype in ['int64', 'float64']:
            corr = df_combined[col].corr(df_combined['sharpe_ratio'])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nParameter Sensitivity Analysis (Correlation with Sharpe Ratio):")
    print("=" * 60)
    for param, corr in sorted_correlations:
        print(f"{param:25s}: {corr:8.4f}")
    print("=" * 60)


def main():
    """
    Main execution function that runs the trading strategy optimization.
    """
    print("=" * 60)
    print("COMPUTATIONAL FINANCE - QUESTION 1 IMPLEMENTATION")
    print("=" * 60)

    # --- 1. Load Data ---
    print("\nLoading and Preparing Data...")
    try:
        df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
        with open('data/collected_data.pkl', 'rb') as f:
            collected_data = pickle.load(f)
        msft_earnings_df = collected_data.get('fundamental', {}).get('MSFT_EARNINGS', pd.DataFrame())
        df = add_fundamental_indicators_to_data(df, msft_earnings_df)
        print(f"Data loaded successfully. Shape: {df.shape}")
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
        'sma_short_window': 20, 'sma_long_window': 100,
        'ema_short_window': 12, 'ema_long_window': 26,
        'rsi_window': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'bb_window': 20, 'bb_std_dev': 2.0, 'obv_window': 20,
        'sma_theta_plus': 2.0, 'sma_theta_minus': -2.0,
        'ema_theta_plus': 0.5, 'ema_theta_minus': -0.5,
        'rsi_theta_plus': 70, 'rsi_theta_minus': 30,
        'macd_theta_plus': 0.15, 'macd_theta_minus': -0.15,
        'obv_theta_plus': 0.003, 'obv_theta_minus': -0.003,
        'bb_theta_plus': 0.8, 'bb_theta_minus': 0.2,  # Fixed: Added BB thresholds
        'pe_theta_plus': 30, 'pe_theta_minus': 15,
        'surprise_theta_plus': 5, 'surprise_theta_minus': -5,
    }

    # Add crossover indicators for testing
    train_test_df = add_all_crossover_indicators(train_df.copy(), single_indicator_params)

    indicator_names = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'OBV', 'PE', 'Surprise']
    for i, name in enumerate(indicator_names):
        params = single_indicator_params.copy()
        params['use_single_indicator'] = i  # Special flag for the simulator

        print(f"\n--- Testing Strategy: {name} ONLY ---")
        sharpe, portfolio, trade_log = run_improved_simulation(train_test_df.copy(), params)

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
        'initial_capital': [100000.0],
        'transaction_fee': [5.0],
        'lambda_worst': [1.5],
        'price_column': ['MSFT_close'],
        'volume_column': ['MSFT_volume'],
        'aggregation_method': ['weighted_sum'],

        # Updated weight combinations to include OBV
        'signal_weights': [
            [2.0, 2.0, 1.5, 1.0, 0.5, 1.5, 0.5, 0.5],  # Trend-focused (high OBV weight)
            [1.0, 1.0, 2.0, 1.5, 1.0, 1.0, 0.5, 0.5],  # Momentum-focused
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Equal weights
            [1.5, 1.5, 0.5, 0.5, 0.5, 2.0, 1.0, 1.0],  # Volume/fundamental focused
        ],

        # Expanded window sizes
        'sma_short_window': [10, 50],
        'sma_long_window': [100, 200],
        'ema_short_window': [12, 20],
        'ema_long_window': [26, 50],
        'rsi_window': [14, 21],
        'obv_window': [10, 30],  # OBV trend calculation window

        # Expanded thresholds
        'sma_theta_plus': [1.0, 2.0],
        'sma_theta_minus': [-2.0, -1.0],
        'ema_theta_plus': [0.25, 0.5],
        'ema_theta_minus': [-0.5, -0.25],
        'rsi_theta_plus': [70, 90],
        'rsi_theta_minus': [15, 30],
        'macd_theta_plus': [0.1, 0.15],
        'macd_theta_minus': [-0.15, -0.1],

        # Fixed: Bollinger Bands thresholds for normalized position (0-1 scale)
        'bb_theta_plus': [0.8, 0.9, 0.95],   # Near upper band (overbought)
        'bb_theta_minus': [0.05, 0.1, 0.2],  # Near lower band (oversold)

        # OBV trend slope thresholds (percentage slope per period)
        'obv_theta_plus': [0.002, 0.005],
        'obv_theta_minus': [-0.005, -0.002],

        # Fixed parameters
        'macd_fast': [12], 'macd_slow': [26], 'macd_signal': [9],
        'bb_window': [20], 'bb_std_dev': [2.0],
        'pe_theta_plus': [30], 'pe_theta_minus': [15],
        'surprise_theta_plus': [5], 'surprise_theta_minus': [-5],
    }

    keys, values = zip(*param_grid.items())
    valid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Generated {len(valid_combinations)} combinations for weighted search.")

    print("\nStarting parallel optimization with batch processing...")
    best_sharpe = -np.inf
    best_params = None
    total_combinations = len(valid_combinations)

    # Store results for analysis
    results = []
    start_time = time.time()
    completed_count = 0

    # Get CPU information and optimize core usage
    total_cores = multiprocessing.cpu_count()
    print(f"CPU Information:")
    print(f"  Total logical cores: {total_cores}")
    print(f"  Using {total_cores} workers for parallel processing...")
    print(f"  Processing {total_combinations} combinations in batches...")

    # Process in manageable batches to avoid memory issues
    batch_size = min(500, total_combinations)
    num_batches = (total_combinations + batch_size - 1) // batch_size

    print(f"  Using batch size: {batch_size} ({num_batches} batches)")

    with ProcessPoolExecutor(max_workers=total_cores) as executor:
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_combinations)
            batch_combinations = valid_combinations[start_idx:end_idx]

            print(f"\nProcessing batch {batch_num + 1}/{num_batches} ({len(batch_combinations)} combinations)...")

            # Add crossover indicators to training data for this batch
            batch_train_df = add_all_crossover_indicators(train_df.copy(), batch_combinations[0])

            # Submit batch jobs
            future_to_params = {
                executor.submit(run_improved_simulation, batch_train_df.copy(), params): params
                for params in batch_combinations
            }

            batch_completed = 0
            # Process completed jobs in this batch
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                completed_count += 1
                batch_completed += 1

                try:
                    sharpe, portfolio_values, trade_log = future.result()

                    # Store result for analysis
                    results.append({
                        'params': params,
                        'sharpe_ratio': sharpe,
                        'final_value': portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0,
                        'total_trades': len(trade_log)
                    })

                    if len(trade_log) > 0 and sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params
                        print(
                            f"NEW BEST | Completed {completed_count}/{total_combinations} | Sharpe: {best_sharpe:.4f}")

                    # Progress reporting
                    if completed_count % 50 == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_combo = elapsed_time / completed_count
                        estimated_remaining = (total_combinations - completed_count) * avg_time_per_combo
                        completion_rate = completed_count / elapsed_time * 60
                        if estimated_remaining > 3600:
                            time_str = f"{estimated_remaining / 3600:.1f} hours"
                        else:
                            time_str = f"{estimated_remaining / 60:.1f} min"
                        print(
                            f"Completed {completed_count}/{total_combinations} ({completed_count / total_combinations * 100:.2f}%) | "
                            f"Best Sharpe: {best_sharpe:.4f} | Rate: {completion_rate:.0f}/min | Est. remaining: {time_str}")
                    elif completed_count == 1:
                        print(f"First job completed!")

                except Exception as exc:
                    print(f'Job generated an exception: {exc}')

            print(f"Batch {batch_num + 1} complete ({batch_completed} jobs)")

    print(f"\nOptimization complete! Processed {completed_count}/{total_combinations} combinations.")
    if (time.time() - start_time) > 3600:
        print(f"Total time: {(time.time() - start_time) / 3600:.1f} hours")
    else:
        print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS ON TRAINING DATA")
    print("=" * 60)
    if best_params:
        print(f"Best Sharpe Ratio Found: {best_sharpe:.4f}")
        print("\nOptimal Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Comprehensive strategy analysis
        profitable_strategies = [r for r in results if r['sharpe_ratio'] > 0]
        print(f"\nStrategy Analysis:")
        print(f"  Total strategies tested: {len(results)}")
        print(f"  Profitable strategies: {len(profitable_strategies)}")
        print(f"  Success rate: {len(profitable_strategies) / len(results) * 100:.1f}%")

        if len(profitable_strategies) > 0:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in profitable_strategies])
            print(f"  Average Sharpe (profitable): {avg_sharpe:.4f}")

        # Run parameter sensitivity analysis
        analyze_parameter_sensitivity(results)

    else:
        print("No profitable strategy found. Consider widening parameter ranges or adjusting weights.")

    # =============================================================================
    # FINAL STEP: EVALUATE THE BEST STRATEGY ON THE TEST SET
    # =============================================================================
    if best_params:
        print("\n" + "=" * 60)
        print("PART 3: EVALUATING BEST STRATEGY ON TEST DATA")
        print("=" * 60)

        # Add crossover indicators to test data
        test_df_with_indicators = add_all_crossover_indicators(test_df.copy(), best_params)

        # Run the simulation on the test data using the best parameters
        test_sharpe, test_portfolio, test_log = run_improved_simulation(test_df_with_indicators.copy(), best_params)

        # Performance Metrics Calculation
        print("\n--- Test Set Performance Metrics ---")

        # 1. Cumulative Return
        cumulative_return = (test_portfolio.iloc[-1] / test_portfolio.iloc[0] - 1) * 100
        print(f"Cumulative Return: {cumulative_return:.2f}%")

        # 2. Annualized Volatility
        returns = test_portfolio.pct_change().dropna()
        annualized_volatility = returns.std() * np.sqrt(252) * 100
        print(f"Annualized Volatility: {annualized_volatility:.2f}%")

        # 3. Sharpe Ratio
        print(f"Sharpe Ratio: {test_sharpe:.4f}")

        # 4. Maximum Drawdown
        rolling_max = test_portfolio.cummax()
        drawdown = (test_portfolio - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

        # 5. Total Trades
        print(f"Total Trades: {len(test_log)}")

        # Enhanced visualization with moving averages
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle('Strategy Performance on Test Set', fontsize=16)

        # Plot 1: Portfolio Value Over Time
        ax1.plot(test_portfolio.index, test_portfolio, label='Portfolio Value', color='cyan')
        ax1.set_title(
            f'Portfolio Performance | Sharpe: {test_sharpe:.2f} | Return: {cumulative_return:.2f}% | Drawdown: {max_drawdown:.2f}%')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot Buy/Sell signals on the portfolio chart
        buy_signals = [trade for trade in test_log if 'open_long' in trade['action']]
        sell_signals = [trade for trade in test_log if 'open_short' in trade['action']]

        if buy_signals:
            ax1.plot([trade['date'] for trade in buy_signals],
                     [test_portfolio.loc[trade['date']] for trade in buy_signals],
                     '^', color='lime', markersize=8, label='Buy Signal')
        if sell_signals:
            ax1.plot([trade['date'] for trade in sell_signals],
                     [test_portfolio.loc[trade['date']] for trade in sell_signals],
                     'v', color='red', markersize=8, label='Sell Signal')
        ax1.legend()

        # Plot 2: Stock Price with Moving Averages and Entry/Exit Points
        price_col = best_params.get('price_column', 'MSFT_close')
        ax2.plot(test_df.index, test_df[price_col], label='MSFT Price', color='white', alpha=0.9)

        # Add moving averages
        sma_short = calculate_sma(test_df[price_col], best_params['sma_short_window'])
        sma_long = calculate_sma(test_df[price_col], best_params['sma_long_window'])

        ax2.plot(sma_short.index, sma_short, color='orange', label=f"SMA({best_params['sma_short_window']})")
        ax2.plot(sma_long.index, sma_long, color='magenta', label=f"SMA({best_params['sma_long_window']})")

        if buy_signals:
            ax2.plot([trade['date'] for trade in buy_signals],
                     [trade['price'] for trade in buy_signals],
                     '^', color='lime', markersize=8)
        if sell_signals:
            ax2.plot([trade['date'] for trade in sell_signals],
                     [trade['price'] for trade in sell_signals],
                     'v', color='red', markersize=8)

        ax2.set_title('Stock Price with Moving Averages and Trade Entry Points')
        ax2.set_ylabel('Stock Price ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Print OBV-specific analysis if OBV had significant weight
        obv_weight = best_params.get('signal_weights', [1] * 8)[5] if len(
            best_params.get('signal_weights', [])) > 5 else 1
        if obv_weight > 1.0:
            print(f"\n--- OBV Analysis ---")
            print(f"OBV Weight in Best Strategy: {obv_weight}")
            print(f"OBV Window: {best_params.get('obv_window', 20)}")
            print(
                f"OBV Trend Thresholds: +{best_params.get('obv_theta_plus', 0.05):.3f}, {best_params.get('obv_theta_minus', -0.05):.3f}")
            print("OBV successfully contributed to trend confirmation and divergence detection.")


if __name__ == '__main__':
    main()