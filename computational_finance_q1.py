import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys
import pickle

# Import necessary functions from other files
from improved_simulator import run_improved_simulation
from fundamental_indicators import add_fundamental_indicators_to_data
from indicators import calculate_sma


# =============================================================================
# Helper Function for Parameter Sensitivity Analysis
# =============================================================================
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


# =============================================================================
# Main Execution
# =============================================================================

print("=" * 60)
print("COMPUTATIONAL FINANCE - QUESTION 1 IMPLEMENTATION")
print("=" * 60)

# --- 1. Load Data ---
print("\nLoading and Preparing Data...")
try:
    # Load the daily price and volume data
    df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
    print(f"Successfully loaded daily price data with shape: {df.shape}")

    # Load the pickled data to get real fundamental data
    with open('data/collected_data.pkl', 'rb') as f:
        collected_data = pickle.load(f)

    # Extract the real earnings data for MSFT
    msft_earnings_df = collected_data.get('fundamental', {}).get('MSFT_EARNINGS', pd.DataFrame())

    if msft_earnings_df.empty:
        print("Warning: Real earnings data not found. Fundamental indicators will be skipped.")
    else:
        print(f"Successfully loaded real earnings data for MSFT with {len(msft_earnings_df)} reports.")

    # Add fundamental indicators to the main DataFrame
    print("Adding fundamental indicators (P/E Ratio, Earnings Surprise)...")
    df = add_fundamental_indicators_to_data(df, msft_earnings_df)
    print(f"Enhanced data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

except FileNotFoundError:
    print("ERROR: Data files not found. Please run TimeSeriesDataCollection.py and data_preparation.py first.")
    sys.exit()
except Exception as e:
    print(f"ERROR loading or preparing data: {e}")
    sys.exit()

# --- 2. Split Data ---
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"\nTraining data from {train_df.index.min()} to {train_df.index.max()}")
print(f"Testing data from {test_df.index.min()} to {test_df.index.max()}")

# --- 3. Parameter Optimization ---
print("\n" + "=" * 60)
print("PARAMETER OPTIMIZATION (GRID SEARCH)")
print("=" * 60)

param_grid = {
    'initial_capital': [100000.0], 'transaction_fee': [5.0],
    'lambda_worst': [1.5, 2.0], 'price_column': ['MSFT_close'],
    'volume_column': ['MSFT_volume'],
    # Test both aggregation methods
    'aggregation_method': ['majority_vote', 'weighted_sum'],
    'sma_short_window': [10, 20, 50], 'sma_long_window': [100, 200],
    'ema_short_window': [12, 20], 'ema_long_window': [26, 50],
    'rsi_window': [7, 14, 21], 'rsi_theta_minus': [25, 30],
    'rsi_theta_plus': [70, 75], 'macd_fast': [12],
    'macd_slow': [26], 'macd_signal': [9],
    'bb_window': [20, 30], 'bb_std_dev': [2.0, 2.5],
    'obv_window': [20, 30], 'pe_theta_minus': [10, 12],
    'pe_theta_plus': [20, 25], 'surprise_theta_minus': [-5],
    'surprise_theta_plus': [5],
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Filter out invalid combinations
valid_combinations = [
    params for params in param_combinations
    if params['sma_short_window'] < params['sma_long_window'] and
       params['ema_short_window'] < params['ema_long_window'] and
       params['rsi_theta_minus'] < params['rsi_theta_plus'] and
       params['pe_theta_minus'] < params['pe_theta_plus'] and
       params['surprise_theta_minus'] < params['surprise_theta_plus']
]

print(f"Generated {len(valid_combinations)} valid parameter combinations for testing.")
print("Starting optimization...")

results = []
best_sharpe = -np.inf
best_params = None

for i, params in enumerate(valid_combinations):
    try:
        sharpe, portfolio_values, trade_log = run_improved_simulation(train_df, params)
        results.append({
            'params': params, 'sharpe_ratio': sharpe,
            'final_value': portfolio_values.iloc[-1],
            'total_trades': len(trade_log)
        })
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(valid_combinations)} | Best Sharpe: {best_sharpe:.4f}")
    except Exception as e:
        print(f"Error in combination {i + 1}: {e}")
        continue

# --- 4. Training Results and Analysis ---
print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS ON TRAINING DATA")
print("=" * 60)

if best_params:
    print(f"Best Sharpe Ratio Found: {best_sharpe:.4f}")
    print("\nOptimal Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    # Run sensitivity analysis
    analyze_parameter_sensitivity(results)
else:
    print("No profitable strategy found during the training phase.")

# --- 5. Test Set Evaluation ---
if best_params:
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST DATA")
    print("=" * 60)

    try:
        test_sharpe, test_portfolio, test_log = run_improved_simulation(test_df, best_params)

        # Performance Metrics
        returns = test_portfolio.pct_change().dropna()
        cum_return = (test_portfolio.iloc[-1] / test_portfolio.iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100

        # Calculate Drawdown
        rolling_max = test_portfolio.cummax()
        drawdown = ((rolling_max - test_portfolio) / rolling_max).max() * 100

        print(f"Sharpe Ratio: {test_sharpe:.4f}")
        print(f"Cumulative Return: {cum_return:.2f}%")
        print(f"Annualized Volatility: {volatility:.2f}%")
        print(f"Maximum Drawdown: {drawdown:.2f}%")
        print(f"Total Trades: {len(test_log)}")

        # --- 6. Visualization ---
        print("\nGenerating plots...")
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

        # Plot 1: Portfolio Value and Trades
        ax1.plot(test_portfolio.index, test_portfolio, color='cyan', label='Portfolio Value')
        ax1.set_title(
            f'Portfolio Performance on Test Set | Sharpe: {test_sharpe:.2f} | Return: {cum_return:.2f}% | Drawdown: {drawdown:.2f}%')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot trade markers
        buy_signals = [trade for trade in test_log if 'open_long' in trade['action']]
        sell_signals = [trade for trade in test_log if 'open_short' in trade['action']]

        ax1.plot([trade['date'] for trade in buy_signals],
                 [trade['price'] for trade in buy_signals],
                 '^', color='lime', markersize=8, label='Buy Signal')

        ax1.plot([trade['date'] for trade in sell_signals],
                 [trade['price'] for trade in sell_signals],
                 'v', color='red', markersize=8, label='Sell Signal')
        ax1.legend()

        # Plot 2: Stock Price with Moving Averages
        price_col = best_params.get('price_column', 'MSFT_close')
        ax2.plot(test_df.index, test_df[price_col], color='white', label='MSFT Price', alpha=0.9)

        sma_short = calculate_sma(test_df[price_col], best_params['sma_short_window'])
        sma_long = calculate_sma(test_df[price_col], best_params['sma_long_window'])

        ax2.plot(sma_short.index, sma_short, color='orange', label=f"SMA({best_params['sma_short_window']})")
        ax2.plot(sma_long.index, sma_long, color='magenta', label=f"SMA({best_params['sma_long_window']})")

        ax2.set_ylabel('Stock Price ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred during test evaluation: {e}")

print("\n" + "=" * 60)
print("EXECUTION COMPLETE")
print("=" * 60)