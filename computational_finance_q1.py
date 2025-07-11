import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys

# Check if data collection and preparation files exist
def check_and_prepare_data():
    """
    Check if necessary data files exist, and run data collection/preparation if needed.
    """
    # Define file paths
    collected_data_path = 'data/collected_data.pkl'
    daily_data_path = 'data/daily_data.csv'

    # Check if collected_data.pkl exists
    if not os.path.exists(collected_data_path):
        print("collected_data.pkl not found. Running data collection...")
        try:
            # Import and run the data collection module instead of exec
            import TimeSeriesDataCollection
            print("Data collection completed successfully.")
        except ImportError:
            print("ERROR: TimeSeriesDataCollection.py not found!")
            print("Please ensure TimeSeriesDataCollection.py is in the current directory.")
            return False
        except Exception as e:
            print(f"ERROR during data collection: {e}")
            return False
    else:
        print("collected_data.pkl already exists. Skipping data collection.")

    # Check if daily_data.csv exists
    if not os.path.exists(daily_data_path):
        print("daily_data.csv not found. Running data preparation...")
        try:
            # Import and run data preparation
            from data_preparation import prepare_daily_data
            prepare_daily_data()
            print("Data preparation completed successfully.")
        except ImportError:
            print("ERROR: data_preparation.py not found!")
            print("Please ensure data_preparation.py is in the current directory.")
            return False
        except Exception as e:
            print(f"ERROR during data preparation: {e}")
            return False
    else:
        print("daily_data.csv already exists. Skipping data preparation.")

    # Verify if daily_data.csv was created after attempting preparation
    if not os.path.exists(daily_data_path):
        print("ERROR: daily_data.csv was not created after preparation attempt.")
        return False

    return True

# Run data checks and preparation
print("=" * 60)
print("CHECKING AND PREPARING DATA")
print("=" * 60)

if not check_and_prepare_data():
    print("Data preparation failed. Please check the error messages above and fix them.")
    # Don't use SystemExit in Colab, just stop execution
    print("Stopping execution due to data preparation failure.")
    sys.exit()

# Import the simulator after ensuring data is ready
try:
    from simulator import run_simulation
except ImportError:
    print("ERROR: simulator.py not found!")
    print("Please ensure simulator.py is in the current directory.")
    sys.exit()

print("\n" + "=" * 60)
print("LOADING PREPARED DATA")
print("=" * 60)

# Load the prepared data
df = None # Initialize df to None
try:
    df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
    print(f"Successfully loaded data with shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
except FileNotFoundError:
    print("ERROR: daily_data.csv not found after successful preparation check.")
    print("Something went wrong between checking and loading.")
    sys.exit()
except Exception as e:
    print(f"ERROR loading daily_data.csv: {e}")
    sys.exit()

# Check if df is loaded before proceeding
if df is None or df.empty:
    print("ERROR: DataFrame is empty or was not loaded correctly.")
    sys.exit()

# Split data into training and testing sets (80/20 split)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"\nTraining data from {train_df.index.min()} to {train_df.index.max()}")
print(f"Testing data from {test_df.index.min()} to {test_df.index.max()}")
print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

print("\n" + "=" * 60)
print("PARAMETER OPTIMIZATION")
print("=" * 60)

# --- Define Parameter Search Space (REDUCED FOR MEMORY) ---
param_grid = {
    # Moving Average Crossover: Test different trend lengths
    'sma_short': [10, 20, 50],
    'sma_long': [100, 200],

    # Relative Strength Index: Test different sensitivities and thresholds
    'rsi_window': [7, 14, 21],
    'rsi_buy': [25, 30],
    'rsi_sell': [70, 75],

    # MACD: Standard parameters are usually best, but we can test a slight variation
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],

    # Bollinger Bands: Test different volatility sensitivities
    'bb_window': [20, 30],
    'bb_std_dev': [2, 2.5],

    # On-Balance Volume: Test different smoothing periods for the signal
    'obv_window': [20, 30],

    # Short-selling risk parameter (usually kept constant)
    'lambda_worst': [1.5]
}

# Generate all possible combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Generated {len(param_combinations)} parameter combinations for testing.")

# Filter out invalid combinations
valid_combinations = []
for params in param_combinations:
    if params['sma_short'] < params['sma_long']:  # Ensure short window < long window
        valid_combinations.append(params)

print(f"Valid parameter combinations: {len(valid_combinations)}")
print(f"Starting optimization...")

# --- Run Optimization on Training Data ---
results = []
best_sharpe = -np.inf
best_params = None

for i, params in enumerate(valid_combinations):
    try:
        sharpe, _ = run_simulation(train_df, params)
        results.append({'params': params, 'sharpe_ratio': sharpe})

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

        # Print progress for each combination (since there are fewer now)
        print(f"Processed {i+1}/{len(valid_combinations)} combinations | "
              f"Current Sharpe: {sharpe:.4f} | Best Sharpe: {best_sharpe:.4f}")

    except Exception as e:
        print(f"Error in combination {i+1}: {e}")
        continue

print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS")
print("=" * 60)

print(f"Best Sharpe Ratio on Training Data: {best_sharpe:.4f}")
print(f"Optimal Parameters: {best_params}")

# --- Evaluate Best Parameters on Test Data ---
if best_params is None:
    print("\nNo profitable strategy found in the training phase.")
    print("All parameter combinations resulted in negative or zero Sharpe ratios.")
else:
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    try:
        test_sharpe, test_portfolio_values = run_simulation(test_df, best_params)
        print(f"Sharpe Ratio on Test Data: {test_sharpe:.4f}")

        # Calculate Cumulative Return on Test Data
        test_cumulative_return = (test_portfolio_values.iloc[-1] / test_portfolio_values.iloc[0] - 1) * 100
        print(f"Cumulative Return on Test Data: {test_cumulative_return:.2f}%")

        # Calculate additional performance metrics
        test_returns = test_portfolio_values.pct_change().dropna()
        test_volatility = test_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        max_drawdown = ((test_portfolio_values.cummax() - test_portfolio_values) / test_portfolio_values.cummax()).max() * 100

        print(f"Annualized Volatility: {test_volatility:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        # Table of Results
        results_summary = {
            'Metric': [
                'Optimal Sharpe (Train)',
                'Optimal Sharpe (Test)',
                'Cumulative Return (Test)',
                'Annualized Volatility (Test)',
                'Maximum Drawdown (Test)'
            ],
            'Value': [
                f"{best_sharpe:.4f}",
                f"{test_sharpe:.4f}",
                f"{test_cumulative_return:.2f}%",
                f"{test_volatility:.2f}%",
                f"{max_drawdown:.2f}%"
            ]
        }
        results_df = pd.DataFrame(results_summary)
        print(results_df.to_string(index=False))

        # Plot of Portfolio Value
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 7))
        plt.plot(test_portfolio_values, linewidth=2, color='cyan')
        plt.title(f'Portfolio Value Over Time (Test Set)\n'
                 f'Sharpe: {test_sharpe:.3f} | Return: {test_cumulative_return:.2f}% | '
                 f'Max Drawdown: {max_drawdown:.2f}%', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Add some styling
        plt.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot returns distribution
        plt.figure(figsize=(10, 6))
        plt.hist(test_returns * 100, bins=50, alpha=0.7, color='lightblue', edgecolor='white')
        plt.title('Distribution of Daily Returns (Test Set)', fontsize=14)
        plt.xlabel('Daily Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.axvline(test_returns.mean() * 100, color='red', linestyle='--',
                   label=f'Mean: {test_returns.mean()*100:.3f}%')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during test evaluation: {e}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)