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


# Add fundamental indicators to the data
def add_fundamental_indicators_to_data(df, price_column='MSFT_close'):
    """
    Add fundamental indicators (P/E Ratio and Earnings Surprise) to the dataset.
    """
    # Generate synthetic earnings data for demonstration
    earnings_info = generate_synthetic_earnings_data(df[price_column])

    # Calculate P/E ratio
    pe_ratio = calculate_pe_ratio(
        df[price_column],
        earnings_info['earnings_data'],
        earnings_info['reporting_dates']
    )
    df['PE_Ratio'] = pe_ratio

    # Calculate earnings surprise
    earnings_surprises = []
    for i, report_date in enumerate(earnings_info['reporting_dates']):
        if i >= 4:  # Need at least 4 historical points for forecasting
            historical = earnings_info['earnings_data'][:i]
            expected = estimate_earnings_with_model(historical, 1)[0]
            actual = earnings_info['earnings_data'][i]
            surprise = calculate_earnings_surprise(actual, expected)
            earnings_surprises.append(surprise)
        else:
            earnings_surprises.append(np.nan)

    # Map surprises to daily data (forward fill from reporting dates)
    surprise_series = pd.Series(index=df.index, dtype=float, data=0.0)
    for i, report_date in enumerate(earnings_info['reporting_dates']):
        if i < len(earnings_surprises) and not pd.isna(earnings_surprises[i]):
            closest_idx = df.index.get_indexer([report_date], method='nearest')[0]
            surprise_series.iloc[closest_idx:] = earnings_surprises[i]

    df['Earnings_Surprise'] = surprise_series
    return df


def generate_synthetic_earnings_data(price_series, noise_level=0.1):
    """Generate synthetic earnings data for demonstration purposes."""
    start_date = price_series.index[0]
    end_date = price_series.index[-1]

    reporting_dates = pd.date_range(start=start_date, end=end_date, freq='Q').tolist()
    earnings_data = []
    base_eps = 2.0

    for i, report_date in enumerate(reporting_dates):
        closest_price_idx = price_series.index.get_indexer([report_date], method='nearest')[0]
        price_at_report = price_series.iloc[closest_price_idx]

        if i == 0:
            eps = base_eps
        else:
            price_change = price_at_report / price_series.iloc[0]
            eps = base_eps * price_change * (1 + np.random.normal(0, noise_level))

        earnings_data.append(max(eps, 0.1))

    return {'earnings_data': earnings_data, 'reporting_dates': reporting_dates}


def calculate_pe_ratio(price_series, earnings_data, reporting_dates):
    """Calculate P/E ratio using most recent earnings data."""
    pe_ratios = pd.Series(index=price_series.index, dtype=float)

    for i, date in enumerate(price_series.index):
        prior_reports = [d for d in reporting_dates if d <= date]
        if prior_reports:
            latest_report_date = max(prior_reports)
            latest_eps = earnings_data[reporting_dates.index(latest_report_date)]

            if latest_eps > 0:
                pe_ratios.iloc[i] = price_series.iloc[i] / latest_eps
            else:
                pe_ratios.iloc[i] = np.nan
        else:
            pe_ratios.iloc[i] = np.nan

    pe_ratios.fillna(method='ffill', inplace=True)
    return pe_ratios


def estimate_earnings_with_model(historical_earnings, forecast_periods=4):
    """Estimate future earnings using linear regression."""
    if len(historical_earnings) < 4:
        return [historical_earnings[-1]] * forecast_periods

    from sklearn.linear_model import LinearRegression
    X = np.arange(len(historical_earnings)).reshape(-1, 1)
    y = np.array(historical_earnings)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(historical_earnings),
                         len(historical_earnings) + forecast_periods).reshape(-1, 1)
    forecasts = model.predict(future_X)

    return forecasts.tolist()


def calculate_earnings_surprise(actual_earnings, expected_earnings):
    """Calculate earnings surprise as percentage deviation."""
    if expected_earnings == 0:
        return 0.0
    return (actual_earnings - expected_earnings) / abs(expected_earnings) * 100


# Run data checks and preparation
print("=" * 60)
print("ENHANCED TRADING STRATEGY - QUESTION 1 IMPLEMENTATION")
print("=" * 60)

if not check_and_prepare_data():
    print("Data preparation failed. Please check the error messages above and fix them.")
    print("Stopping execution due to data preparation failure.")
    sys.exit()

# Import the enhanced simulator after ensuring data is ready
try:
    from improved_simulator import run_improved_simulation

    print("Using enhanced simulator with fundamental indicators...")
except ImportError:
    print("Enhanced simulator not found. Falling back to basic simulator...")
    try:
        from simulator import run_simulation

        run_improved_simulation = run_simulation  # Use alias for compatibility
    except ImportError:
        print("ERROR: No simulator found!")
        print("Please ensure either improved_simulator.py or simulator.py is in the current directory.")
        sys.exit()

print("\n" + "=" * 60)
print("LOADING AND ENHANCING DATA")
print("=" * 60)

# Load the prepared data
df = None
try:
    df = pd.read_csv('data/daily_data.csv', index_col=0, parse_dates=True)
    print(f"Successfully loaded data with shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Add fundamental indicators
    print("Adding fundamental indicators (P/E Ratio, Earnings Surprise)...")
    df = add_fundamental_indicators_to_data(df)
    print(f"Enhanced data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

except FileNotFoundError:
    print("ERROR: daily_data.csv not found after successful preparation check.")
    sys.exit()
except Exception as e:
    print(f"ERROR loading or enhancing data: {e}")
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
print("COMPREHENSIVE PARAMETER OPTIMIZATION")
print("=" * 60)

# --- Enhanced Parameter Search Space ---
param_grid = {
    # === BASIC SIMULATION PARAMETERS ===
    'initial_capital': [100000.0],
    'transaction_fee': [5.0],
    'lambda_worst': [1.5, 2.0],
    'price_column': ['MSFT_close'],
    'volume_column': ['MSFT_volume'],
    'aggregation_method': ['majority_vote'],

    # === MOVING AVERAGE PARAMETERS ===
    'sma_short_window': [10, 20, 50],
    'sma_long_window': [100, 200],
    'ema_short_window': [12, 20],
    'ema_long_window': [26, 50],

    # === RSI PARAMETERS ===
    'rsi_window': [7, 14, 21],
    'rsi_theta_minus': [25, 30],  # Oversold threshold
    'rsi_theta_plus': [70, 75],  # Overbought threshold

    # === MACD PARAMETERS ===
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],

    # === BOLLINGER BANDS PARAMETERS ===
    'bb_window': [20, 30],
    'bb_std_dev': [2.0, 2.5],

    # === OBV PARAMETERS ===
    'obv_window': [20, 30],

    # === FUNDAMENTAL INDICATOR PARAMETERS ===
    'pe_theta_minus': [10, 12],  # Low P/E (undervalued)
    'pe_theta_plus': [20, 25],  # High P/E (overvalued)
    'surprise_theta_minus': [-5],  # Negative surprise
    'surprise_theta_plus': [5],  # Positive surprise
}

# Generate all possible combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Generated {len(param_combinations)} parameter combinations for testing.")

# Filter out invalid combinations with enhanced constraints
valid_combinations = []
for params in param_combinations:
    # Constraint 1: SMA windows
    if params['sma_short_window'] >= params['sma_long_window']:
        continue

    # Constraint 2: EMA windows
    if params['ema_short_window'] >= params['ema_long_window']:
        continue

    # Constraint 3: RSI thresholds
    if params['rsi_theta_minus'] >= params['rsi_theta_plus']:
        continue

    # Constraint 4: P/E thresholds
    if params['pe_theta_minus'] >= params['pe_theta_plus']:
        continue

    # Constraint 5: Earnings surprise thresholds
    if params['surprise_theta_minus'] >= params['surprise_theta_plus']:
        continue

    valid_combinations.append(params)

print(f"Valid parameter combinations: {len(valid_combinations)}")
print(f"Starting comprehensive optimization...")

# --- Run Enhanced Optimization on Training Data ---
results = []
best_sharpe = -np.inf
best_params = None

for i, params in enumerate(valid_combinations):
    try:
        if 'improved_simulator' in sys.modules:
            sharpe, portfolio_values, trade_log = run_improved_simulation(train_df, params)
        else:
            # Fallback to basic simulator with parameter mapping
            basic_params = {
                'sma_short': params.get('sma_short_window', 20),
                'sma_long': params.get('sma_long_window', 100),
                'rsi_window': params.get('rsi_window', 14),
                'rsi_buy': params.get('rsi_theta_minus', 30),
                'rsi_sell': params.get('rsi_theta_plus', 70),
                'macd_fast': params.get('macd_fast', 12),
                'macd_slow': params.get('macd_slow', 26),
                'macd_signal': params.get('macd_signal', 9),
                'bb_window': params.get('bb_window', 20),
                'bb_std_dev': params.get('bb_std_dev', 2.0),
                'obv_window': params.get('obv_window', 20),
                'lambda_worst': params.get('lambda_worst', 1.5),
            }
            sharpe, portfolio_values = run_improved_simulation(train_df, basic_params)
            trade_log = []

        results.append({
            'params': params,
            'sharpe_ratio': sharpe,
            'final_value': portfolio_values.iloc[-1],
            'total_trades': len(trade_log) if trade_log else 0
        })

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params

        # Print progress for every combination
        print(f"Processed {i + 1}/{len(valid_combinations)} | "
              f"Sharpe: {sharpe:.4f} | Best: {best_sharpe:.4f} | "
              f"Trades: {len(trade_log) if trade_log else 'N/A'}")

    except Exception as e:
        print(f"Error in combination {i + 1}: {e}")
        continue

print("\n" + "=" * 60)
print("COMPREHENSIVE OPTIMIZATION RESULTS")
print("=" * 60)

if best_params is not None:
    print(f"Best Sharpe Ratio on Training Data: {best_sharpe:.4f}")
    print(f"\nOptimal Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Parameter sensitivity analysis
    if len(results) > 5:
        print(f"\nParameter Sensitivity Analysis:")
        print("-" * 40)

        # Analyze top 5 performing parameter sets
        sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)[:5]
        print("Top 5 Parameter Combinations:")
        for i, result in enumerate(sorted_results):
            print(f"#{i + 1}: Sharpe = {result['sharpe_ratio']:.4f}")
else:
    print("No profitable strategy found in the training phase.")
    print("All parameter combinations resulted in negative or zero Sharpe ratios.")

# --- Evaluate Best Parameters on Test Data ---
if best_params is not None:
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    try:
        if 'improved_simulator' in sys.modules:
            test_sharpe, test_portfolio_values, test_trade_log = run_improved_simulation(test_df, best_params)
        else:
            # Map to basic simulator parameters
            basic_params = {
                'sma_short': best_params.get('sma_short_window', 20),
                'sma_long': best_params.get('sma_long_window', 100),
                'rsi_window': best_params.get('rsi_window', 14),
                'rsi_buy': best_params.get('rsi_theta_minus', 30),
                'rsi_sell': best_params.get('rsi_theta_plus', 70),
                'macd_fast': best_params.get('macd_fast', 12),
                'macd_slow': best_params.get('macd_slow', 26),
                'macd_signal': best_params.get('macd_signal', 9),
                'bb_window': best_params.get('bb_window', 20),
                'bb_std_dev': best_params.get('bb_std_dev', 2.0),
                'obv_window': best_params.get('obv_window', 20),
                'lambda_worst': best_params.get('lambda_worst', 1.5),
            }
            test_sharpe, test_portfolio_values = run_improved_simulation(test_df, basic_params)
            test_trade_log = []

        print(f"Sharpe Ratio on Test Data: {test_sharpe:.4f}")

        # Calculate enhanced performance metrics
        test_cumulative_return = (test_portfolio_values.iloc[-1] / test_portfolio_values.iloc[0] - 1) * 100
        test_returns = test_portfolio_values.pct_change().dropna()
        test_volatility = test_returns.std() * np.sqrt(252) * 100
        max_drawdown = ((
                                    test_portfolio_values.cummax() - test_portfolio_values) / test_portfolio_values.cummax()).max() * 100

        # Additional metrics if trade log available
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_duration = 0.0

        if test_trade_log:
            # Simple trade analysis
            profitable_trades = 0
            total_profit = 0
            total_loss = 0

            for trade in test_trade_log:
                if 'profit' in trade:
                    if trade['profit'] > 0:
                        profitable_trades += 1
                        total_profit += trade['profit']
                    else:
                        total_loss += abs(trade['profit'])

            if len(test_trade_log) > 0:
                win_rate = (profitable_trades / len(test_trade_log)) * 100
            if total_loss > 0:
                profit_factor = total_profit / total_loss

        print(f"Cumulative Return on Test Data: {test_cumulative_return:.2f}%")
        print(f"Annualized Volatility: {test_volatility:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {len(test_trade_log)}")
        if test_trade_log:
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Profit Factor: {profit_factor:.2f}")

        print("\n" + "=" * 60)
        print("ENHANCED PERFORMANCE SUMMARY")
        print("=" * 60)

        # Enhanced results table
        results_summary = {
            'Metric': [
                'Training Sharpe Ratio',
                'Test Sharpe Ratio',
                'Cumulative Return (%)',
                'Annualized Volatility (%)',
                'Maximum Drawdown (%)',
                'Total Trades',
                'Win Rate (%)',
                'Profit Factor'
            ],
            'Value': [
                f"{best_sharpe:.4f}",
                f"{test_sharpe:.4f}",
                f"{test_cumulative_return:.2f}",
                f"{test_volatility:.2f}",
                f"{max_drawdown:.2f}",
                f"{len(test_trade_log)}",
                f"{win_rate:.2f}",
                f"{profit_factor:.2f}"
            ]
        }
        results_df = pd.DataFrame(results_summary)
        print(results_df.to_string(index=False))

        # Enhanced plotting
        plt.style.use('dark_background')

        # Plot 1: Portfolio Performance with Enhanced Details
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Portfolio value over time
        ax1.plot(test_portfolio_values, linewidth=2, color='cyan')
        ax1.set_title(f'Portfolio Value Over Time (Test Set)\n'
                      f'Sharpe: {test_sharpe:.3f} | Return: {test_cumulative_return:.2f}% | '
                      f'Max Drawdown: {max_drawdown:.2f}% | Trades: {len(test_trade_log)}', fontsize=14)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.ticklabel_format(style='plain', axis='y')

        # Stock price with indicators (if available)
        price_col = best_params.get('price_column', 'MSFT_close')
        ax2.plot(test_df.index, test_df[price_col], linewidth=1, color='white', alpha=0.8, label='Stock Price')

        # Add moving averages if we can calculate them
        try:
            from indicators import calculate_sma

            sma_short = calculate_sma(test_df[price_col], best_params.get('sma_short_window', 20))
            sma_long = calculate_sma(test_df[price_col], best_params.get('sma_long_window', 100))
            ax2.plot(test_df.index, sma_short, linewidth=1, color='orange', alpha=0.7,
                     label=f'SMA {best_params.get("sma_short_window", 20)}')
            ax2.plot(test_df.index, sma_long, linewidth=1, color='red', alpha=0.7,
                     label=f'SMA {best_params.get("sma_long_window", 100)}')
        except:
            pass

        ax2.set_title('Stock Price with Technical Indicators', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot 2: Returns Distribution with Enhanced Statistics
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(test_returns * 100, bins=50, alpha=0.7, color='lightblue', edgecolor='white')
        plt.title('Distribution of Daily Returns', fontsize=14)
        plt.xlabel('Daily Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.axvline(test_returns.mean() * 100, color='red', linestyle='--',
                    label=f'Mean: {test_returns.mean() * 100:.3f}%')
        plt.axvline(test_returns.median() * 100, color='orange', linestyle='--',
                    label=f'Median: {test_returns.median() * 100:.3f}%')
        plt.legend()

        # Drawdown plot
        plt.subplot(1, 2, 2)
        rolling_max = test_portfolio_values.expanding().max()
        drawdown_series = (rolling_max - test_portfolio_values) / rolling_max * 100
        plt.fill_between(test_portfolio_values.index, 0, drawdown_series, alpha=0.7, color='red')
        plt.title('Drawdown Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during test evaluation: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 60)
print("ENHANCED ANALYSIS COMPLETE")
print("=" * 60)
print("\nImplementation Features:")
print("✓ Technical Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, OBV")
print("✓ Fundamental Indicators: P/E Ratio, Earnings Surprise")
print("✓ Enhanced Parameter Optimization with Constraints")
print("✓ Comprehensive Performance Metrics")
print("✓ Advanced Visualization with Technical Indicators")
print("✓ Robust Error Handling and Fallback Options")
print("✓ Trade Analysis and Win Rate Calculation")
print("✓ Parameter Sensitivity Analysis")