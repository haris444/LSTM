# expanded_parameter_optimization.py
import itertools
import numpy as np


def create_comprehensive_parameter_grid():
    """
    Create a comprehensive parameter grid that includes all indicators
    and their thresholds as required by the assignment.

    Returns:
        dict: Parameter grid with all required parameters
    """

    param_grid = {
        # === BASIC SIMULATION PARAMETERS ===
        'initial_capital': [100000.0],
        'transaction_fee': [5.0],
        'lambda_worst': [1.5, 2.0],  # Short position risk parameter
        'price_column': ['MSFT_close'],
        'volume_column': ['MSFT_volume'],
        'aggregation_method': ['majority_vote'],  # Could add 'weighted_sum'

        # === MOVING AVERAGE PARAMETERS ===
        # SMA windows
        'sma_short_window': [5, 10, 20],
        'sma_long_window': [50, 100, 200],

        # EMA windows
        'ema_short_window': [5, 12, 20],
        'ema_long_window': [26, 50, 100],

        # === RSI PARAMETERS ===
        'rsi_window': [7, 14, 21, 28],
        # RSI thresholds (traditional values around 30/70)
        'rsi_theta_minus': [20, 25, 30, 35],  # Oversold threshold (buy signal)
        'rsi_theta_plus': [65, 70, 75, 80],  # Overbought threshold (sell signal)

        # === MACD PARAMETERS ===
        'macd_fast': [12],  # Standard MACD fast period
        'macd_slow': [26],  # Standard MACD slow period
        'macd_signal': [9],  # Standard MACD signal period

        # === BOLLINGER BANDS PARAMETERS ===
        'bb_window': [20, 30],
        'bb_std_dev': [1.5, 2.0, 2.5],

        # === OBV PARAMETERS ===
        'obv_window': [10, 20, 30],  # Window for OBV smoothing

        # === FUNDAMENTAL INDICATOR PARAMETERS ===
        # P/E Ratio thresholds
        'pe_theta_minus': [8, 10, 12],  # Low P/E (undervalued, potential buy)
        'pe_theta_plus': [20, 25, 30],  # High P/E (overvalued, potential sell)

        # Earnings Surprise thresholds (in percentage)
        'surprise_theta_minus': [-10, -5, -2],  # Negative surprise (sell signal)
        'surprise_theta_plus': [2, 5, 10],  # Positive surprise (buy signal)
    }

    return param_grid


def generate_valid_parameter_combinations(param_grid, max_combinations=None):
    """
    Generate all valid parameter combinations with constraint checking.

    Args:
        param_grid: Dictionary of parameter lists
        max_combinations: Optional limit on number of combinations

    Returns:
        List of valid parameter dictionaries
    """
    # Get all parameter names and their possible values
    keys, values = zip(*param_grid.items())

    # Generate all combinations
    all_combinations = list(itertools.product(*values))

    # Filter valid combinations
    valid_combinations = []

    for combination in all_combinations:
        params = dict(zip(keys, combination))

        # Constraint 1: Short MA window must be less than long MA window
        if params['sma_short_window'] >= params['sma_long_window']:
            continue

        if params['ema_short_window'] >= params['ema_long_window']:
            continue

        # Constraint 2: RSI thresholds must be properly ordered
        if params['rsi_theta_minus'] >= params['rsi_theta_plus']:
            continue

        # Constraint 3: P/E thresholds must be properly ordered
        if params['pe_theta_minus'] >= params['pe_theta_plus']:
            continue

        # Constraint 4: Earnings surprise thresholds must be properly ordered
        if params['surprise_theta_minus'] >= params['surprise_theta_plus']:
            continue

        valid_combinations.append(params)

        # Limit combinations if specified
        if max_combinations and len(valid_combinations) >= max_combinations:
            break

    return valid_combinations


def create_reduced_parameter_grid_for_testing():
    """
    Create a smaller parameter grid for faster testing/debugging.

    Returns:
        dict: Reduced parameter grid
    """

    param_grid = {
        # Basic parameters
        'initial_capital': [100000.0],
        'transaction_fee': [5.0],
        'lambda_worst': [1.5],
        'price_column': ['MSFT_close'],
        'volume_column': ['MSFT_volume'],
        'aggregation_method': ['majority_vote'],

        # Reduced technical indicator parameters
        'sma_short_window': [10, 20],
        'sma_long_window': [50, 100],
        'ema_short_window': [12, 20],
        'ema_long_window': [26, 50],

        # RSI parameters
        'rsi_window': [14, 21],
        'rsi_theta_minus': [25, 30],
        'rsi_theta_plus': [70, 75],

        # MACD parameters (standard)
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],

        # Bollinger Bands
        'bb_window': [20],
        'bb_std_dev': [2.0],

        # OBV
        'obv_window': [20],

        # Fundamental indicators
        'pe_theta_minus': [10, 12],
        'pe_theta_plus': [20, 25],
        'surprise_theta_minus': [-5],
        'surprise_theta_plus': [5],
    }

    return param_grid


def optimize_parameters_with_progress(data, param_combinations, simulation_func, train_data):
    """
    Run parameter optimization with progress tracking and error handling.

    Args:
        data: Full dataset
        param_combinations: List of parameter dictionaries to test
        simulation_func: Function to run simulation (e.g., run_improved_simulation)
        train_data: Training subset of data

    Returns:
        tuple: (results_list, best_params, best_sharpe)
    """

    results = []
    best_sharpe = -np.inf
    best_params = None

    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")

    for i, params in enumerate(param_combinations):
        try:
            # Run simulation on training data
            sharpe, portfolio_values, trade_log = simulation_func(train_data, params)

            # Store results
            result = {
                'params': params.copy(),
                'sharpe_ratio': sharpe,
                'final_portfolio_value': portfolio_values.iloc[-1],
                'total_trades': len(trade_log),
                'combination_index': i
            }
            results.append(result)

            # Update best parameters
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()

            # Progress reporting
            if (i + 1) % 50 == 0 or (i + 1) == total_combinations:
                print(f"Progress: {i + 1}/{total_combinations} | "
                      f"Current Sharpe: {sharpe:.4f} | "
                      f"Best Sharpe: {best_sharpe:.4f}")

        except Exception as e:
            print(f"Error in combination {i + 1}: {str(e)}")
            continue

    return results, best_params, best_sharpe


def analyze_parameter_sensitivity(results):
    """
    Analyze which parameters have the most impact on performance.

    Args:
        results: List of optimization results

    Returns:
        DataFrame with parameter sensitivity analysis
    """
    import pandas as pd

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Extract parameters into separate columns
    param_cols = {}
    for key in results[0]['params'].keys():
        param_cols[key] = [r['params'][key] for r in results]

    df_params = pd.DataFrame(param_cols)
    df_combined = pd.concat([df_params, df_results[['sharpe_ratio']]], axis=1)

    # Calculate correlation between each parameter and Sharpe ratio
    correlations = {}
    for col in df_params.columns:
        if df_params[col].dtype in ['int64', 'float64']:  # Only numeric parameters
            corr = df_combined[col].corr(df_combined['sharpe_ratio'])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True)

    print("\nParameter Sensitivity Analysis (Correlation with Sharpe Ratio):")
    print("=" * 60)
    for param, corr in sorted_correlations:
        print(f"{param:25s}: {corr:8.4f}")

    return df_combined, sorted_correlations


# Example usage:
if __name__ == "__main__":
    # For testing, use reduced parameter grid
    test_grid = create_reduced_parameter_grid_for_testing()
    test_combinations = generate_valid_parameter_combinations(test_grid)

    print(f"Generated {len(test_combinations)} test parameter combinations")

    # For full optimization, use comprehensive grid
    # full_grid = create_comprehensive_parameter_grid()
    # full_combinations = generate_valid_parameter_combinations(full_grid, max_combinations=1000)
    # print(f"Generated {len(full_combinations)} full parameter combinations")