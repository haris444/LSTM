# fundamental_indicators.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def calculate_pe_ratio(price_series, earnings_data, reporting_dates):
    """
    Calculate P/E ratio using most recent earnings data.

    Args:
        price_series: Daily stock prices
        earnings_data: Series of EPS values
        reporting_dates: Dates when earnings were reported

    Returns:
        Series of P/E ratios
    """
    pe_ratios = pd.Series(index=price_series.index, dtype=float)

    for i, date in enumerate(price_series.index):
        # Find most recent earnings report
        prior_reports = [d for d in reporting_dates if d <= date]
        if prior_reports:
            latest_report_date = max(prior_reports)
            latest_eps = earnings_data[reporting_dates.index(latest_report_date)]

            if latest_eps > 0:  # Avoid division by zero
                pe_ratios.iloc[i] = price_series.iloc[i] / latest_eps
            else:
                pe_ratios.iloc[i] = np.nan
        else:
            pe_ratios.iloc[i] = np.nan

    # Forward fill NaN values
    pe_ratios.fillna(method='ffill', inplace=True)
    return pe_ratios


def estimate_earnings_with_model(historical_earnings, forecast_periods=4):
    """
    Estimate future earnings using a simple linear regression model.

    Args:
        historical_earnings: Past EPS values
        forecast_periods: Number of periods to forecast

    Returns:
        Forecasted EPS values
    """
    if len(historical_earnings) < 4:
        # Not enough data, return simple trend
        return [historical_earnings[-1]] * forecast_periods

    # Prepare data for regression
    X = np.arange(len(historical_earnings)).reshape(-1, 1)
    y = np.array(historical_earnings)

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Generate forecasts
    future_X = np.arange(len(historical_earnings),
                         len(historical_earnings) + forecast_periods).reshape(-1, 1)
    forecasts = model.predict(future_X)

    return forecasts.tolist()


def calculate_earnings_surprise(actual_earnings, expected_earnings):
    """
    Calculate earnings surprise as percentage deviation from expected.

    Args:
        actual_earnings: Actual reported EPS
        expected_earnings: Model-predicted EPS

    Returns:
        Earnings surprise percentage
    """
    if expected_earnings == 0:
        return 0.0

    surprise = (actual_earnings - expected_earnings) / abs(expected_earnings)
    return surprise * 100  # Return as percentage


def generate_synthetic_earnings_data(price_series, noise_level=0.1):
    """
    Generate synthetic earnings data for demonstration purposes.
    In real implementation, this would come from financial data providers.

    Args:
        price_series: Stock price series
        noise_level: Amount of randomness in earnings

    Returns:
        Dictionary with earnings data and reporting dates
    """
    # Generate quarterly reporting dates
    start_date = price_series.index[0]
    end_date = price_series.index[-1]

    reporting_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq='Q'  # Quarterly
    ).tolist()

    # Generate synthetic EPS data correlated with price trends
    earnings_data = []
    base_eps = 2.0  # Starting EPS

    for i, report_date in enumerate(reporting_dates):
        # Find corresponding price
        closest_price_idx = price_series.index.get_indexer([report_date], method='nearest')[0]
        price_at_report = price_series.iloc[closest_price_idx]

        # Generate EPS with some correlation to price and noise
        if i == 0:
            eps = base_eps
        else:
            price_change = price_at_report / price_series.iloc[0]
            eps = base_eps * price_change * (1 + np.random.normal(0, noise_level))

        earnings_data.append(max(eps, 0.1))  # Ensure positive earnings

    return {
        'earnings_data': earnings_data,
        'reporting_dates': reporting_dates
    }


# Example usage and integration
def add_fundamental_indicators_to_data(df, price_column='MSFT_close'):
    """
    Add fundamental indicators to the main dataset.

    Args:
        df: Main dataset DataFrame
        price_column: Name of the price column

    Returns:
        DataFrame with added fundamental indicators
    """
    # Generate synthetic earnings data (replace with real data in production)
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
    expected_earnings_list = []

    for i, report_date in enumerate(earnings_info['reporting_dates']):
        if i >= 4:  # Need at least 4 historical points for forecasting
            historical = earnings_info['earnings_data'][:i]
            expected = estimate_earnings_with_model(historical, 1)[0]
            actual = earnings_info['earnings_data'][i]
            surprise = calculate_earnings_surprise(actual, expected)

            expected_earnings_list.append(expected)
            earnings_surprises.append(surprise)
        else:
            expected_earnings_list.append(np.nan)
            earnings_surprises.append(np.nan)

    # Map surprises to daily data (forward fill from reporting dates)
    surprise_series = pd.Series(index=df.index, dtype=float)
    for i, report_date in enumerate(earnings_info['reporting_dates']):
        closest_idx = df.index.get_indexer([report_date], method='nearest')[0]
        if i < len(earnings_surprises):
            surprise_series.iloc[closest_idx:] = earnings_surprises[i]

    surprise_series.fillna(method='ffill', inplace=True)
    surprise_series.fillna(0, inplace=True)  # Fill remaining NaNs with 0

    df['Earnings_Surprise'] = surprise_series

    return df