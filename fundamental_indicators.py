# fundamental_indicators.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_pe_ratio(price_series, earnings_data):
    """
    Calculate P/E ratio using real, historical earnings data.

    Args:
        price_series: Daily stock prices.
        earnings_data: DataFrame with 'date' and 'eps' columns.

    Returns:
        Series of P/E ratios.
    """
    pe_ratios = pd.Series(index=price_series.index, dtype=float)
    reporting_dates = earnings_data['date'].tolist()
    eps_values = earnings_data['eps'].tolist()

    for i, date in enumerate(price_series.index):
        # Find most recent earnings report
        prior_reports_dates = [d for d in reporting_dates if d <= date]
        if prior_reports_dates:
            latest_report_date = max(prior_reports_dates)
            latest_eps = eps_values[reporting_dates.index(latest_report_date)]

            if latest_eps > 0:
                pe_ratios.iloc[i] = price_series.iloc[i] / latest_eps
            else:
                pe_ratios.iloc[i] = np.nan
        else:
            pe_ratios.iloc[i] = np.nan

    pe_ratios.ffill(inplace=True)
    return pe_ratios


def estimate_earnings_with_model(historical_earnings, forecast_periods=1):
    """
    Estimate future earnings using linear regression on historical earnings.
    """
    if len(historical_earnings) < 4:
        return [historical_earnings[-1]] * forecast_periods if historical_earnings else [0]

    X = np.arange(len(historical_earnings)).reshape(-1, 1)
    y = np.array(historical_earnings)

    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(historical_earnings), len(historical_earnings) + forecast_periods).reshape(-1, 1)
    return model.predict(future_X)


def calculate_earnings_surprise(actual_earnings, expected_earnings):
    """
    Calculate earnings surprise as a percentage.
    """
    if expected_earnings == 0:
        return 0.0
    return ((actual_earnings - expected_earnings) / abs(expected_earnings)) * 100


# ----> MAJOR CHANGE: NO MORE SYNTHETIC DATA <----
def add_fundamental_indicators_to_data(df, earnings_df, price_column='MSFT_close'):
    """
    Add fundamental indicators to the main dataset using real earnings data.

    Args:
        df: Main DataFrame with price/volume data.
        earnings_df: DataFrame containing the real historical earnings data.
        price_column: The name of the price column.

    Returns:
        DataFrame with added fundamental indicators.
    """
    if earnings_df.empty:
        print("Warning: Earnings data is empty. Skipping fundamental indicators.")
        df['PE_Ratio'] = np.nan
        df['Earnings_Surprise'] = 0
        return df

    # 1. Calculate P/E Ratio
    df['PE_Ratio'] = calculate_pe_ratio(df[price_column], earnings_df)

    # 2. Calculate Earnings Surprise
    historical_eps = earnings_df['eps'].tolist()
    reporting_dates = earnings_df['date'].tolist()
    surprises = []

    for i in range(len(historical_eps)):
        if i >= 4:  # Need a few data points to make a forecast
            expected_eps = estimate_earnings_with_model(historical_eps[:i])[0]
            actual_eps = historical_eps[i]
            surprise = calculate_earnings_surprise(actual_eps, expected_eps)
            surprises.append(surprise)
        else:
            surprises.append(0)  # Not enough data for a surprise calc

    # Map the surprises to the daily dataframe
    surprise_series = pd.Series(index=df.index, dtype=float, data=0.0)
    for i, report_date in enumerate(reporting_dates):
        # Get the index in the daily data that is closest to the report date
        closest_idx = df.index.get_indexer([report_date], method='nearest')[0]
        # Forward-fill the surprise value from that date onwards
        if i < len(surprises):
            surprise_series.iloc[closest_idx:] = surprises[i]

    df['Earnings_Surprise'] = surprise_series

    print("Successfully added P/E Ratio and Earnings Surprise from real data.")
    return df