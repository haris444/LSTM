# fundamental_indicators.py - Fixed for Question 1 requirements
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_pe_ratio(price_series, earnings_data):
    """
    Calculate P/E ratio using real, historical earnings data.
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


def add_fundamental_indicators_to_data(df, earnings_df, price_column='MSFT_close'):
    """
    Add fundamental indicators to the main dataset using real earnings data.
    FIXED: Earnings Surprise is now truly discrete for Question 1 requirements.
    """
    if earnings_df.empty:
        print("Warning: Earnings data is empty. Skipping fundamental indicators.")
        df['PE_Ratio'] = np.nan
        df['Earnings_Surprise'] = 0
        return df

    # 1. Calculate P/E Ratio (unchanged - this can be forward-filled)
    df['PE_Ratio'] = calculate_pe_ratio(df[price_column], earnings_df)

    # 2. Calculate Earnings Surprise - TRULY DISCRETE FOR Q1
    historical_eps = earnings_df['eps'].tolist()
    reporting_dates = earnings_df['date'].tolist()

    # Initialize ALL dates with 0 (neutral signal) - this is the key difference
    surprise_series = pd.Series(index=df.index, dtype=float, data=0.0)

    # Only set surprise values on actual earnings announcement dates
    # NO forward-filling or persistence - signal is only active on announcement day
    for i in range(len(historical_eps)):
        if i >= 4:  # Need enough data for forecast
            expected_eps = estimate_earnings_with_model(historical_eps[:i])[0]
            actual_eps = historical_eps[i]
            surprise = calculate_earnings_surprise(actual_eps, expected_eps)

            # Find the exact date or closest date in our dataset
            report_date = reporting_dates[i]
            if report_date in df.index:
                # Exact match - ONLY this date gets the signal
                surprise_series.loc[report_date] = surprise
            else:
                # Find closest trading day - ONLY this day gets the signal
                closest_idx = df.index.get_indexer([report_date], method='nearest')[0]
                closest_date = df.index[closest_idx]
                surprise_series.loc[closest_date] = surprise

    # CRITICAL: Do NOT forward-fill for Question 1
    # The signal should be 0 on all days except announcement days
    df['Earnings_Surprise'] = surprise_series

    print("Successfully added P/E Ratio and truly discrete Earnings Surprise for Q1.")
    print(f"Total trading days: {len(df)}")
    print(f"Days with earnings announcements (surprise != 0): {(surprise_series != 0).sum()}")
    print(f"Percentage of days with earnings signals: {(surprise_series != 0).sum() / len(df) * 100:.2f}%")

    return df


# Additional function for Question 2 where forward-filling is allowed
def add_fundamental_indicators_to_data_q2(df, earnings_df, price_column='MSFT_close'):
    """
    Version for Question 2 where forward-filling earnings surprise is acceptable.
    """
    # Same as above, but with forward-filling
    df = add_fundamental_indicators_to_data(df, earnings_df, price_column)

    # For Q2: Forward-fill the earnings surprise signal
    df['Earnings_Surprise'] = df['Earnings_Surprise'].replace(0, np.nan).ffill().fillna(0)

    print("Applied forward-filling for Question 2 implementation.")
    return df