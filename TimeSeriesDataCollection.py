# =============================================================================
# This script file provides fundamental computational functionality for the 
# implementation of Case Study I: Supervised Learning Models for Stock Market
# Prediction.
# =============================================================================
# Import required Python modules.
import requests
import datetime
import pandas as pd
import time
import pickle
import os

# Set the Twelve Data and Alpha Vantage API Keys
TWELVE_DATA_API_KEY = "05b22eb6c3c449379f45ac6571f5a4b3"
# ----> ALPHA VANTAGE KEY <----
ALPHA_VANTAGE_API_KEY = "T5JRS9Q6QUY44LHK"


# =============================================================================
#                   FUNCTIONS DEFINITION SECTION:
# =============================================================================

def check_rate_limit(request_timestamps, max_requests=8, window_sec=60):
    now = time.time()
    pruned_timestamps = [t for t in request_timestamps if (now - t) < window_sec]
    used = len(pruned_timestamps)
    left = max_requests - used

    if used >= max_requests:
        oldest_in_window = pruned_timestamps[0]
        wait_time = int(window_sec - (now - oldest_in_window)) + 1
        raise RuntimeError(
            f"Rate limit exceeded: {max_requests} requests in the last {window_sec}s.\n"
            f"Please wait ~{wait_time} seconds before trying again."
        )

    return pruned_timestamps, used, left


def fetch_twelvedata(symbol, start_dt, end_dt, interval='1day', apikey=TWELVE_DATA_API_KEY):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": interval,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "apikey": apikey, "outputsize": 5000
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    if "status" in data and data["status"] == "error":
        raise ValueError(f"Error from Twelve Data API for '{symbol}': {data.get('message', 'Unknown error')}")
    if "values" not in data:
        raise ValueError(f"No 'values' found for symbol '{symbol}'. Response: {data}")

    df = pd.DataFrame(data["values"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' in response for symbol '{symbol}'.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    return df


# ----> NEW FUNCTION TO FETCH EARNINGS DATA <----
def fetch_earnings_data(symbol, apikey=ALPHA_VANTAGE_API_KEY):
    """
    Fetches historical quarterly earnings data from Alpha Vantage.
    """
    print(f"Fetching earnings data for {symbol} from Alpha Vantage...")
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={apikey}'
    r = requests.get(url)
    data = r.json()

    if "quarterlyEarnings" not in data:
        print(f"Warning: Could not find earnings data for {symbol}. Response: {data}")
        return pd.DataFrame()

    earnings_df = pd.DataFrame(data['quarterlyEarnings'])
    earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])
    earnings_df.rename(columns={'fiscalDateEnding': 'date', 'reportedEPS': 'eps'}, inplace=True)

    # Convert EPS to numeric, coercing errors to NaN
    earnings_df['eps'] = pd.to_numeric(earnings_df['eps'], errors='coerce')

    # Drop rows with NaN EPS values and sort by date
    earnings_df.dropna(subset=['eps'], inplace=True)
    earnings_df.sort_values('date', inplace=True)

    print(f"Successfully fetched {len(earnings_df)} earnings reports for {symbol}.")
    return earnings_df[['date', 'eps']]


def download_dataset(td_symbol_groups, apikey=TWELVE_DATA_API_KEY):
    try:
        collected_data = {}
        request_timestamps = []

        for group_name, symbols in td_symbol_groups.items():
            print(f"\n--- Downloading {group_name} ---")
            group_data = {}
            for symbol in symbols:
                request_timestamps, used, left = check_rate_limit(request_timestamps)
                print(f"Request usage in last 60s: used={used}, left={left}")
                try:
                    df = fetch_twelvedata(symbol, start_date, end_date, interval, apikey)
                    group_data[symbol] = df
                    print(f"Downloaded {symbol}, rows={len(df)}")
                except Exception as ex:
                    print(f"Error downloading {symbol}: {ex}")
                request_timestamps.append(time.time())
            collected_data[group_name] = group_data

        # ----> ADDED: FETCH AND STORE EARNINGS DATA FOR MSFT <----
        msft_earnings = fetch_earnings_data("MSFT")
        if not msft_earnings.empty:
            collected_data['fundamental'] = {'MSFT_EARNINGS': msft_earnings}

        print("\nAll downloads finished.\n")
        return collected_data
    except RuntimeError as e:
        print(f"\n** Rate Limit Error **\n{e}")


# =============================================================================
#                   MAIN CODE SECTION:
# =============================================================================

start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2024, 1, 1)
interval = '1day'

td_symbol_groups = {
    "stocks": ["MSFT", "IBM", "GOOGL"],
    "currencies": ["USD/JPY", "GBP/USD"],
    "indices": ["SPY", "DIA", "QQQ"]
}

collected_data = download_dataset(td_symbol_groups, apikey=TWELVE_DATA_API_KEY)

# Save the collected data
data_directory = './data'
data_file = 'collected_data.pkl'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
save_path = os.path.join(data_directory, data_file)

with open(save_path, "wb") as f:
    pickle.dump(collected_data, f)
print(f"Data successfully saved to '{save_path}'")