import pandas as pd
import pickle
import os


def prepare_daily_data(pickle_path='data/collected_data.pkl', output_csv='data/daily_data.csv'):
    """
    Loads raw data, extracts daily close and volume for all symbols,
    synchronizes them to a common index, and saves to a CSV file.

    Args:
        pickle_path (str): Path to the input pickle file.
        output_csv (str): Path to save the final CSV file.
    """
    print("Starting data preparation...")
    # Load the collected data from the pickle file
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"Pickle file not found at '{pickle_path}'. Please run TimeSeriesDataCollection.py first.")

    with open(pickle_path, "rb") as f:
        collected_data = pickle.load(f)

    # Dictionary to hold the relevant series (close and volume) for each symbol
    all_series = {}

    # Extract 'close' and 'volume' for each symbol
    for group, group_data in collected_data.items():
        for symbol, df in group_data.items():
            if 'close' in df.columns:
                all_series[f'{symbol}_close'] = df['close']
            if 'volume' in df.columns:
                all_series[f'{symbol}_volume'] = df['volume']

    # Create a unified DataFrame from all the extracted series
    main_df = pd.DataFrame(all_series)

    # The index is already a DatetimeIndex. Let's ensure it's sorted.
    main_df.sort_index(inplace=True)

    # Forward-fill missing values to handle non-trading days
    main_df.ffill(inplace=True)
    # Backward-fill any remaining NaNs at the beginning
    main_df.bfill(inplace=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save the prepared data to a new CSV file
    main_df.to_csv(output_csv)
    print(f"Data preparation complete. Clean data saved to '{output_csv}'")


if __name__ == '__main__':
    # You can run this script directly after running TimeSeriesDataCollection.py
    prepare_daily_data()
