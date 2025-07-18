# download_data.py (Respecting the 730-day limit for H1 data)

import yfinance as yf
import pandas as pd
import config

def download_reliable_historical_data():
    """
    Downloads reliable H1 historical data for Gold using the yfinance library
    and saves it to the location specified in the config.
    """
    print("--- Starting Reliable H1 Data Download using yfinance ---")

    symbol = "GC=F"
    
    # --- ### THE FIX IS HERE ### ---
    # Change the period from "3y" to "730d" to respect the API limit.
    period = "730d" 
    interval = "1h"

    print(f"Fetching {period} of {interval} data for Gold ({symbol})...")
    
    # Download the data
    df = yf.download(tickers=symbol, period=period, interval=interval, progress=True)

    if df.empty:
        print(f"❌ No data downloaded for {symbol}. Check the ticker symbol and your internet connection.")
        return

    # --- Data Cleaning ---
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'tick_volume'
    }, inplace=True)

    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'time'}, inplace=True)
    
    if 'Timezone' in df.columns:
        df.drop(columns=['Timezone'], inplace=True)
        
    if pd.api.types.is_datetime64_any_dtype(df['time']) and df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)

    columns_to_save = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    final_df = df[columns_to_save].copy()
    
    final_df.dropna(inplace=True)

    print(f"\n✅ Successfully downloaded a total of {len(final_df)} unique data points.")

    try:
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(config.RAW_DATA_FILE, index=False)
        print(f"✅ H1 Data successfully saved to: {config.RAW_DATA_FILE}")
    except Exception as e:
        print(f"❌ Failed to save data. Error: {e}")

if __name__ == "__main__":
    if config.RAW_DATA_FILE.exists():
        print(f"Deleting old data file: {config.RAW_DATA_FILE}")
        config.RAW_DATA_FILE.unlink()
    
    download_reliable_historical_data()