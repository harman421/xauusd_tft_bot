# src/data_processing.py

import pandas as pd
import numpy as np
import pandas_ta as ta

def load_and_clean_data(file_path):
    """
    Loads raw data, cleans it, and resamples it to a perfect hourly frequency
    using a robust manual reindexing method.
    """
    df = pd.read_csv(file_path)
    
    # Force all expected numeric columns to numeric types.
    # 'coerce' will turn any non-numeric garbage data into NaN (Not a Number)
    cols_to_numeric = ['open', 'high', 'low', 'close', 'tick_volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop any rows that now have NaN values because of bad data
    df.dropna(subset=cols_to_numeric, inplace=True)
    
    df['time'] = pd.to_datetime(df['time'])

    # Filter out any data from the future, just in case the source file is strange.
    df = df[df['time'] <= pd.Timestamp.now()]
    
    df.set_index('time', inplace=True)
    df.rename(columns={
        'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'
    }, inplace=True)
    
    # --- DATA CLEANING PIPELINE ---
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # --- Robust Resampling Method ---
    start_time = df.index.min().floor('h')
    end_time = df.index.max().ceil('h')
    perfect_hourly_index = pd.date_range(start=start_time, end=end_time, freq='h')
    
    df_reindexed = df.reindex(perfect_hourly_index)
    df_reindexed.ffill(inplace=True)
    df_reindexed.dropna(inplace=True)
    
    return df_reindexed


def create_features(df):
    """
    Engineers a sophisticated set of features for the TFT model,
    treating indicators as parallel time-series with explicit column assignment.
    """
    df_copy = df.copy()

    # --- 1. Core Time-Based Features ---
    df_copy['time_idx'] = (df_copy.index.astype(np.int64) // 10**9 // 3600).astype(int)
    df_copy['time_idx'] -= df_copy['time_idx'].min()
    df_copy['hour'] = df_copy.index.hour
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    
    # --- 2. Calculate Indicators with Explicit Assignment (More Robust) ---
    df_copy['hma'] = ta.hma(df_copy['Close'], length=14)
    df_copy['rsi'] = ta.rsi(df_copy['Close'], length=14)
    macd_df = ta.macd(df_copy['Close'], fast=12, slow=26, signal=9)
    df_copy['macd'] = macd_df['MACD_12_26_9']
    df_copy['macd_signal'] = macd_df['MACDs_12_26_9']
    df_copy['atr'] = ta.atr(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)
    bb_df = ta.bbands(df_copy['Close'], length=20, std=2)
    df_copy['bb_lower'] = bb_df['BBL_20_2.0']
    df_copy['bb_upper'] = bb_df['BBU_20_2.0']
    adx_df = ta.adx(df_copy['High'], df_copy['Low'], df_copy['Close'], length=14)
    df_copy['adx'] = adx_df['ADX_14']
    typical_price = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
    volume = df_copy['Volume']
    df_copy['vwap'] = (typical_price * volume).rolling(window=24).sum() / volume.rolling(window=24).sum()

    # --- 3. Custom Derived Features ---
    df_copy['hma_slope'] = df_copy['hma'].diff()
    df_copy['hma_dist_pct'] = (df_copy['Close'] - df_copy['hma']) / df_copy['hma'] * 100
    df_copy['vwap_dist_pct'] = (df_copy['Close'] - df_copy['vwap']) / df_copy['vwap'] * 100
    df_copy['atr_norm'] = (df_copy['atr'] / df_copy['Close']) * 100
    df_copy['bb_width_norm'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['Close'] * 100
    df_copy['volume_zscore'] = (df_copy['Volume'] - df_copy['Volume'].rolling(50).mean()) / df_copy['Volume'].rolling(50).std()

    # --- 4. Final Processing & Type Conversion ---
    
    # Convert categorical time features to strings as required by the library
    categorical_cols = ['hour', 'day_of_week', 'month']
    for col in categorical_cols:
        df_copy[col] = df_copy[col].astype(str)
        
    # Add a group ID (required by TFT)
    df_copy['group'] = "XAUUSD"

    # Robustly fill all NaNs created by the indicators
    df_copy.bfill(inplace=True)
    df_copy.dropna(inplace=True)

    return df_copy