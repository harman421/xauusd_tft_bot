# src/trading_logic.py

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime

import config

def initialize_mt5():
    """Initializes and logs in to the MetaTrader 5 terminal."""
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    print(f"Connecting to {config.MT5_SERVER}...")
    authorized = mt5.login(config.MT5_LOGIN, config.MT5_PASSWORD, config.MT5_SERVER)
    if not authorized:
        print(f"Failed to connect to account #{config.MT5_LOGIN}, error code = {mt5.last_error()}")
        mt5.shutdown()
        return False
        
    print("Connection to MT5 successful.")
    return True

def get_live_data(symbol, timeframe, n_bars):
    """Fetches the latest N bars from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print(f"Failed to get rates for {symbol}, error: {mt5.last_error()}")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def generate_signal(prediction_df, current_price):
    """
    Generates a trading signal based on model predictions.
    - BUY if the lower prediction quantile (p10) is above the current price.
    - SELL if the upper prediction quantile (p90) is below the current price.
    - HOLD otherwise.
    """
    # Get the prediction for the next hour
    next_hour_pred = prediction_df.iloc[0]
    
    lower_bound = next_hour_pred['p10']
    upper_bound = next_hour_pred['p90']

    if current_price < lower_bound:
        return "BUY"
    elif current_price > upper_bound:
        return "SELL"
    else:
        return "HOLD"
        
def execute_trade(signal, symbol, volume, dry_run=True):
    """Executes a trade based on the signal."""
    if signal == "HOLD":
        print("Signal is HOLD. No action taken.")
        return

    price = mt5.symbol_info_tick(symbol).ask if signal == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    if signal == "BUY":
        trade_type = mt5.ORDER_TYPE_BUY
        sl = price - 100 * mt5.symbol_info(symbol).point # Example SL
        tp = price + 200 * mt5.symbol_info(symbol).point # Example TP
    else: # SELL
        trade_type = mt5.ORDER_TYPE_SELL
        sl = price + 100 * mt5.symbol_info(symbol).point # Example SL
        tp = price - 200 * mt5.symbol_info(symbol).point # Example TP

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "TFT Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print(f"Preparing to {signal} {volume} lots of {symbol} at {price}")
    if dry_run:
        print("DRY RUN: Trade not sent.")
        return None

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order_send failed, retcode={result.retcode}")
        print(f"Result: {result}")
    else:
        print(f"Trade executed successfully: {result}")
    
    return result

def log_live_data(df: pd.DataFrame):
    """Appends live data to a log file."""
    try:
        df.to_csv(config.LIVE_DATA_LOG_FILE, mode='a', header=not config.LIVE_DATA_LOG_FILE.exists())
    except Exception as e:
        print(f"Error logging live data: {e}")