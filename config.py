# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- MT5 Credentials ---
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_SYMBOL = "XAUUSDm"

# --- ### THE MISSING SECTION IS HERE ### ---
# --- Project Directories ---
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LIVE_DATA_DIR = DATA_DIR / "live"

# Model and reports paths
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
# --- ### END OF MISSING SECTION ### ---


# --- File Paths ---
RAW_DATA_FILE = RAW_DATA_DIR / "xauusd_historical_3y.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "xauusd_cleaned.csv"
LIVE_DATA_LOG_FILE = LIVE_DATA_DIR / "live_data_log.csv"
MODEL_FILE = SAVED_MODELS_DIR / "tft_xauusd_v1_h1.pt" # Renamed for clarity

# --- Model & Trading Parameters for H1 Timeframe ---

# ENCODER_LENGTH: Use 7 days of hourly historical data
ENCODER_LENGTH = 24 * 7

# PREDICTION_LENGTH: Predict the next 24 hours
PREDICTION_LENGTH = 24

# Technical Indicator parameters (original H1 values)
RSI_PERIOD = 14
MA_SHORT = 20
MA_LONG = 50

# Trading parameters
TRADE_VOLUME = 0.01