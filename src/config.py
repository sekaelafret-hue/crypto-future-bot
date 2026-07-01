"""Global configuration for the trading bot."""

# Top 20 USDT-Margined Perpetual Futures symbols
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
    "UNI/USDT",
    "LTC/USDT",
    "ATOM/USDT",
    "ETC/USDT",
    "FIL/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "NEAR/USDT",
]

INITIAL_BALANCE = 10_000.0  # USDT
DEFAULT_LEVERAGE = 5
MAX_POSITION_SIZE_PCT = 0.10  # 10% of balance per position
STOP_LOSS_PCT = 0.03  # 3% stop loss
TAKE_PROFIT_PCT = 0.06  # 6% take profit
LIQUIDATION_THRESHOLD = 0.80  # 80% margin loss triggers liquidation

OHLCV_TIMEFRAME = "1h"
OHLCV_LIMIT = 200  # candles to fetch

EXCHANGE_ID = "okx"
FALLBACK_EXCHANGE_IDS = ["binance", "bybit"]

# ML model path
MODEL_PATH = "models/signal_model.joblib"

# CoinGecko public API
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Dashboard refresh interval (seconds)
DASHBOARD_REFRESH_INTERVAL = 30
