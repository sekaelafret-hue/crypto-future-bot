"""Public OHLCV data ingestion via CCXT (no API keys required)."""

import time
import logging
from typing import Optional

import ccxt
import pandas as pd

from src.config import (
    EXCHANGE_ID,
    FALLBACK_EXCHANGE_IDS,
    OHLCV_TIMEFRAME,
    OHLCV_LIMIT,
    SYMBOLS,
)

logger = logging.getLogger(__name__)


def _create_exchange(exchange_id: str):
    """Create a CCXT exchange instance and test connectivity."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    # Quick connectivity check
    exchange.fetch_ohlcv("BTC/USDT", "1h", limit=1)
    return exchange


class DataIngestion:
    """Fetches public OHLCV data from a crypto exchange using CCXT."""

    def __init__(self, exchange_id: str = EXCHANGE_ID):
        self.exchange = self._init_exchange(exchange_id)
        self._price_cache: dict[str, float] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: dict[str, float] = {}

    @staticmethod
    def _init_exchange(exchange_id: str):
        """Try primary exchange, fall back to alternatives if restricted."""
        ids_to_try = [exchange_id] + [
            eid for eid in FALLBACK_EXCHANGE_IDS if eid != exchange_id
        ]
        for eid in ids_to_try:
            try:
                ex = _create_exchange(eid)
                logger.info("Connected to exchange: %s", eid)
                return ex
            except Exception as e:
                logger.warning("Exchange %s unavailable: %s", eid, e)
        # Last resort: return the primary without testing
        logger.error("All exchanges failed; using %s without validation", exchange_id)
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = OHLCV_TIMEFRAME,
        limit: int = OHLCV_LIMIT,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for a symbol. Returns a DataFrame."""
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()
        if cache_key in self._last_fetch and (now - self._last_fetch[cache_key]) < 30:
            return self._ohlcv_cache[cache_key]

        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            self._ohlcv_cache[cache_key] = df
            self._last_fetch[cache_key] = now
            self._price_cache[symbol] = float(df["close"].iloc[-1])
            return df
        except Exception as e:
            logger.error("Failed to fetch OHLCV for %s: %s", symbol, e)
            if cache_key in self._ohlcv_cache:
                return self._ohlcv_cache[cache_key]
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest close price for a symbol."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        df = self.fetch_ohlcv(symbol, limit=5)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])

    def fetch_all_symbols(
        self, symbols: list[str] = SYMBOLS
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all configured symbols."""
        result = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol)
            if not df.empty:
                result[symbol] = df
        return result

    def get_all_latest_prices(
        self, symbols: list[str] = SYMBOLS
    ) -> dict[str, float]:
        """Get latest prices for all symbols."""
        prices = {}
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices
