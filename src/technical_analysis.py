"""Technical Analysis module using pandas-ta."""

import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Calculates technical indicators on OHLCV data."""

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA, RSI, MACD, and ATR indicators to the DataFrame."""
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for indicator calculation")
            return df

        result = df.copy()

        # EMA (Exponential Moving Averages)
        result["ema_9"] = ta.ema(result["close"], length=9)
        result["ema_21"] = ta.ema(result["close"], length=21)
        result["ema_50"] = ta.ema(result["close"], length=50)

        # RSI (Relative Strength Index)
        result["rsi"] = ta.rsi(result["close"], length=14)

        # MACD
        macd_result = ta.macd(result["close"], fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.empty:
            result["macd"] = macd_result.iloc[:, 0]
            result["macd_histogram"] = macd_result.iloc[:, 1]
            result["macd_signal"] = macd_result.iloc[:, 2]

        # ATR (Average True Range)
        result["atr"] = ta.atr(
            result["high"], result["low"], result["close"], length=14
        )

        # Bollinger Bands
        bbands = ta.bbands(result["close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            result["bb_lower"] = bbands.iloc[:, 0]
            result["bb_mid"] = bbands.iloc[:, 1]
            result["bb_upper"] = bbands.iloc[:, 2]

        # Volume SMA
        result["volume_sma_20"] = ta.sma(result["volume"], length=20)

        return result

    @staticmethod
    def get_latest_signals(df: pd.DataFrame) -> dict:
        """Extract the latest indicator values as a feature dict for ML."""
        if df.empty:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        features = {}

        # Price action
        features["price"] = latest.get("close", 0)
        features["price_change_pct"] = (
            (latest["close"] - prev["close"]) / prev["close"] * 100
            if prev["close"] != 0
            else 0
        )

        # EMA features
        if "ema_9" in latest.index and "ema_21" in latest.index:
            features["ema_9"] = latest["ema_9"]
            features["ema_21"] = latest["ema_21"]
            features["ema_50"] = latest.get("ema_50", 0)
            features["ema_cross"] = 1 if latest["ema_9"] > latest["ema_21"] else -1
            features["price_vs_ema50_pct"] = (
                (latest["close"] - latest.get("ema_50", latest["close"]))
                / latest.get("ema_50", latest["close"])
                * 100
                if latest.get("ema_50", 0) != 0
                else 0
            )

        # RSI
        features["rsi"] = latest.get("rsi", 50)

        # MACD
        features["macd"] = latest.get("macd", 0)
        features["macd_histogram"] = latest.get("macd_histogram", 0)
        features["macd_signal"] = latest.get("macd_signal", 0)
        features["macd_cross"] = (
            1 if features["macd"] > features["macd_signal"] else -1
        )

        # ATR
        features["atr"] = latest.get("atr", 0)
        features["atr_pct"] = (
            features["atr"] / latest["close"] * 100
            if latest["close"] != 0
            else 0
        )

        # Bollinger Bands
        features["bb_position"] = 0
        if "bb_upper" in latest.index and "bb_lower" in latest.index:
            bb_range = latest["bb_upper"] - latest["bb_lower"]
            if bb_range != 0:
                features["bb_position"] = (
                    (latest["close"] - latest["bb_lower"]) / bb_range
                )

        # Volume
        features["volume"] = latest.get("volume", 0)
        features["volume_ratio"] = (
            latest["volume"] / latest.get("volume_sma_20", latest["volume"])
            if latest.get("volume_sma_20", 0) != 0
            else 1
        )

        return features

    @staticmethod
    def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Build a feature matrix from the full OHLCV+indicator DataFrame
        for ML training. Each row is a time step with computed features."""
        if df.empty:
            return pd.DataFrame()

        feature_cols = [
            "close", "volume",
            "ema_9", "ema_21", "ema_50",
            "rsi",
            "macd", "macd_histogram", "macd_signal",
            "atr",
        ]

        available = [c for c in feature_cols if c in df.columns]
        result = df[available].copy()

        # Derived features
        if "ema_9" in result.columns and "ema_21" in result.columns:
            result["ema_cross"] = (result["ema_9"] > result["ema_21"]).astype(int)

        if "close" in result.columns and "ema_50" in result.columns:
            result["price_vs_ema50_pct"] = (
                (result["close"] - result["ema_50"]) / result["ema_50"] * 100
            )

        if "macd" in result.columns and "macd_signal" in result.columns:
            result["macd_cross"] = (result["macd"] > result["macd_signal"]).astype(int)

        if "bb_upper" in result.columns and "bb_lower" in result.columns:
            bb_range = result["bb_upper"] - result["bb_lower"]
            result["bb_position"] = (result["close"] - result["bb_lower"]) / bb_range

        if "volume" in result.columns and "volume_sma_20" in df.columns:
            result["volume_ratio"] = result["volume"] / df["volume_sma_20"]

        # Price change
        result["price_change_pct"] = result["close"].pct_change() * 100

        # Future return for labeling (used in training)
        result["future_return"] = result["close"].shift(-1) / result["close"] - 1

        result.dropna(inplace=True)
        return result
