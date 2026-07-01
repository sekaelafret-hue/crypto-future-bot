"""Quick integration smoke test."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO)

def test_mock_exchange():
    print("=== Testing MockExchange ===")
    from src.mock_exchange import MockExchange, Side

    ex = MockExchange()
    assert ex.balance == 10_000.0
    assert ex.equity == 10_000.0

    pos = ex.open_position("BTC/USDT", Side.LONG, 50000.0)
    assert pos is not None
    assert pos.symbol == "BTC/USDT"
    assert pos.side == Side.LONG
    print(f"  Opened position: {pos.to_dict()}")
    print(f"  Balance after open: {ex.balance}")

    events = ex.update_positions({"BTC/USDT": 51000.0})
    print(f"  Unrealized PnL after price move: {pos.unrealized_pnl:.2f}")

    pnl = ex.close_position(pos, 51000.0, "test")
    print(f"  Realized PnL: {pnl:.2f}")
    print(f"  Balance after close: {ex.balance:.2f}")
    print(f"  Portfolio: {ex.get_portfolio_summary()}")
    print("  PASS\n")


def test_ml_engine():
    print("=== Testing ML Engine ===")
    from src.ml_engine import MLEngine

    ml = MLEngine()
    assert ml.model is not None

    features = {
        "rsi": 28.0,
        "macd": 0.5,
        "macd_histogram": 0.3,
        "macd_signal": 0.2,
        "macd_cross": 1,
        "ema_cross": 1,
        "price_vs_ema50_pct": 2.0,
        "atr": 1.5,
        "price_change_pct": 0.5,
        "volume_ratio": 1.8,
        "bb_position": 0.15,
    }
    signal = ml.predict(features)
    print(f"  Signal: {signal.action} (conf: {signal.confidence:.3f})")
    print(f"  Reasoning: {signal.reasoning}")
    print("  PASS\n")


def test_technical_analysis():
    print("=== Testing Technical Analysis ===")
    import pandas as pd
    import numpy as np
    from src.technical_analysis import TechnicalAnalyzer

    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        "open": close - np.random.rand(n) * 50,
        "high": close + np.random.rand(n) * 100,
        "low": close - np.random.rand(n) * 100,
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=dates)

    analyzer = TechnicalAnalyzer()
    result = analyzer.compute_indicators(df)
    print(f"  Columns: {list(result.columns)}")
    assert "ema_9" in result.columns
    assert "rsi" in result.columns
    assert "macd" in result.columns

    features = analyzer.get_latest_signals(result)
    print(f"  Features: {features}")
    print("  PASS\n")


def test_data_ingestion():
    print("=== Testing Data Ingestion (public API) ===")
    from src.data_ingestion import DataIngestion

    di = DataIngestion()
    df = di.fetch_ohlcv("BTC/USDT", limit=10)
    print(f"  Fetched {len(df)} candles for BTC/USDT")
    assert len(df) > 0
    price = di.get_latest_price("BTC/USDT")
    print(f"  Latest BTC price: ${price:,.2f}")
    print("  PASS\n")


if __name__ == "__main__":
    test_mock_exchange()
    test_ml_engine()
    test_technical_analysis()
    test_data_ingestion()
    print("=== ALL TESTS PASSED ===")
