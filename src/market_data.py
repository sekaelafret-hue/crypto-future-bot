"""
Market data generator - creates realistic 6-month crypto price data
using geometric Brownian motion with regime changes, volatility clustering,
and mean-reversion characteristics typical of crypto markets.
"""

import numpy as np
import pandas as pd


def generate_ohlcv_data(
    days: int = 180,
    candle_interval_minutes: int = 60,
    initial_price: float = 40000.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate realistic OHLCV data for backtesting.

    Uses a regime-switching model with:
    - Trending regimes (bull/bear)
    - Mean-reverting regimes (range-bound)
    - Volatility clustering (GARCH-like)
    - Realistic volume patterns
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    candles_per_day = (24 * 60) // candle_interval_minutes
    total_candles = days * candles_per_day

    # Regime parameters
    # 0 = bull trend, 1 = bear trend, 2 = range-bound
    regime = 0
    regime_duration = 0
    min_regime_duration = 36  # minimum candles in a regime

    prices = np.zeros(total_candles)
    volumes = np.zeros(total_candles)
    prices[0] = initial_price

    # Volatility state (GARCH-like)
    base_volatility = 0.0009  # per-candle volatility (calibrated to BTC 1h)
    current_volatility = base_volatility

    for i in range(1, total_candles):
        regime_duration += 1

        # Regime transition
        if regime_duration > min_regime_duration and rng.random() < 0.005:
            regime = rng.integers(0, 3)
            regime_duration = 0

        # Drift based on regime (calibrated to real BTC 1h data characteristics)
        if regime == 0:  # Bull
            drift = 0.00025
        elif regime == 1:  # Bear
            drift = -0.00020
        else:  # Range-bound
            # Mean revert toward a local mean
            local_mean = np.mean(prices[max(0, i - 100) : i])
            drift = 0.002 * (local_mean - prices[i - 1]) / prices[i - 1]

        # Volatility clustering
        shock = rng.standard_normal()
        current_volatility = (
            0.9 * current_volatility + 0.1 * base_volatility + 0.05 * abs(shock) * base_volatility
        )

        # Price update
        ret = drift + current_volatility * shock
        prices[i] = prices[i - 1] * (1 + ret)

        # Volume (higher in trends, spikes on large moves)
        base_volume = 1000 + 500 * abs(ret) / base_volatility
        volumes[i] = base_volume * (1 + rng.exponential(0.5))

    # Build OHLCV from close prices
    opens = np.zeros(total_candles)
    highs = np.zeros(total_candles)
    lows = np.zeros(total_candles)
    closes = prices.copy()

    opens[0] = prices[0]
    for i in range(1, total_candles):
        opens[i] = closes[i - 1]
        intra_vol = current_volatility * 0.5
        high_ext = abs(rng.standard_normal()) * intra_vol * prices[i]
        low_ext = abs(rng.standard_normal()) * intra_vol * prices[i]
        highs[i] = max(opens[i], closes[i]) + high_ext
        lows[i] = min(opens[i], closes[i]) - low_ext

    highs[0] = prices[0] * 1.001
    lows[0] = prices[0] * 0.999

    timestamps = pd.date_range(
        start="2025-01-01", periods=total_candles, freq=f"{candle_interval_minutes}min"
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    return df
