"""
Technical indicators for the trading algorithm.
All indicators operate on numpy arrays for performance.
"""

import numpy as np


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full_like(data, np.nan, dtype=np.float64)
    cumsum = np.cumsum(data)
    result[period - 1 :] = (cumsum[period - 1 :] - np.concatenate([[0], cumsum[:-period]])) / period
    # Fill initial NaN with expanding mean
    for i in range(period - 1):
        result[i] = np.mean(data[: i + 1])
    return result


def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros(len(data), dtype=np.float64)
    avg_loss = np.zeros(len(data), dtype=np.float64)

    # Initial SMA
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Smoothed averages
    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    rsi_values[:period] = 50.0  # Neutral for warmup period
    return rsi_values


def macd(
    data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD indicator. Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    data: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = sma(data, period)
    std = np.zeros_like(data, dtype=np.float64)
    for i in range(len(data)):
        window_start = max(0, i - period + 1)
        std[i] = np.std(data[window_start : i + 1])
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    tr = np.zeros(len(high), dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return ema(tr, period)


def adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average Directional Index - measures trend strength."""
    plus_dm = np.zeros(len(high), dtype=np.float64)
    minus_dm = np.zeros(len(high), dtype=np.float64)

    for i in range(1, len(high)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr_values = atr(high, low, close, period)
    plus_di = 100 * ema(plus_dm, period) / np.where(atr_values > 0, atr_values, 1.0)
    minus_di = 100 * ema(minus_dm, period) / np.where(atr_values > 0, atr_values, 1.0)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1.0)
    adx_values = ema(dx, period)
    return adx_values


def stochastic_rsi(data: np.ndarray, rsi_period: int = 14, stoch_period: int = 14) -> np.ndarray:
    """Stochastic RSI - RSI of RSI for momentum."""
    rsi_values = rsi(data, rsi_period)
    stoch_rsi = np.zeros_like(data, dtype=np.float64)
    for i in range(stoch_period, len(data)):
        window = rsi_values[i - stoch_period + 1 : i + 1]
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val - min_val > 0:
            stoch_rsi[i] = (rsi_values[i] - min_val) / (max_val - min_val) * 100
        else:
            stoch_rsi[i] = 50.0
    stoch_rsi[:stoch_period] = 50.0
    return stoch_rsi
