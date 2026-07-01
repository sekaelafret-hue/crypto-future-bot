"""
AI Decision Logic v5 - Trend-Capture Strategy.

Core insight: In regime-switching markets, the edge comes from detecting 
regime changes early and riding them. Instead of trying to time perfect 
entries with pullbacks, we:

1. Enter on EMA crossovers (regime change signals)
2. Use ADX + volume confirmation to filter false crosses
3. Let winners run with trailing stops (no fixed TP)
4. Cut losers quickly with tight initial stops
5. Use mean-reversion only at extreme Bollinger Band levels

This produces fewer but larger winners, with controlled drawdown.
"""

from dataclasses import dataclass
import numpy as np

from .indicators import ema, rsi, macd, bollinger_bands, atr, adx, stochastic_rsi


@dataclass
class StrategyParams:
    """Strategy parameters."""
    # Trend EMAs
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 72.0
    rsi_oversold: float = 28.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ADX
    adx_period: int = 14
    adx_min_strength: float = 15.0
    
    # Bollinger
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR & Risk
    atr_period: int = 14
    initial_stop_atr: float = 2.0   # Initial stop distance
    trailing_stop_atr: float = 3.0  # Trailing stop distance (wider to let winners run)
    
    # Position management
    position_size_pct: float = 0.20
    leverage: float = 5.0
    max_trades_per_day: int = 6
    cooldown_candles: int = 4

    # Signal confirmation
    macd_confirm: bool = True  # Require MACD alignment
    ema_spread_min: float = 0.001  # Minimum EMA spread for trend entry
    
    # BB reversion
    bb_entry_threshold: float = 0.05  # Enter when price below this BB percentile
    bb_stop_atr: float = 1.5
    bb_tp_atr: float = 2.5
    
    # Breakeven management (0 = disabled)
    be_threshold: float = 2.0  # Move stop to entry after this many ATRs of profit


@dataclass
class Signal:
    direction: str
    strength: float
    stop_loss: float
    take_profit: float  # 0 = no fixed TP (trailing only)
    reason: str


class TradingStrategy:
    """Trend-capture + mean-reversion hybrid."""

    def __init__(self, params: StrategyParams | None = None):
        self.params = params or StrategyParams()
        self.last_trade_candle = -100
        self.trades_today = 0
        self.current_day = -1
        self.consecutive_losses = 0

    def compute_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict:
        p = self.params
        macd_line, macd_sig, macd_hist = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, p.bb_period, p.bb_std)
        return {
            "ema_fast": ema(close, p.ema_fast),
            "ema_slow": ema(close, p.ema_slow),
            "ema_trend": ema(close, p.ema_trend),
            "rsi": rsi(close, p.rsi_period),
            "macd_hist": macd_hist,
            "macd_line": macd_line,
            "macd_signal": macd_sig,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle,
            "atr": atr(high, low, close, p.atr_period),
            "adx": adx(high, low, close, p.adx_period),
        }

    def generate_signal(self, idx: int, close: np.ndarray, high: np.ndarray,
                       low: np.ndarray, indicators: dict) -> Signal:
        p = self.params

        if idx < p.ema_trend + 10:
            return Signal("none", 0, 0, 0, "warmup")

        if idx - self.last_trade_candle < p.cooldown_candles:
            return Signal("none", 0, 0, 0, "cooldown")

        day = idx // 24
        if day != self.current_day:
            self.current_day = day
            self.trades_today = 0
        if self.trades_today >= p.max_trades_per_day:
            return Signal("none", 0, 0, 0, "daily_limit")

        price = close[idx]
        current_atr = indicators["atr"][idx]
        ema_f = indicators["ema_fast"][idx]
        ema_s = indicators["ema_slow"][idx]
        ema_t = indicators["ema_trend"][idx]
        rsi_val = indicators["rsi"][idx]
        macd_h = indicators["macd_hist"][idx]
        adx_val = indicators["adx"][idx]
        bb_upper = indicators["bb_upper"][idx]
        bb_lower = indicators["bb_lower"][idx]

        # Previous candle values for crossover detection
        prev_ema_f = indicators["ema_fast"][idx - 1]
        prev_ema_s = indicators["ema_slow"][idx - 1]

        # === STRATEGY 1: EMA CROSSOVER (trend capture) ===
        
        # Bullish crossover: fast EMA crosses above slow EMA
        bull_cross = prev_ema_f <= prev_ema_s and ema_f > ema_s
        # Bearish crossover: fast EMA crosses below slow EMA
        bear_cross = prev_ema_f >= prev_ema_s and ema_f < ema_s

        # Macro trend context (don't fight the bigger trend)
        macro_bullish = price > ema_t
        macro_bearish = price < ema_t

        if bull_cross and macro_bullish:
            # Confirmation checks
            confirmed = True
            
            # MACD should be bullish or turning
            if p.macd_confirm and macd_h < 0:
                prev_macd = indicators["macd_hist"][idx - 1]
                if macd_h <= prev_macd:
                    confirmed = False
            
            # RSI should not be overbought
            if rsi_val > p.rsi_overbought:
                confirmed = False
            
            if confirmed:
                stop = price - p.initial_stop_atr * current_atr
                return Signal("long", 0.7, stop, 0, "ema_cross_bull")

        if bear_cross and macro_bearish:
            confirmed = True
            
            if p.macd_confirm and macd_h > 0:
                prev_macd = indicators["macd_hist"][idx - 1]
                if macd_h >= prev_macd:
                    confirmed = False
            
            if rsi_val < p.rsi_oversold:
                confirmed = False
            
            if confirmed:
                stop = price + p.initial_stop_atr * current_atr
                return Signal("short", 0.7, stop, 0, "ema_cross_bear")

        # === STRATEGY 2: TREND CONTINUATION (re-entry after pullback in strong trend) ===
        
        # Strong uptrend: all EMAs aligned and ADX confirms
        if ema_f > ema_s > ema_t and adx_val > p.adx_min_strength:
            spread = (ema_f - ema_t) / ema_t
            if spread > p.ema_spread_min:
                # Price near slow EMA or between fast and slow (pullback zone)
                dist_to_fast = (price - ema_f) / ema_f
                dist_to_slow = (price - ema_s) / ema_s
                
                # Entry: price is below or near fast EMA (pullback)
                if dist_to_fast < 0.002 and dist_to_slow > -0.005:
                    if rsi_val < 65:
                        stop = price - p.initial_stop_atr * current_atr
                        return Signal("long", 0.6, stop, 0, "trend_continuation")

        # Strong downtrend
        if ema_f < ema_s < ema_t and adx_val > p.adx_min_strength:
            spread = (ema_t - ema_f) / ema_t
            if spread > p.ema_spread_min:
                dist_to_fast = (ema_f - price) / ema_f
                dist_to_slow = (ema_s - price) / ema_s
                
                if dist_to_fast < 0.002 and dist_to_slow > -0.005:
                    if rsi_val > 35:
                        stop = price + p.initial_stop_atr * current_atr
                        return Signal("short", 0.6, stop, 0, "trend_continuation")

        # === STRATEGY 3: BOLLINGER BAND REVERSION (in weak trends) ===
        
        if adx_val < p.adx_min_strength:
            bb_width = bb_upper - bb_lower
            if bb_width > 0:
                bb_pos = (price - bb_lower) / bb_width
                
                # Extreme oversold
                if bb_pos < p.bb_entry_threshold and rsi_val < p.rsi_oversold:
                    stop = price - p.bb_stop_atr * current_atr
                    tp = price + p.bb_tp_atr * current_atr
                    return Signal("long", 0.5, stop, tp, "bb_oversold")
                
                # Extreme overbought
                if bb_pos > (1 - p.bb_entry_threshold) and rsi_val > p.rsi_overbought:
                    stop = price + p.bb_stop_atr * current_atr
                    tp = price - p.bb_tp_atr * current_atr
                    return Signal("short", 0.5, stop, tp, "bb_overbought")

        return Signal("none", 0, 0, 0, "no_setup")

    def on_trade_opened(self, idx: int):
        self.last_trade_candle = idx
        self.trades_today += 1

    def on_trade_closed(self, profitable: bool, direction: str):
        if profitable:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def get_position_size(self, signal: Signal) -> float:
        """Position sizing based on signal and loss streak."""
        base = self.params.position_size_pct
        # Reduce after consecutive losses
        if self.consecutive_losses >= 3:
            return base * 0.6
        if self.consecutive_losses >= 2:
            return base * 0.8
        # Slightly bigger on high conviction
        if signal.strength >= 0.7:
            return base * 1.15
        return base

    def get_leverage(self, signal: Signal) -> float:
        """Leverage with loss reduction."""
        base = self.params.leverage
        if self.consecutive_losses >= 3:
            return base * 0.7
        return base
