"""
MockExchange - Simulates a crypto futures exchange with realistic execution.
Supports long/short positions, leverage, funding rates, slippage, and fees.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Position:
    side: str  # "long" or "short"
    entry_price: float
    size: float  # in base currency units
    leverage: float
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None
    highest_pnl: float = 0.0


@dataclass
class Trade:
    entry_time: int  # candle index
    exit_time: int
    side: str
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class ExchangeConfig:
    initial_balance: float = 10000.0
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%
    slippage_pct: float = 0.0001  # 0.01%
    max_leverage: float = 20.0
    funding_rate: float = 0.0001  # per 8 hours
    liquidation_threshold: float = 0.9  # 90% margin loss


class MockExchange:
    """Simulates futures exchange execution with realistic constraints."""

    def __init__(self, config: ExchangeConfig | None = None):
        self.config = config or ExchangeConfig()
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [self.config.initial_balance]
        self.candle_index = 0

    def reset(self):
        self.balance = self.config.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = [self.config.initial_balance]
        self.candle_index = 0

    def get_equity(self, current_price: float) -> float:
        """Current account equity including unrealized PnL and locked margin."""
        if self.position is None:
            return self.balance
        unrealized = self._calc_pnl(self.position, current_price)
        return self.balance + self.position.size + unrealized

    def _calc_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        if position.side == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        return pnl_pct * position.size * position.leverage

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        if side == "long":
            return price * (1 + self.config.slippage_pct)
        return price * (1 - self.config.slippage_pct)

    def open_position(
        self,
        side: str,
        price: float,
        size_pct: float,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> bool:
        """Open a new position. size_pct is fraction of balance to use as margin."""
        if self.position is not None:
            return False  # Already in a position

        leverage = min(leverage, self.config.max_leverage)
        margin = self.balance * size_pct
        exec_price = self._apply_slippage(price, side)

        # Fee on entry
        fee = margin * leverage * self.config.taker_fee
        self.balance -= margin  # Lock margin
        self.balance -= fee

        self.position = Position(
            side=side,
            entry_price=exec_price,
            size=margin,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
        )
        return True

    def close_position(self, price: float, reason: str = "signal") -> Trade | None:
        """Close the current position."""
        if self.position is None:
            return None

        exit_side = "short" if self.position.side == "long" else "long"
        exec_price = self._apply_slippage(price, exit_side)

        pnl = self._calc_pnl(self.position, exec_price)
        pnl_pct = pnl / self.position.size

        # Fee on exit
        fee = self.position.size * self.position.leverage * self.config.taker_fee
        pnl -= fee

        self.balance += self.position.size + pnl  # Return margin + net PnL

        trade = Trade(
            entry_time=self.candle_index - 1,
            exit_time=self.candle_index,
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=exec_price,
            size=self.position.size,
            leverage=self.position.leverage,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.position = None
        return trade

    def update(self, high: float, low: float, close: float) -> Trade | None:
        """Update exchange state for a new candle. Checks stops/liquidation."""
        self.candle_index += 1
        trade = None

        if self.position is not None:
            # Check liquidation
            margin_loss_pct = -self._calc_pnl(self.position, close) / self.position.size
            if margin_loss_pct >= self.config.liquidation_threshold:
                trade = self.close_position(close, "liquidation")
                self.equity_curve.append(self.get_equity(close))
                return trade

            # Check stop loss (using high/low for realistic fills)
            if self.position.stop_loss is not None:
                if self.position.side == "long" and low <= self.position.stop_loss:
                    trade = self.close_position(self.position.stop_loss, "stop_loss")
                elif self.position.side == "short" and high >= self.position.stop_loss:
                    trade = self.close_position(self.position.stop_loss, "stop_loss")

            # Check take profit
            if trade is None and self.position is not None and self.position.take_profit is not None:
                if self.position.side == "long" and high >= self.position.take_profit:
                    trade = self.close_position(self.position.take_profit, "take_profit")
                elif self.position.side == "short" and low <= self.position.take_profit:
                    trade = self.close_position(self.position.take_profit, "take_profit")

            # Update trailing stop
            if trade is None and self.position is not None and self.position.trailing_stop_pct is not None:
                current_pnl = self._calc_pnl(self.position, close)
                if current_pnl > self.position.highest_pnl:
                    self.position.highest_pnl = current_pnl
                    # Move stop loss up
                    if self.position.side == "long":
                        new_stop = close * (1 - self.position.trailing_stop_pct)
                        if self.position.stop_loss is None or new_stop > self.position.stop_loss:
                            self.position.stop_loss = new_stop
                    else:
                        new_stop = close * (1 + self.position.trailing_stop_pct)
                        if self.position.stop_loss is None or new_stop < self.position.stop_loss:
                            self.position.stop_loss = new_stop

        self.equity_curve.append(self.get_equity(close))
        return trade

    def apply_funding(self) -> None:
        """Apply funding rate to open position (called every 8 hours)."""
        if self.position is not None:
            funding_cost = self.position.size * self.position.leverage * self.config.funding_rate
            if self.position.side == "long":
                self.balance -= funding_cost
            else:
                self.balance += funding_cost
