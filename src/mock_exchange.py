"""Paper Trading / Mock Execution Engine."""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.config import (
    INITIAL_BALANCE,
    DEFAULT_LEVERAGE,
    MAX_POSITION_SIZE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    LIQUIDATION_THRESHOLD,
)

logger = logging.getLogger(__name__)


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"
    STOPPED_OUT = "STOPPED_OUT"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    leverage: int
    margin: float
    stop_loss: float
    take_profit: float
    status: OrderStatus = OrderStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    close_price: Optional[float] = None
    close_reason: str = ""

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.quantity

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": round(self.entry_price, 4),
            "quantity": round(self.quantity, 6),
            "leverage": self.leverage,
            "margin": round(self.margin, 2),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "status": self.status.value,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "close_price": self.close_price,
            "close_reason": self.close_reason,
        }


class MockExchange:
    """Simulated futures exchange for paper trading."""

    def __init__(
        self,
        initial_balance: float = INITIAL_BALANCE,
        default_leverage: int = DEFAULT_LEVERAGE,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.default_leverage = default_leverage
        self.open_positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.trade_log: list[dict] = []

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions)

    @property
    def equity(self) -> float:
        return self.balance + self.total_unrealized_pnl

    @property
    def total_trades(self) -> int:
        return len(self.closed_positions)

    @property
    def winning_trades(self) -> int:
        return sum(1 for p in self.closed_positions if p.realized_pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for p in self.closed_positions if p.realized_pnl <= 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.closed_positions)

    def open_position(
        self,
        symbol: str,
        side: Side,
        current_price: float,
        leverage: Optional[int] = None,
        position_size_pct: float = MAX_POSITION_SIZE_PCT,
    ) -> Optional[Position]:
        """Open a new simulated position."""
        if leverage is None:
            leverage = self.default_leverage

        # Check if already have a position in this symbol
        for pos in self.open_positions:
            if pos.symbol == symbol:
                logger.warning("Already have an open position for %s", symbol)
                return None

        margin = self.balance * position_size_pct
        if margin > self.balance:
            logger.warning("Insufficient balance for position on %s", symbol)
            return None

        notional = margin * leverage
        quantity = notional / current_price

        if side == Side.LONG:
            stop_loss = current_price * (1 - STOP_LOSS_PCT)
            take_profit = current_price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = current_price * (1 + STOP_LOSS_PCT)
            take_profit = current_price * (1 - TAKE_PROFIT_PCT)

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            quantity=quantity,
            leverage=leverage,
            margin=margin,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.balance -= margin
        self.open_positions.append(position)

        log_entry = {
            "action": "OPEN",
            "symbol": symbol,
            "side": side.value,
            "price": current_price,
            "quantity": quantity,
            "margin": margin,
            "leverage": leverage,
            "timestamp": time.time(),
        }
        self.trade_log.append(log_entry)
        logger.info(
            "Opened %s %s @ %.4f | Qty: %.6f | Margin: %.2f",
            side.value, symbol, current_price, quantity, margin,
        )
        return position

    def close_position(
        self, position: Position, current_price: float, reason: str = "manual"
    ) -> float:
        """Close a position and realize PnL."""
        if position.side == Side.LONG:
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity

        position.realized_pnl = pnl
        position.unrealized_pnl = 0.0
        position.status = OrderStatus.CLOSED
        position.closed_at = time.time()
        position.close_price = current_price
        position.close_reason = reason

        self.balance += position.margin + pnl
        self.open_positions.remove(position)
        self.closed_positions.append(position)

        log_entry = {
            "action": "CLOSE",
            "symbol": position.symbol,
            "side": position.side.value,
            "entry_price": position.entry_price,
            "exit_price": current_price,
            "pnl": pnl,
            "reason": reason,
            "timestamp": time.time(),
        }
        self.trade_log.append(log_entry)
        logger.info(
            "Closed %s %s @ %.4f | PnL: %.2f | Reason: %s",
            position.side.value, position.symbol, current_price, pnl, reason,
        )
        return pnl

    def update_positions(self, prices: dict[str, float]) -> list[dict]:
        """Update all open positions with current prices.
        Checks stop-loss, take-profit, and liquidation.
        Returns list of events (closes triggered).
        """
        events = []
        positions_to_close = []

        for position in self.open_positions:
            if position.symbol not in prices:
                continue

            current_price = prices[position.symbol]

            # Calculate unrealized PnL
            if position.side == Side.LONG:
                position.unrealized_pnl = (
                    (current_price - position.entry_price) * position.quantity
                )
            else:
                position.unrealized_pnl = (
                    (position.entry_price - current_price) * position.quantity
                )

            # Check liquidation (margin loss exceeds threshold)
            if position.unrealized_pnl <= -(position.margin * LIQUIDATION_THRESHOLD):
                positions_to_close.append((position, current_price, "liquidation"))
                continue

            # Check stop-loss
            if position.side == Side.LONG and current_price <= position.stop_loss:
                positions_to_close.append((position, current_price, "stop_loss"))
                continue
            if position.side == Side.SHORT and current_price >= position.stop_loss:
                positions_to_close.append((position, current_price, "stop_loss"))
                continue

            # Check take-profit
            if position.side == Side.LONG and current_price >= position.take_profit:
                positions_to_close.append((position, current_price, "take_profit"))
                continue
            if position.side == Side.SHORT and current_price <= position.take_profit:
                positions_to_close.append((position, current_price, "take_profit"))
                continue

        for position, price, reason in positions_to_close:
            if reason == "liquidation":
                position.status = OrderStatus.LIQUIDATED
            elif reason == "stop_loss":
                position.status = OrderStatus.STOPPED_OUT
            elif reason == "take_profit":
                position.status = OrderStatus.TAKE_PROFIT

            pnl = self.close_position(position, price, reason)
            events.append({
                "symbol": position.symbol,
                "side": position.side.value,
                "reason": reason,
                "pnl": pnl,
                "price": price,
            })

        return events

    def get_portfolio_summary(self) -> dict:
        """Get a summary of the current portfolio state."""
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "initial_balance": self.initial_balance,
            "total_unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "open_positions": len(self.open_positions),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "return_pct": round(
                ((self.equity - self.initial_balance) / self.initial_balance) * 100, 2
            ),
        }
