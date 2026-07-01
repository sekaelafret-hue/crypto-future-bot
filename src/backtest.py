"""
Backtesting engine - runs strategy against historical data and computes performance metrics.
Includes advanced position management (breakeven move, dynamic trailing).
"""

from dataclasses import dataclass
import numpy as np

from .market_data import generate_ohlcv_data
from .mock_exchange import MockExchange, ExchangeConfig
from .strategy import TradingStrategy, StrategyParams


@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_pnl_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    longest_winning_streak: int
    longest_losing_streak: int
    final_balance: float
    initial_balance: float

    def meets_targets(self) -> bool:
        return (
            self.total_trades >= 100
            and self.total_pnl_pct > 15.0
            and self.max_drawdown_pct < 10.0
        )

    def summary(self) -> str:
        return (
            f"{'='*60}\n"
            f"BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"Total Trades:        {self.total_trades}\n"
            f"Win Rate:            {self.win_rate:.1f}%\n"
            f"Net P/L:             ${self.total_pnl:.2f} ({self.total_pnl_pct:.2f}%)\n"
            f"Sharpe Ratio:        {self.sharpe_ratio:.3f}\n"
            f"Max Drawdown:        {self.max_drawdown_pct:.2f}%\n"
            f"Profit Factor:       {self.profit_factor:.2f}\n"
            f"Avg Win:             {self.avg_win:.2f}%\n"
            f"Avg Loss:            {self.avg_loss:.2f}%\n"
            f"Win Streak:          {self.longest_winning_streak}\n"
            f"Loss Streak:         {self.longest_losing_streak}\n"
            f"Final Balance:       ${self.final_balance:.2f}\n"
            f"{'='*60}\n"
            f"TARGETS: Trades>=100 [{self.total_trades >= 100}] | "
            f"P/L>15% [{self.total_pnl_pct > 15.0}] | "
            f"DD<10% [{self.max_drawdown_pct < 10.0}]\n"
            f"{'='*60}"
        )


def compute_metrics(exchange: MockExchange) -> BacktestResult:
    """Compute all performance metrics."""
    trades = exchange.trades
    total_trades = len(trades)

    if total_trades == 0:
        return BacktestResult(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl=0, total_pnl_pct=0, sharpe_ratio=0, max_drawdown=0,
            max_drawdown_pct=0, avg_trade_pnl_pct=0, profit_factor=0,
            avg_win=0, avg_loss=0, longest_winning_streak=0,
            longest_losing_streak=0, final_balance=exchange.balance,
            initial_balance=exchange.initial_balance,
        )

    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct * 100 for t in trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    total_pnl_pct = (exchange.balance - exchange.initial_balance) / exchange.initial_balance * 100

    win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0

    # Sharpe Ratio
    returns = np.array(pnl_pcts)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(min(total_trades, 252))
    else:
        sharpe = 0.0

    # Max Drawdown
    equity = np.array(exchange.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_dd_pct = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    max_dd = float(np.max(peak - equity)) if len(equity) > 0 else 0.0

    # Profit factor
    gross_profit = sum(winning) if winning else 0
    gross_loss = abs(sum(losing)) if losing else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Averages
    win_pcts = [t.pnl_pct * 100 for t in trades if t.pnl > 0]
    loss_pcts = [t.pnl_pct * 100 for t in trades if t.pnl <= 0]
    avg_win = float(np.mean(win_pcts)) if win_pcts else 0
    avg_loss = float(np.mean(loss_pcts)) if loss_pcts else 0

    # Streaks
    longest_win = longest_loss = current_win = current_loss = 0
    for t in trades:
        if t.pnl > 0:
            current_win += 1
            current_loss = 0
            longest_win = max(longest_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            longest_loss = max(longest_loss, current_loss)

    return BacktestResult(
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_trade_pnl_pct=float(np.mean(pnl_pcts)),
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        longest_winning_streak=longest_win,
        longest_losing_streak=longest_loss,
        final_balance=exchange.balance,
        initial_balance=exchange.initial_balance,
    )


def run_backtest(
    strategy_params: StrategyParams | None = None,
    exchange_config: ExchangeConfig | None = None,
    data_seed: int | None = None,
    days: int = 180,
) -> BacktestResult:
    """Run a full backtest with advanced position management."""
    df = generate_ohlcv_data(days=days, seed=data_seed)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    strategy = TradingStrategy(strategy_params)
    exchange = MockExchange(exchange_config)

    indicators = strategy.compute_indicators(close, high, low)

    funding_interval = 8
    
    # Track entry ATR for breakeven/trailing management
    entry_atr = 0.0
    # Breakeven threshold: move stop to entry after this many ATRs of profit
    be_threshold = getattr(strategy_params or StrategyParams(), 'be_threshold', 2.0)

    for i in range(1, len(close)):
        # Manual position management BEFORE exchange update
        if exchange.position is not None and be_threshold > 0:
            atr_move = (close[i] - exchange.position.entry_price) / entry_atr if entry_atr > 0 else 0
            
            if exchange.position.side == "short":
                atr_move = -atr_move
            
            # Move to breakeven after threshold ATR profit
            if atr_move > be_threshold:
                be_price = exchange.position.entry_price
                if exchange.position.side == "long":
                    if exchange.position.stop_loss is None or be_price > exchange.position.stop_loss:
                        exchange.position.stop_loss = be_price
                else:
                    if exchange.position.stop_loss is None or be_price < exchange.position.stop_loss:
                        exchange.position.stop_loss = be_price

        # Update exchange (check stops, etc.)
        trade = exchange.update(high[i], low[i], close[i])

        if trade is not None:
            strategy.on_trade_closed(trade.pnl > 0, trade.side)

        # Funding
        if i % funding_interval == 0:
            exchange.apply_funding()

        # Generate signal if no position
        if exchange.position is None:
            signal = strategy.generate_signal(i, close, high, low, indicators)

            if signal.direction != "none":
                size_pct = strategy.get_position_size(signal)
                leverage = strategy.get_leverage(signal)

                # Compute trailing stop percentage from ATR
                current_atr = indicators["atr"][i]
                trail_atr_mult = getattr(strategy.params, 'trailing_stop_atr', 2.5)
                trail_pct = (trail_atr_mult * current_atr) / close[i]

                tp = signal.take_profit if signal.take_profit != 0 else None

                success = exchange.open_position(
                    side=signal.direction,
                    price=close[i],
                    size_pct=size_pct,
                    leverage=leverage,
                    stop_loss=signal.stop_loss,
                    take_profit=tp,
                    trailing_stop_pct=trail_pct,
                )
                if success:
                    strategy.on_trade_opened(i)
                    entry_atr = current_atr

    # Close remaining
    if exchange.position is not None:
        trade = exchange.close_position(close[-1], "end_of_data")
        if trade is not None:
            strategy.on_trade_closed(trade.pnl > 0, trade.side)

    return compute_metrics(exchange)


def main():
    """Run a single backtest with default parameters."""
    result = run_backtest(data_seed=42)
    print(result.summary())


if __name__ == "__main__":
    main()
