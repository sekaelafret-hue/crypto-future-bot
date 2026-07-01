"""Main Trading Bot orchestrator — ties all modules together."""

import logging
import time
from dataclasses import dataclass, field

from src.config import SYMBOLS
from src.data_ingestion import DataIngestion
from src.mock_exchange import MockExchange, Side
from src.technical_analysis import TechnicalAnalyzer
from src.ml_engine import MLEngine, Signal

logger = logging.getLogger(__name__)


@dataclass
class SymbolState:
    """Tracks the current state of analysis for a symbol."""
    symbol: str
    price: float = 0.0
    signal: Signal | None = None
    features: dict = field(default_factory=dict)
    last_updated: float = 0.0


class TradingBot:
    """Orchestrates data ingestion, analysis, ML signals, and mock trading."""

    def __init__(self, symbols: list[str] = SYMBOLS):
        self.symbols = symbols
        self.data = DataIngestion()
        self.exchange = MockExchange()
        self.analyzer = TechnicalAnalyzer()
        self.ml = MLEngine()
        self.states: dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in symbols
        }
        self.events: list[dict] = []
        self.cycle_count = 0

        # Confidence threshold for taking trades
        self.min_confidence = 0.55

    def run_cycle(self) -> dict:
        """Run one full analysis + trading cycle.
        Returns a summary dict of what happened.
        """
        self.cycle_count += 1
        cycle_events = []
        signals_generated = []

        # 1. Fetch data and compute indicators
        for symbol in self.symbols:
            try:
                df = self.data.fetch_ohlcv(symbol)
                if df.empty:
                    continue

                df = self.analyzer.compute_indicators(df)
                features = self.analyzer.get_latest_signals(df)
                price = features.get("price", 0)

                signal = self.ml.predict(features)

                self.states[symbol] = SymbolState(
                    symbol=symbol,
                    price=price,
                    signal=signal,
                    features=features,
                    last_updated=time.time(),
                )
                signals_generated.append({
                    "symbol": symbol,
                    "action": signal.action,
                    "confidence": signal.confidence,
                })
            except Exception as e:
                logger.error("Error processing %s: %s", symbol, e)

        # 2. Update existing positions with latest prices
        prices = {s: st.price for s, st in self.states.items() if st.price > 0}
        position_events = self.exchange.update_positions(prices)
        cycle_events.extend(position_events)

        # 3. Execute new trades based on signals
        for symbol, state in self.states.items():
            if state.signal is None:
                continue

            signal = state.signal

            # Skip if below confidence threshold
            if signal.confidence < self.min_confidence:
                continue

            # Skip HOLD signals
            if signal.action == "HOLD":
                continue

            # Check if we already have a position
            has_position = any(
                p.symbol == symbol for p in self.exchange.open_positions
            )

            if signal.action == "LONG" and not has_position:
                pos = self.exchange.open_position(
                    symbol=symbol,
                    side=Side.LONG,
                    current_price=state.price,
                )
                if pos:
                    cycle_events.append({
                        "type": "new_trade",
                        "symbol": symbol,
                        "side": "LONG",
                        "price": state.price,
                        "confidence": signal.confidence,
                    })

            elif signal.action == "SHORT" and not has_position:
                pos = self.exchange.open_position(
                    symbol=symbol,
                    side=Side.SHORT,
                    current_price=state.price,
                )
                if pos:
                    cycle_events.append({
                        "type": "new_trade",
                        "symbol": symbol,
                        "side": "SHORT",
                        "price": state.price,
                        "confidence": signal.confidence,
                    })

            # Close positions on opposing signals
            elif has_position:
                for pos in self.exchange.open_positions:
                    if pos.symbol != symbol:
                        continue
                    if (
                        (pos.side == Side.LONG and signal.action == "SHORT")
                        or (pos.side == Side.SHORT and signal.action == "LONG")
                    ):
                        pnl = self.exchange.close_position(
                            pos, state.price, reason="signal_reversal"
                        )
                        cycle_events.append({
                            "type": "close_trade",
                            "symbol": symbol,
                            "reason": "signal_reversal",
                            "pnl": pnl,
                        })

        self.events.extend(cycle_events)

        return {
            "cycle": self.cycle_count,
            "signals": signals_generated,
            "events": cycle_events,
            "portfolio": self.exchange.get_portfolio_summary(),
        }

    def get_full_state(self) -> dict:
        """Get complete bot state for dashboard rendering."""
        return {
            "portfolio": self.exchange.get_portfolio_summary(),
            "open_positions": [
                p.to_dict() for p in self.exchange.open_positions
            ],
            "closed_positions": [
                p.to_dict() for p in self.exchange.closed_positions[-50:]
            ],
            "signals": {
                symbol: {
                    "price": state.price,
                    "signal": state.signal.to_dict() if state.signal else None,
                    "last_updated": state.last_updated,
                }
                for symbol, state in self.states.items()
            },
            "recent_events": self.events[-20:],
            "cycle_count": self.cycle_count,
        }
