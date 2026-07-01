"""Streamlit Dashboard for the Crypto Futures Trading Bot."""

import sys
import os
import time
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trading_bot import TradingBot
from src.config import SYMBOLS, DASHBOARD_REFRESH_INTERVAL

logging.basicConfig(level=logging.INFO)

# --- Page Config ---
st.set_page_config(
    page_title="Crypto Futures Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Init ---
if "bot" not in st.session_state:
    with st.spinner("Initializing trading bot..."):
        st.session_state.bot = TradingBot()
        st.session_state.last_cycle = 0
        st.session_state.auto_trade = False

bot: TradingBot = st.session_state.bot


def run_cycle():
    """Execute one trading cycle."""
    with st.spinner("Running analysis cycle..."):
        result = bot.run_cycle()
        st.session_state.last_cycle = time.time()
        return result


# --- Sidebar ---
with st.sidebar:
    st.title("Trading Bot Controls")
    st.divider()

    if st.button("Run Analysis Cycle", type="primary", use_container_width=True):
        run_cycle()

    st.session_state.auto_trade = st.toggle(
        "Auto-refresh",
        value=st.session_state.get("auto_trade", False),
        help="Auto-refresh the dashboard periodically",
    )

    st.divider()

    # Portfolio summary in sidebar
    portfolio = bot.exchange.get_portfolio_summary()
    st.subheader("Account Overview")
    st.metric("Balance", f"${portfolio['balance']:,.2f}")
    st.metric("Equity", f"${portfolio['equity']:,.2f}")
    st.metric(
        "Total Return",
        f"{portfolio['return_pct']:+.2f}%",
        delta=f"${portfolio['total_realized_pnl']:+,.2f} realized",
    )
    st.metric("Unrealized PnL", f"${portfolio['total_unrealized_pnl']:+,.2f}")

    st.divider()
    st.subheader("Trade Stats")
    col1, col2 = st.columns(2)
    col1.metric("Total Trades", portfolio["total_trades"])
    col2.metric("Win Rate", f"{portfolio['win_rate']:.1f}%")
    col1.metric("Wins", portfolio["winning_trades"])
    col2.metric("Losses", portfolio["losing_trades"])

    st.divider()
    st.caption(f"Cycles completed: {bot.cycle_count}")
    if st.session_state.last_cycle:
        elapsed = time.time() - st.session_state.last_cycle
        st.caption(f"Last cycle: {elapsed:.0f}s ago")

# --- Main Content ---
st.title("Crypto Futures Algorithmic Trading Dashboard")
st.caption("Simulated paper trading with ML-driven signals | No API keys required")

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Initial Balance", f"${portfolio['initial_balance']:,.2f}")
col2.metric("Current Equity", f"${portfolio['equity']:,.2f}")
col3.metric("Open Positions", portfolio["open_positions"])
col4.metric("Win Rate", f"{portfolio['win_rate']:.1f}%")
col5.metric("Return", f"{portfolio['return_pct']:+.2f}%")

st.divider()

# --- Tabs ---
tab_signals, tab_positions, tab_history, tab_charts, tab_details = st.tabs(
    ["ML Signals", "Open Positions", "Trade History", "Charts", "Symbol Details"]
)

# --- ML Signals Tab ---
with tab_signals:
    st.subheader("Current ML Signals & AI Reasoning")

    if bot.cycle_count == 0:
        st.info(
            "No signals yet. Click 'Run Analysis Cycle' in the sidebar to start."
        )
    else:
        state = bot.get_full_state()
        signals_data = []
        for symbol, info in state["signals"].items():
            if info["signal"] is None:
                continue
            sig = info["signal"]
            signals_data.append({
                "Symbol": symbol,
                "Price": f"${info['price']:,.4f}" if info["price"] > 0 else "N/A",
                "Signal": sig["action"],
                "Confidence": f"{sig['confidence']:.1%}",
                "Reasoning": sig["reasoning"],
            })

        if signals_data:
            df_signals = pd.DataFrame(signals_data)

            def color_signal(val):
                if val == "LONG":
                    return "background-color: #1a472a; color: #4ade80"
                elif val == "SHORT":
                    return "background-color: #4a1a1a; color: #f87171"
                return "background-color: #3a3a1a; color: #fbbf24"

            styled = df_signals.style.map(
                color_signal, subset=["Signal"]
            )
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=min(len(signals_data) * 40 + 50, 600),
            )

            # AI Reasoning expandable for each
            st.subheader("Detailed AI Reasoning")
            for item in signals_data:
                signal_emoji = (
                    "🟢" if item["Signal"] == "LONG"
                    else "🔴" if item["Signal"] == "SHORT"
                    else "🟡"
                )
                with st.expander(
                    f"{signal_emoji} {item['Symbol']} — {item['Signal']} "
                    f"({item['Confidence']})"
                ):
                    st.write(item["Reasoning"])
        else:
            st.warning("No signals generated yet.")

# --- Open Positions Tab ---
with tab_positions:
    st.subheader("Active Simulated Positions")

    if not bot.exchange.open_positions:
        st.info("No open positions.")
    else:
        positions_data = []
        for pos in bot.exchange.open_positions:
            d = pos.to_dict()
            positions_data.append({
                "Symbol": d["symbol"],
                "Side": d["side"],
                "Entry Price": f"${d['entry_price']:,.4f}",
                "Quantity": f"{d['quantity']:.6f}",
                "Leverage": f"{d['leverage']}x",
                "Margin": f"${d['margin']:,.2f}",
                "Unrealized PnL": f"${d['unrealized_pnl']:+,.2f}",
                "Stop Loss": f"${d['stop_loss']:,.4f}",
                "Take Profit": f"${d['take_profit']:,.4f}",
            })

        df_positions = pd.DataFrame(positions_data)

        def color_side(val):
            if val == "LONG":
                return "color: #4ade80"
            return "color: #f87171"

        def color_pnl(val):
            if "+" in str(val):
                return "color: #4ade80"
            elif "-" in str(val):
                return "color: #f87171"
            return ""

        styled = df_positions.style.map(
            color_side, subset=["Side"]
        ).map(color_pnl, subset=["Unrealized PnL"])

        st.dataframe(styled, use_container_width=True, hide_index=True)

# --- Trade History Tab ---
with tab_history:
    st.subheader("Closed Trades History")

    if not bot.exchange.closed_positions:
        st.info("No closed trades yet.")
    else:
        history_data = []
        for pos in reversed(bot.exchange.closed_positions[-50:]):
            d = pos.to_dict()
            history_data.append({
                "Symbol": d["symbol"],
                "Side": d["side"],
                "Entry": f"${d['entry_price']:,.4f}",
                "Exit": (
                    f"${d['close_price']:,.4f}" if d["close_price"] else "N/A"
                ),
                "PnL": f"${d['realized_pnl']:+,.2f}",
                "Reason": d["close_reason"],
                "Status": d["status"],
            })

        df_history = pd.DataFrame(history_data)

        def color_hist_pnl(val):
            if "+" in str(val):
                return "color: #4ade80"
            elif "-" in str(val):
                return "color: #f87171"
            return ""

        styled = df_history.style.map(color_hist_pnl, subset=["PnL"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # PnL chart
        pnl_values = [p.realized_pnl for p in bot.exchange.closed_positions]
        cumulative_pnl = []
        running = 0
        for v in pnl_values:
            running += v
            cumulative_pnl.append(running)

        fig_pnl = go.Figure()
        fig_pnl.add_trace(
            go.Scatter(
                y=cumulative_pnl,
                mode="lines+markers",
                name="Cumulative PnL",
                line={"color": "#4ade80", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(74, 222, 128, 0.1)",
            )
        )
        fig_pnl.update_layout(
            title="Cumulative Realized PnL",
            yaxis_title="PnL ($)",
            xaxis_title="Trade #",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

# --- Charts Tab ---
with tab_charts:
    st.subheader("Price Charts with Indicators")

    available_symbols = [
        s for s, st_data in bot.states.items() if st_data.price > 0
    ]

    if not available_symbols:
        st.info("Run an analysis cycle first to load chart data.")
    else:
        selected_symbol = st.selectbox(
            "Select Symbol", available_symbols, index=0
        )

        if selected_symbol:
            df = bot.data.fetch_ohlcv(selected_symbol)
            if not df.empty:
                df = bot.analyzer.compute_indicators(df)

                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=[
                        f"{selected_symbol} Price",
                        "RSI",
                        "MACD",
                    ],
                )

                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="Price",
                    ),
                    row=1,
                    col=1,
                )

                # EMAs
                if "ema_9" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["ema_9"],
                            name="EMA 9",
                            line={"color": "#fbbf24", "width": 1},
                        ),
                        row=1,
                        col=1,
                    )
                if "ema_21" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["ema_21"],
                            name="EMA 21",
                            line={"color": "#60a5fa", "width": 1},
                        ),
                        row=1,
                        col=1,
                    )
                if "ema_50" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["ema_50"],
                            name="EMA 50",
                            line={"color": "#f472b6", "width": 1},
                        ),
                        row=1,
                        col=1,
                    )

                # Bollinger Bands
                if "bb_upper" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["bb_upper"],
                            name="BB Upper",
                            line={"color": "rgba(255,255,255,0.3)", "width": 1},
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["bb_lower"],
                            name="BB Lower",
                            line={"color": "rgba(255,255,255,0.3)", "width": 1},
                            fill="tonexty",
                            fillcolor="rgba(255,255,255,0.05)",
                        ),
                        row=1,
                        col=1,
                    )

                # RSI
                if "rsi" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["rsi"],
                            name="RSI",
                            line={"color": "#a78bfa", "width": 1.5},
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_hline(
                        y=70, line_dash="dash", line_color="red",
                        opacity=0.5, row=2, col=1,
                    )
                    fig.add_hline(
                        y=30, line_dash="dash", line_color="green",
                        opacity=0.5, row=2, col=1,
                    )

                # MACD
                if "macd" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["macd"],
                            name="MACD",
                            line={"color": "#4ade80", "width": 1.5},
                        ),
                        row=3,
                        col=1,
                    )
                if "macd_signal" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["macd_signal"],
                            name="Signal",
                            line={"color": "#f87171", "width": 1.5},
                        ),
                        row=3,
                        col=1,
                    )
                if "macd_histogram" in df.columns:
                    colors = [
                        "#4ade80" if v >= 0 else "#f87171"
                        for v in df["macd_histogram"]
                    ]
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df["macd_histogram"],
                            name="Histogram",
                            marker_color=colors,
                        ),
                        row=3,
                        col=1,
                    )

                fig.update_layout(
                    template="plotly_dark",
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Symbol Details Tab ---
with tab_details:
    st.subheader("Symbol Details & Features")

    if bot.cycle_count == 0:
        st.info("Run an analysis cycle first.")
    else:
        state = bot.get_full_state()
        detail_symbols = [
            s for s in SYMBOLS if state["signals"].get(s, {}).get("price", 0) > 0
        ]

        if detail_symbols:
            selected = st.selectbox(
                "Select symbol for details",
                detail_symbols,
                key="detail_select",
            )

            sym_state = bot.states.get(selected)
            if sym_state and sym_state.features:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Price Info**")
                    st.write(f"Price: ${sym_state.price:,.4f}")
                    st.write(
                        f"Change: "
                        f"{sym_state.features.get('price_change_pct', 0):.2f}%"
                    )
                    st.write(
                        f"ATR: {sym_state.features.get('atr', 0):.4f} "
                        f"({sym_state.features.get('atr_pct', 0):.2f}%)"
                    )

                with col2:
                    st.markdown("**Moving Averages**")
                    st.write(
                        f"EMA 9: "
                        f"${sym_state.features.get('ema_9', 0):,.4f}"
                    )
                    st.write(
                        f"EMA 21: "
                        f"${sym_state.features.get('ema_21', 0):,.4f}"
                    )
                    st.write(
                        f"EMA 50: "
                        f"${sym_state.features.get('ema_50', 0):,.4f}"
                    )
                    cross = sym_state.features.get("ema_cross", 0)
                    st.write(
                        f"EMA Cross: "
                        f"{'Bullish' if cross == 1 else 'Bearish'}"
                    )

                with col3:
                    st.markdown("**Oscillators**")
                    rsi = sym_state.features.get("rsi", 0)
                    st.write(f"RSI: {rsi:.1f}")
                    st.write(
                        f"MACD: "
                        f"{sym_state.features.get('macd', 0):.4f}"
                    )
                    st.write(
                        f"MACD Hist: "
                        f"{sym_state.features.get('macd_histogram', 0):.4f}"
                    )
                    st.write(
                        f"BB Position: "
                        f"{sym_state.features.get('bb_position', 0):.2f}"
                    )
                    st.write(
                        f"Volume Ratio: "
                        f"{sym_state.features.get('volume_ratio', 0):.2f}x"
                    )

                if sym_state.signal:
                    st.divider()
                    st.markdown("**ML Signal**")
                    st.write(sym_state.signal.reasoning)

# --- Recent Events ---
st.divider()
with st.expander("Recent Bot Events", expanded=False):
    if bot.events:
        for event in reversed(bot.events[-20:]):
            st.json(event)
    else:
        st.info("No events yet.")

# --- Auto-refresh ---
if st.session_state.auto_trade:
    time.sleep(DASHBOARD_REFRESH_INTERVAL)
    run_cycle()
    st.rerun()
