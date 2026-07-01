---
name: testing-crypto-dashboard
description: Test the Crypto Futures Trading Bot Streamlit dashboard end-to-end. Use when verifying dashboard UI, ML signals, mock trading, or chart rendering.
---

# Testing the Crypto Futures Trading Bot Dashboard

## Prerequisites

- Python 3.10+ with dependencies from `requirements.txt`
- No API keys or secrets required — all public endpoints

## Setup

```bash
cd /path/to/crypto-futures-bot
pip install -r requirements.txt
streamlit run dashboard.py --server.port 8501 --server.headless true &
# Wait for "Uvicorn server started on 0.0.0.0:8501"
```

## Exchange Connectivity

- Primary exchange: OKX (configured in `src/config.py`)
- Fallback: Binance, then Bybit
- Binance might be geo-restricted from some VMs (HTTP 451). The `DataIngestion` class handles this automatically with fallback logic.
- If all exchanges fail, check network connectivity and consider adding another exchange to `FALLBACK_EXCHANGE_IDS` in `src/config.py`.

## Key Test Flows

### 1. Initial State Verification
Navigate to `http://localhost:8501`. Verify:
- Balance = $10,000.00, Equity = $10,000.00
- Open Positions = 0, Win Rate = 0.0%, Return = +0.00%
- ML Signals tab shows "No signals yet" info message

### 2. Run Analysis Cycle
Click "Run Analysis Cycle" button in sidebar. This fetches OHLCV data for 20 symbols — takes ~15-30 seconds. After completion:
- ML Signals table populates with 15-20 rows
- Each row has Symbol, Price, Signal (LONG/SHORT/HOLD), Confidence, Reasoning
- Sidebar balance may decrease (margin deducted for opened positions)
- Signal colors: green=LONG, red=SHORT, yellow=HOLD

### 3. Open Positions Tab
If signals triggered trades (confidence > 55%), positions appear with:
- Symbol, Side, Entry Price, Quantity, Leverage (5x), Margin, Unrealized PnL, Stop Loss, Take Profit

### 4. Charts Tab
Select a symbol from dropdown. Verify 3 Plotly subplots:
- Candlestick with EMA 9/21/50 and Bollinger Bands
- RSI with 30/70 threshold lines
- MACD with histogram and signal line

### 5. Symbol Details Tab
Select a symbol. Verify 3-column layout:
- Price Info (price, change %, ATR)
- Moving Averages (EMA 9/21/50, EMA Cross direction)
- Oscillators (RSI, MACD, BB Position, Volume Ratio)
- ML Signal reasoning text below

## Integration Test

Run `python test_integration.py` for a quick non-GUI smoke test of all modules.

## Common Issues

- If the dashboard shows a Streamlit error about `st.set_page_config`, ensure it's the first Streamlit command in `dashboard.py`.
- The ML model auto-trains on synthetic data on first run and saves to `models/signal_model.joblib`. Delete this file to force retraining.
- The analysis cycle fetches data sequentially for 20 symbols. If it's slow, reduce `SYMBOLS` list in `src/config.py`.

## Devin Secrets Needed

None — this system runs entirely on public APIs.
