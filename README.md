# Crypto Futures Algorithmic Trading Bot

End-to-end simulated crypto futures trading system with ML-driven signals and a Streamlit dashboard. Runs entirely without API keys.

## Features

- **Public Data Ingestion**: Real-time OHLCV data for top 20 USDT perpetual futures via CCXT (Binance public API)
- **Paper Trading Engine**: Custom `MockExchange` class with $10,000 simulated balance, long/short futures, stop-loss, take-profit, and liquidation logic
- **Technical Analysis**: EMA, RSI, MACD, ATR, Bollinger Bands via pandas-ta
- **ML Signal Engine**: Random Forest classifier generating LONG/SHORT/HOLD signals with confidence scores and explainable reasoning
- **Streamlit Dashboard**: Real-time monitoring of signals, positions, trade history, and interactive price charts

## Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Architecture

```
src/
  config.py            - Global configuration
  data_ingestion.py    - CCXT public OHLCV fetcher
  mock_exchange.py     - Paper trading engine (MockExchange)
  technical_analysis.py - pandas-ta indicator computation
  ml_engine.py         - Random Forest signal generator
  coingecko.py         - CoinGecko public API client
  trading_bot.py       - Main orchestrator
dashboard.py           - Streamlit UI
```

## No API Keys Required

The system uses only public exchange endpoints (CCXT) and a locally trained ML model. No authentication or paid APIs needed.
