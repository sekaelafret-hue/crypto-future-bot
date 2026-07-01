"""CoinGecko public API for fundamental/metadata."""

import logging
from typing import Optional

import requests

from src.config import COINGECKO_BASE_URL

logger = logging.getLogger(__name__)

# Map trading symbols to CoinGecko IDs
SYMBOL_TO_COINGECKO = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "BNB/USDT": "binancecoin",
    "SOL/USDT": "solana",
    "XRP/USDT": "ripple",
    "DOGE/USDT": "dogecoin",
    "ADA/USDT": "cardano",
    "AVAX/USDT": "avalanche-2",
    "DOT/USDT": "polkadot",
    "LINK/USDT": "chainlink",
    "MATIC/USDT": "matic-network",
    "UNI/USDT": "uniswap",
    "LTC/USDT": "litecoin",
    "ATOM/USDT": "cosmos",
    "ETC/USDT": "ethereum-classic",
    "FIL/USDT": "filecoin",
    "APT/USDT": "aptos",
    "ARB/USDT": "arbitrum",
    "OP/USDT": "optimism",
    "NEAR/USDT": "near",
}


class CoinGeckoClient:
    """Fetch metadata from CoinGecko public API."""

    def __init__(self):
        self.base_url = COINGECKO_BASE_URL
        self._cache: dict[str, dict] = {}

    def get_coin_data(self, symbol: str) -> Optional[dict]:
        """Get market data for a coin from CoinGecko."""
        coin_id = SYMBOL_TO_COINGECKO.get(symbol)
        if not coin_id:
            return None

        if coin_id in self._cache:
            return self._cache[coin_id]

        try:
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                result = {
                    "name": data.get("name", ""),
                    "symbol": data.get("symbol", ""),
                    "market_cap_rank": data.get("market_cap_rank"),
                    "market_cap": data.get("market_data", {})
                    .get("market_cap", {})
                    .get("usd"),
                    "total_volume_24h": data.get("market_data", {})
                    .get("total_volume", {})
                    .get("usd"),
                    "price_change_24h_pct": data.get("market_data", {}).get(
                        "price_change_percentage_24h"
                    ),
                    "price_change_7d_pct": data.get("market_data", {}).get(
                        "price_change_percentage_7d"
                    ),
                    "ath": data.get("market_data", {})
                    .get("ath", {})
                    .get("usd"),
                    "ath_change_pct": data.get("market_data", {})
                    .get("ath_change_percentage", {})
                    .get("usd"),
                }
                self._cache[coin_id] = result
                return result
            elif resp.status_code == 429:
                logger.warning("CoinGecko rate limit hit")
                return None
            else:
                logger.warning(
                    "CoinGecko returned status %d for %s",
                    resp.status_code,
                    coin_id,
                )
                return None
        except Exception as e:
            logger.error("CoinGecko request failed for %s: %s", coin_id, e)
            return None

    def get_trending(self) -> list[dict]:
        """Get trending coins from CoinGecko."""
        try:
            url = f"{self.base_url}/search/trending"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "name": coin["item"]["name"],
                        "symbol": coin["item"]["symbol"],
                        "market_cap_rank": coin["item"].get("market_cap_rank"),
                    }
                    for coin in data.get("coins", [])[:10]
                ]
            return []
        except Exception as e:
            logger.error("CoinGecko trending request failed: %s", e)
            return []
