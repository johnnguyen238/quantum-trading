"""Async ccxt Bybit wrapper for exchange connectivity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import ccxt.async_support as ccxt_async

if TYPE_CHECKING:
    from src.config.settings import ExchangeSettings

logger = logging.getLogger(__name__)


class BybitClient:
    """Async wrapper around ccxt's Bybit exchange.

    Parameters
    ----------
    settings:
        Exchange configuration (API keys, testnet flag, rate limit).
    """

    def __init__(self, settings: "ExchangeSettings") -> None:
        self._settings = settings
        self._exchange: ccxt_async.bybit | None = None

    @property
    def exchange(self) -> ccxt_async.bybit:
        """Return the underlying ccxt exchange instance."""
        if self._exchange is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._exchange

    async def connect(self) -> None:
        """Initialize the ccxt async Bybit instance."""
        import aiohttp

        # Use threaded resolver — aiohttp's AsyncResolver (c-ares) can fail
        # on some Windows configurations even when system DNS works fine.
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        )
        self._exchange = ccxt_async.bybit(
            {
                "apiKey": self._settings.api_key,
                "secret": self._settings.api_secret,
                "enableRateLimit": True,
                "rateLimit": self._settings.rate_limit,
                "session": session,
                "options": {
                    "defaultType": "swap",
                },
            }
        )
        if self._settings.testnet:
            self._exchange.set_sandbox_mode(True)
        await self._exchange.load_markets()
        logger.info(
            "Connected to Bybit %s (%d markets loaded)",
            "testnet" if self._settings.testnet else "mainnet",
            len(self._exchange.markets),
        )

    async def disconnect(self) -> None:
        """Close the ccxt connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
            logger.info("Disconnected from Bybit")

    async def fetch_balance(self) -> dict[str, Any]:
        """Fetch account balance."""
        return await self.exchange.fetch_balance()

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker for a symbol."""
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: int | None = None,
        limit: int = 100,
    ) -> list[list]:
        """Fetch OHLCV candles from the exchange.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT:USDT"``).
        timeframe:
            Candle timeframe (e.g. ``"15m"``).
        since:
            Start timestamp in milliseconds. None = latest.
        limit:
            Number of candles to fetch (max 200 for Bybit).

        Returns
        -------
        List of ``[timestamp_ms, open, high, low, close, volume]``.
        """
        return await self.exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit
        )

    async def fetch_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch open positions from the exchange."""
        symbols = [symbol] if symbol else None
        return await self.exchange.fetch_positions(symbols)

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            await self.exchange.set_leverage(leverage, symbol)
            logger.info("Leverage set to %dx for %s", leverage, symbol)
        except ccxt_async.ExchangeError as e:
            # Bybit may error if leverage is already set to same value
            logger.warning("set_leverage for %s: %s", symbol, e)

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Place an order on Bybit."""
        result = await self.exchange.create_order(
            symbol, order_type, side, amount, price, params or {}
        )
        logger.info(
            "Order placed: %s %s %s %.4f @ %s → id=%s",
            symbol,
            side,
            order_type,
            amount,
            price or "market",
            result.get("id"),
        )
        return result

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an open order."""
        result = await self.exchange.cancel_order(order_id, symbol)
        logger.info("Order cancelled: %s on %s", order_id, symbol)
        return result

    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Fetch order status."""
        return await self.exchange.fetch_order(order_id, symbol)

    async def __aenter__(self) -> "BybitClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.disconnect()
