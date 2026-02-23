"""Tests for the Bybit exchange client (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import ExchangeSettings
from src.exchange.client import BybitClient


@pytest.fixture
def exchange_settings() -> ExchangeSettings:
    return ExchangeSettings(
        name="bybit",
        testnet=True,
        api_key="test_key",
        api_secret="test_secret",
        rate_limit=50,
    )


@pytest.fixture
def mock_ccxt():
    """Create a mock ccxt bybit exchange instance."""
    mock = AsyncMock()
    mock.markets = {"BTC/USDT:USDT": {"id": "BTCUSDT"}}
    mock.load_markets = AsyncMock(return_value=mock.markets)
    mock.set_sandbox_mode = MagicMock()
    mock.close = AsyncMock()
    mock.fetch_balance = AsyncMock(return_value={"USDT": {"free": 10000.0, "total": 10000.0}})
    mock.fetch_ticker = AsyncMock(
        return_value={"symbol": "BTC/USDT:USDT", "last": 42000.0, "bid": 41999.0, "ask": 42001.0}
    )
    mock.fetch_ohlcv = AsyncMock(
        return_value=[
            [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1704068100000, 42050.0, 42150.0, 41950.0, 42100.0, 110.0],
        ]
    )
    mock.fetch_positions = AsyncMock(return_value=[])
    mock.set_leverage = AsyncMock()
    mock.create_order = AsyncMock(
        return_value={
            "id": "order-123",
            "status": "closed",
            "amount": 0.1,
            "price": 42000.0,
            "filled": 0.1,
            "average": 42000.0,
            "fee": {"cost": 2.52, "currency": "USDT"},
        }
    )
    mock.cancel_order = AsyncMock(return_value={"id": "order-123", "status": "canceled"})
    mock.fetch_order = AsyncMock(
        return_value={
            "id": "order-123",
            "status": "closed",
            "filled": 0.1,
            "average": 42000.0,
            "fee": {"cost": 2.52},
        }
    )
    return mock


class TestBybitClient:
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            client = BybitClient(exchange_settings)
            await client.connect()
            assert client._exchange is not None
            mock_ccxt.set_sandbox_mode.assert_called_once_with(True)
            mock_ccxt.load_markets.assert_awaited_once()

            await client.disconnect()
            assert client._exchange is None
            mock_ccxt.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exchange_property_raises_when_not_connected(self, exchange_settings):
        client = BybitClient(exchange_settings)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.exchange

    @pytest.mark.asyncio
    async def test_fetch_balance(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                balance = await client.fetch_balance()
                assert balance["USDT"]["free"] == 10000.0

    @pytest.mark.asyncio
    async def test_fetch_ticker(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                ticker = await client.fetch_ticker("BTC/USDT:USDT")
                assert ticker["last"] == 42000.0

    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                candles = await client.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=2)
                assert len(candles) == 2
                assert candles[0][4] == 42050.0  # close price

    @pytest.mark.asyncio
    async def test_create_order(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                order = await client.create_order(
                    "BTC/USDT:USDT", "buy", "market", 0.1
                )
                assert order["id"] == "order-123"
                mock_ccxt.create_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                result = await client.cancel_order("order-123", "BTC/USDT:USDT")
                assert result["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_set_leverage(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                await client.set_leverage("BTC/USDT:USDT", 5)
                mock_ccxt.set_leverage.assert_awaited_once_with(5, "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_context_manager(self, exchange_settings, mock_ccxt):
        with patch("src.exchange.client.ccxt_async.bybit", return_value=mock_ccxt):
            async with BybitClient(exchange_settings) as client:
                assert client._exchange is not None
            mock_ccxt.close.assert_awaited_once()
