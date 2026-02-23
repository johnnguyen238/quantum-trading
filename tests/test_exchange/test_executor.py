"""Tests for the order executor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.config.constants import OrderSide, OrderStatus, OrderType
from src.exchange.executor import OrderExecutor, _map_ccxt_status


@pytest.fixture
def mock_client():
    """Mock BybitClient for executor tests."""
    client = AsyncMock()
    client.create_order = AsyncMock(
        return_value={
            "id": "order-001",
            "status": "closed",
            "amount": 0.1,
            "price": 42000.0,
            "filled": 0.1,
            "average": 42000.0,
            "fee": {"cost": 2.52, "currency": "USDT"},
        }
    )
    client.cancel_order = AsyncMock(return_value={"id": "order-001", "status": "canceled"})
    client.fetch_order = AsyncMock(
        return_value={
            "id": "order-001",
            "status": "closed",
            "filled": 0.1,
            "average": 42000.0,
            "fee": {"cost": 2.52},
        }
    )
    return client


class TestMapCcxtStatus:
    def test_map_known_statuses(self):
        assert _map_ccxt_status("open") == OrderStatus.OPEN
        assert _map_ccxt_status("closed") == OrderStatus.FILLED
        assert _map_ccxt_status("canceled") == OrderStatus.CANCELLED
        assert _map_ccxt_status("cancelled") == OrderStatus.CANCELLED
        assert _map_ccxt_status("rejected") == OrderStatus.REJECTED

    def test_map_unknown_status(self):
        assert _map_ccxt_status("something_else") == OrderStatus.PENDING


class TestOrderExecutor:
    @pytest.mark.asyncio
    async def test_execute_market_order(self, mock_client):
        executor = OrderExecutor(mock_client)
        trade = await executor.execute_market_order("BTC/USDT:USDT", "buy", 0.1)

        assert trade.symbol == "BTC/USDT:USDT"
        assert trade.side == OrderSide.BUY
        assert trade.order_type == OrderType.MARKET
        assert trade.amount == 0.1
        assert trade.status == OrderStatus.FILLED
        assert trade.exchange_order_id == "order-001"
        assert trade.filled_amount == 0.1
        assert trade.fee == 2.52
        mock_client.create_order.assert_awaited_once_with(
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="market",
            amount=0.1,
        )

    @pytest.mark.asyncio
    async def test_execute_limit_order(self, mock_client):
        mock_client.create_order = AsyncMock(
            return_value={
                "id": "order-002",
                "status": "open",
                "amount": 0.1,
                "price": 41500.0,
                "filled": 0.0,
                "average": None,
                "fee": None,
            }
        )
        executor = OrderExecutor(mock_client)
        trade = await executor.execute_limit_order("BTC/USDT:USDT", "buy", 0.1, 41500.0)

        assert trade.order_type == OrderType.LIMIT
        assert trade.status == OrderStatus.OPEN
        assert trade.filled_amount == 0.0

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, mock_client):
        executor = OrderExecutor(mock_client)
        result = await executor.cancel_order("order-001", "BTC/USDT:USDT")
        assert result is True
        mock_client.cancel_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, mock_client):
        mock_client.cancel_order = AsyncMock(side_effect=Exception("Order not found"))
        executor = OrderExecutor(mock_client)
        result = await executor.cancel_order("order-999", "BTC/USDT:USDT")
        assert result is False

    @pytest.mark.asyncio
    async def test_sync_order_status(self, mock_client):
        from datetime import datetime, timezone

        from src.data.models import Trade

        trade = Trade(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=42000.0,
            timestamp=datetime.now(timezone.utc),
            status=OrderStatus.OPEN,
            exchange_order_id="order-001",
        )

        executor = OrderExecutor(mock_client)
        updated = await executor.sync_order_status(trade)

        assert updated.status == OrderStatus.FILLED
        assert updated.filled_amount == 0.1
        assert updated.filled_price == 42000.0
        assert updated.fee == 2.52

    @pytest.mark.asyncio
    async def test_close_position_long(self, mock_client):
        executor = OrderExecutor(mock_client)
        trade = await executor.close_position("BTC/USDT:USDT", "buy", 0.1)
        # Closing a long position should place a sell order
        mock_client.create_order.assert_awaited_once_with(
            symbol="BTC/USDT:USDT",
            side="sell",
            order_type="market",
            amount=0.1,
        )
        assert trade.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_close_position_short(self, mock_client):
        executor = OrderExecutor(mock_client)
        trade = await executor.close_position("BTC/USDT:USDT", "sell", 0.1)
        # Closing a short position should place a buy order
        mock_client.create_order.assert_awaited_once_with(
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="market",
            amount=0.1,
        )
        assert trade.side == OrderSide.BUY
