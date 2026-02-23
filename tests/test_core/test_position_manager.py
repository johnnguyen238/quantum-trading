"""Tests for the position manager."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.constants import OrderSide, OrderStatus, OrderType, PositionState
from src.core.position_manager import PositionManager
from src.data.models import Position, Trade


def _make_trade(
    side: OrderSide = OrderSide.BUY,
    amount: float = 0.01,
    price: float = 42000.0,
    filled_amount: float = 0.01,
    filled_price: float = 42000.0,
) -> Trade:
    return Trade(
        id=1,
        symbol="BTC/USDT:USDT",
        side=side,
        order_type=OrderType.MARKET,
        amount=amount,
        price=price,
        timestamp=datetime.now(timezone.utc),
        status=OrderStatus.FILLED,
        exchange_order_id="ord-123",
        filled_amount=filled_amount,
        filled_price=filled_price,
    )


def _make_position(
    side: OrderSide = OrderSide.BUY,
    entry_price: float = 42000.0,
    amount: float = 0.1,
    dca_count: int = 0,
    position_id: int = 1,
) -> Position:
    return Position(
        id=position_id,
        symbol="BTC/USDT:USDT",
        side=side,
        state=PositionState.OPEN,
        entry_price=entry_price,
        current_price=42500.0,
        amount=amount,
        leverage=5,
        unrealized_pnl=50.0,
        dca_count=dca_count,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.get_open_positions = AsyncMock(return_value=[])
    repo.save_position = AsyncMock(return_value=1)
    repo.update_position = AsyncMock()
    repo.update_trade_status = AsyncMock()
    return repo


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.fetch_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture
def pos_mgr(mock_repo, mock_client):
    return PositionManager(mock_repo, mock_client)


class TestGetOpenPositions:
    @pytest.mark.asyncio
    async def test_delegates_to_repo(self, pos_mgr, mock_repo):
        expected = [_make_position()]
        mock_repo.get_open_positions = AsyncMock(return_value=expected)
        result = await pos_mgr.get_open_positions("BTC/USDT:USDT")
        mock_repo.get_open_positions.assert_called_once_with("BTC/USDT:USDT")
        assert result == expected

    @pytest.mark.asyncio
    async def test_without_symbol(self, pos_mgr, mock_repo):
        await pos_mgr.get_open_positions()
        mock_repo.get_open_positions.assert_called_once_with(None)


class TestOpenPosition:
    @pytest.mark.asyncio
    async def test_creates_position_from_trade(self, pos_mgr, mock_repo):
        trade = _make_trade(
            side=OrderSide.BUY, filled_amount=0.01, filled_price=42000.0
        )
        position = await pos_mgr.open_position(trade, leverage=5)

        assert position.symbol == "BTC/USDT:USDT"
        assert position.side == OrderSide.BUY
        assert position.state == PositionState.OPEN
        assert position.entry_price == 42000.0
        assert position.amount == 0.01
        assert position.leverage == 5
        assert position.dca_count == 0
        mock_repo.save_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_links_trade_to_position(self, pos_mgr, mock_repo):
        mock_repo.save_position = AsyncMock(return_value=42)
        trade = _make_trade()
        await pos_mgr.open_position(trade, leverage=3)
        assert trade.position_id == 42
        mock_repo.update_trade_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_price_when_no_filled_price(self, pos_mgr, mock_repo):
        trade = _make_trade(filled_amount=0.0, filled_price=0.0)
        trade.price = 41500.0
        trade.amount = 0.05
        position = await pos_mgr.open_position(trade, leverage=2)
        assert position.entry_price == 41500.0
        assert position.amount == 0.05

    @pytest.mark.asyncio
    async def test_short_position(self, pos_mgr, mock_repo):
        trade = _make_trade(side=OrderSide.SELL)
        position = await pos_mgr.open_position(trade, leverage=10)
        assert position.side == OrderSide.SELL


class TestUpdateFromTrade:
    @pytest.mark.asyncio
    async def test_dca_updates_average_and_amount(self, pos_mgr, mock_repo):
        position = _make_position(entry_price=42000.0, amount=0.1, dca_count=0)
        trade = _make_trade(
            side=OrderSide.BUY, filled_amount=0.1, filled_price=41000.0
        )

        updated = await pos_mgr.update_from_trade(position, trade)
        # weighted avg: (42000*0.1 + 41000*0.1) / 0.2 = 41500
        assert updated.entry_price == pytest.approx(41500.0)
        assert updated.amount == pytest.approx(0.2)
        assert updated.dca_count == 1
        mock_repo.update_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_dca_increments_count(self, pos_mgr, mock_repo):
        position = _make_position(dca_count=1)
        trade = _make_trade(filled_amount=0.05, filled_price=41500.0)
        updated = await pos_mgr.update_from_trade(position, trade)
        assert updated.dca_count == 2

    @pytest.mark.asyncio
    async def test_marks_trade_as_dca(self, pos_mgr, mock_repo):
        position = _make_position(dca_count=0)
        trade = _make_trade()
        await pos_mgr.update_from_trade(position, trade)
        assert trade.is_dca is True
        assert trade.dca_layer == 1  # after increment from 0

    @pytest.mark.asyncio
    async def test_links_trade_to_position(self, pos_mgr, mock_repo):
        position = _make_position(position_id=7)
        trade = _make_trade()
        await pos_mgr.update_from_trade(position, trade)
        assert trade.position_id == 7


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_long_profitable(self, pos_mgr, mock_repo):
        position = _make_position(
            side=OrderSide.BUY, entry_price=42000.0, amount=0.1
        )
        position.leverage = 5
        trade = _make_trade(filled_price=43000.0)

        closed = await pos_mgr.close_position(position, trade)
        # PnL = (43000 - 42000) * 0.1 * 5 = 500
        assert closed.realized_pnl == pytest.approx(500.0)
        assert closed.state == PositionState.CLOSED
        assert closed.unrealized_pnl == 0.0
        assert closed.closed_at is not None
        mock_repo.update_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_long_losing(self, pos_mgr, mock_repo):
        position = _make_position(
            side=OrderSide.BUY, entry_price=42000.0, amount=0.1
        )
        position.leverage = 5
        trade = _make_trade(filled_price=41000.0)

        closed = await pos_mgr.close_position(position, trade)
        # PnL = (41000 - 42000) * 0.1 * 5 = -500
        assert closed.realized_pnl == pytest.approx(-500.0)

    @pytest.mark.asyncio
    async def test_close_short_profitable(self, pos_mgr, mock_repo):
        position = _make_position(
            side=OrderSide.SELL, entry_price=42000.0, amount=0.1
        )
        position.leverage = 5
        trade = _make_trade(filled_price=41000.0)

        closed = await pos_mgr.close_position(position, trade)
        # PnL = (42000 - 41000) * 0.1 * 5 = 500
        assert closed.realized_pnl == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_close_short_losing(self, pos_mgr, mock_repo):
        position = _make_position(
            side=OrderSide.SELL, entry_price=42000.0, amount=0.1
        )
        position.leverage = 5
        trade = _make_trade(filled_price=43000.0)

        closed = await pos_mgr.close_position(position, trade)
        # PnL = (42000 - 43000) * 0.1 * 5 = -500
        assert closed.realized_pnl == pytest.approx(-500.0)

    @pytest.mark.asyncio
    async def test_links_trade_to_position(self, pos_mgr, mock_repo):
        position = _make_position(position_id=10)
        trade = _make_trade()
        await pos_mgr.close_position(position, trade)
        assert trade.position_id == 10


class TestSyncWithExchange:
    @pytest.mark.asyncio
    async def test_updates_unrealized_pnl(self, pos_mgr, mock_repo, mock_client):
        position = _make_position()
        mock_repo.get_open_positions = AsyncMock(return_value=[position])
        mock_client.fetch_positions = AsyncMock(
            return_value=[
                {
                    "side": "long",
                    "contracts": 0.1,
                    "markPrice": 43000.0,
                    "unrealizedPnl": 100.0,
                }
            ]
        )

        result = await pos_mgr.sync_with_exchange("BTC/USDT:USDT")
        assert len(result) == 1
        assert result[0].current_price == 43000.0
        assert result[0].unrealized_pnl == 100.0
        mock_repo.update_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_local_positions(self, pos_mgr, mock_repo, mock_client):
        mock_repo.get_open_positions = AsyncMock(return_value=[])
        result = await pos_mgr.sync_with_exchange("BTC/USDT:USDT")
        assert result == []
        mock_client.fetch_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_position_sync(self, pos_mgr, mock_repo, mock_client):
        position = _make_position(side=OrderSide.SELL)
        mock_repo.get_open_positions = AsyncMock(return_value=[position])
        mock_client.fetch_positions = AsyncMock(
            return_value=[
                {
                    "side": "short",
                    "contracts": 0.1,
                    "markPrice": 41000.0,
                    "unrealizedPnl": 100.0,
                }
            ]
        )

        result = await pos_mgr.sync_with_exchange("BTC/USDT:USDT")
        assert result[0].current_price == 41000.0
        assert result[0].unrealized_pnl == 100.0

    @pytest.mark.asyncio
    async def test_no_matching_exchange_position(
        self, pos_mgr, mock_repo, mock_client
    ):
        position = _make_position()
        mock_repo.get_open_positions = AsyncMock(return_value=[position])
        mock_client.fetch_positions = AsyncMock(return_value=[])

        result = await pos_mgr.sync_with_exchange("BTC/USDT:USDT")
        assert len(result) == 1
        # Position was not updated since there's no exchange match
        mock_repo.update_position.assert_not_called()
