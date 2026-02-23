"""Tests for the order manager."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.constants import OrderSide, OrderStatus, OrderType
from src.core.order_manager import OrderManager
from src.data.models import Trade
from src.strategy.base import TradeAction


def _make_trade(
    side: str = "buy",
    status: OrderStatus = OrderStatus.FILLED,
    exchange_order_id: str = "ord-123",
) -> Trade:
    return Trade(
        symbol="BTC/USDT:USDT",
        side=OrderSide(side),
        order_type=OrderType.MARKET,
        amount=0.01,
        price=42000.0,
        timestamp=datetime.now(timezone.utc),
        status=status,
        exchange_order_id=exchange_order_id,
        filled_amount=0.01,
        filled_price=42000.0,
    )


def _make_action(
    action: str = "open_long",
    amount: float = 0.01,
    symbol: str = "BTC/USDT:USDT",
    reason: str = "",
) -> TradeAction:
    return TradeAction(
        action=action,
        symbol=symbol,
        amount=amount,
        price=42000.0,
        reason=reason,
    )


@pytest.fixture
def mock_executor():
    executor = MagicMock()
    executor.execute_market_order = AsyncMock(return_value=_make_trade())
    executor.close_position = AsyncMock(return_value=_make_trade(side="sell"))
    executor.sync_order_status = AsyncMock(return_value=_make_trade())
    return executor


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.save_trade = AsyncMock(return_value=1)
    repo.update_trade_status = AsyncMock()
    return repo


@pytest.fixture
def order_mgr(mock_executor, mock_repo):
    return OrderManager(mock_executor, mock_repo)


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_open_long(self, order_mgr, mock_executor, mock_repo):
        action = _make_action("open_long")
        trade = await order_mgr.submit(action)
        mock_executor.execute_market_order.assert_called_once_with(
            symbol="BTC/USDT:USDT", side="buy", amount=0.01
        )
        mock_repo.save_trade.assert_called_once()
        assert trade is not None

    @pytest.mark.asyncio
    async def test_submit_open_short(self, order_mgr, mock_executor):
        action = _make_action("open_short")
        await order_mgr.submit(action)
        mock_executor.execute_market_order.assert_called_once_with(
            symbol="BTC/USDT:USDT", side="sell", amount=0.01
        )

    @pytest.mark.asyncio
    async def test_submit_dca_long(self, order_mgr, mock_executor):
        action = _make_action("dca_long", amount=0.02)
        trade = await order_mgr.submit(action)
        mock_executor.execute_market_order.assert_called_once_with(
            symbol="BTC/USDT:USDT", side="buy", amount=0.02
        )
        assert trade.is_dca is True

    @pytest.mark.asyncio
    async def test_submit_dca_short(self, order_mgr, mock_executor):
        action = _make_action("dca_short", amount=0.03)
        trade = await order_mgr.submit(action)
        mock_executor.execute_market_order.assert_called_once_with(
            symbol="BTC/USDT:USDT", side="sell", amount=0.03
        )
        assert trade.is_dca is True

    @pytest.mark.asyncio
    async def test_submit_close_long_position(self, order_mgr, mock_executor):
        action = _make_action("close", reason="Trend reversal: buy → SHORT")
        await order_mgr.submit(action)
        mock_executor.close_position.assert_called_once_with(
            symbol="BTC/USDT:USDT", side="buy", amount=0.01
        )

    @pytest.mark.asyncio
    async def test_submit_close_short_position(self, order_mgr, mock_executor):
        action = _make_action("close", reason="Trend reversal: sell → LONG")
        await order_mgr.submit(action)
        mock_executor.close_position.assert_called_once_with(
            symbol="BTC/USDT:USDT", side="sell", amount=0.01
        )

    @pytest.mark.asyncio
    async def test_submit_unknown_action_raises(self, order_mgr):
        action = _make_action("unknown_action")
        with pytest.raises(ValueError, match="Unknown action type"):
            await order_mgr.submit(action)


class TestSubmitBatch:
    @pytest.mark.asyncio
    async def test_submit_batch_multiple(self, order_mgr, mock_executor):
        actions = [
            _make_action("open_long"),
            _make_action("open_short"),
        ]
        trades = await order_mgr.submit_batch(actions)
        assert len(trades) == 2
        assert mock_executor.execute_market_order.call_count == 2

    @pytest.mark.asyncio
    async def test_submit_batch_empty(self, order_mgr):
        trades = await order_mgr.submit_batch([])
        assert trades == []

    @pytest.mark.asyncio
    async def test_submit_batch_continues_on_error(self, mock_executor, mock_repo):
        # First call raises, second succeeds
        mock_executor.execute_market_order = AsyncMock(
            side_effect=[RuntimeError("API error"), _make_trade()]
        )
        mgr = OrderManager(mock_executor, mock_repo)
        actions = [_make_action("open_long"), _make_action("open_short")]
        trades = await mgr.submit_batch(actions)
        assert len(trades) == 1


class TestSyncTrade:
    @pytest.mark.asyncio
    async def test_sync_updates_status(self, order_mgr, mock_executor, mock_repo):
        trade = _make_trade(status=OrderStatus.OPEN)
        trade.id = 1

        filled_trade = _make_trade(status=OrderStatus.FILLED)
        mock_executor.sync_order_status = AsyncMock(return_value=filled_trade)

        result = await order_mgr.sync_trade(trade)
        mock_executor.sync_order_status.assert_called_once_with(trade)
        mock_repo.update_trade_status.assert_called_once_with(1, filled_trade)
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_sync_skips_no_exchange_id(self, order_mgr, mock_executor):
        trade = _make_trade()
        trade.exchange_order_id = ""
        result = await order_mgr.sync_trade(trade)
        mock_executor.sync_order_status.assert_not_called()
        assert result is trade

    @pytest.mark.asyncio
    async def test_sync_no_update_when_unchanged(
        self, order_mgr, mock_executor, mock_repo
    ):
        trade = _make_trade(status=OrderStatus.FILLED)
        trade.id = 1
        mock_executor.sync_order_status = AsyncMock(
            return_value=_make_trade(status=OrderStatus.FILLED)
        )
        await order_mgr.sync_trade(trade)
        mock_repo.update_trade_status.assert_not_called()


class TestInferCloseSide:
    def test_infer_buy_from_reason(self):
        action = _make_action("close", reason="closing buy position")
        assert OrderManager._infer_close_side(action) == "buy"

    def test_infer_sell_from_reason(self):
        action = _make_action("close", reason="closing sell position")
        assert OrderManager._infer_close_side(action) == "sell"

    def test_infer_long_from_reason(self):
        action = _make_action("close", reason="Trend reversal: long → SHORT")
        assert OrderManager._infer_close_side(action) == "buy"

    def test_infer_short_from_reason(self):
        action = _make_action("close", reason="Trend reversal: short → LONG")
        assert OrderManager._infer_close_side(action) == "sell"

    def test_defaults_to_buy(self):
        action = _make_action("close", reason="")
        assert OrderManager._infer_close_side(action) == "buy"
