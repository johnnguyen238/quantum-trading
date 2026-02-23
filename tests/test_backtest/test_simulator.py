"""Tests for the backtest order simulator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.backtest.simulator import OrderSimulator
from src.config.constants import OrderSide, OrderStatus, PositionState
from src.data.models import Position
from src.strategy.base import TradeAction


def _make_action(
    action: str = "open_long",
    amount: float = 0.1,
    symbol: str = "BTC/USDT:USDT",
    reason: str = "",
) -> TradeAction:
    return TradeAction(
        action=action, symbol=symbol, amount=amount, price=42000.0, reason=reason
    )


def _make_position(
    side: OrderSide = OrderSide.BUY,
    entry_price: float = 42000.0,
    amount: float = 0.1,
    leverage: int = 5,
) -> Position:
    return Position(
        id=1,
        symbol="BTC/USDT:USDT",
        side=side,
        state=PositionState.OPEN,
        entry_price=entry_price,
        current_price=entry_price,
        amount=amount,
        leverage=leverage,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestSimulateFill:
    @pytest.fixture
    def sim(self) -> OrderSimulator:
        return OrderSimulator(fee_rate=0.001, slippage=0.0001)

    def test_open_long_buy_side(self, sim):
        action = _make_action("open_long")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.side == OrderSide.BUY
        assert trade.status == OrderStatus.FILLED

    def test_open_long_slippage_up(self, sim):
        action = _make_action("open_long")
        trade = sim.simulate_fill(action, 42000.0)
        # Buy slippage: price goes up
        assert trade.filled_price > 42000.0
        assert trade.filled_price == pytest.approx(42000.0 * 1.0001)

    def test_open_short_sell_side(self, sim):
        action = _make_action("open_short")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.side == OrderSide.SELL

    def test_open_short_slippage_down(self, sim):
        action = _make_action("open_short")
        trade = sim.simulate_fill(action, 42000.0)
        # Sell slippage: price goes down
        assert trade.filled_price < 42000.0
        assert trade.filled_price == pytest.approx(42000.0 * 0.9999)

    def test_dca_long_is_buy(self, sim):
        action = _make_action("dca_long")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.side == OrderSide.BUY
        assert trade.is_dca is True

    def test_dca_short_is_sell(self, sim):
        action = _make_action("dca_short")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.side == OrderSide.SELL
        assert trade.is_dca is True

    def test_close_long_is_sell(self, sim):
        action = _make_action("close", reason="Trend reversal: buy → SHORT")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.side == OrderSide.SELL

    def test_close_short_is_buy(self, sim):
        action = _make_action("close", reason="Trend reversal: sell → LONG")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.side == OrderSide.BUY

    def test_fee_calculation(self, sim):
        action = _make_action("open_long", amount=0.1)
        trade = sim.simulate_fill(action, 42000.0)
        # fee = amount * fill_price * fee_rate
        expected_notional = 0.1 * trade.filled_price
        assert trade.fee == pytest.approx(expected_notional * 0.001)

    def test_filled_amount_matches(self, sim):
        action = _make_action("open_long", amount=0.05)
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.filled_amount == 0.05
        assert trade.amount == 0.05

    def test_unknown_action_raises(self, sim):
        action = _make_action("unknown")
        with pytest.raises(ValueError, match="Unknown action type"):
            sim.simulate_fill(action, 42000.0)

    def test_zero_slippage(self):
        sim = OrderSimulator(fee_rate=0.0, slippage=0.0)
        action = _make_action("open_long")
        trade = sim.simulate_fill(action, 42000.0)
        assert trade.filled_price == 42000.0
        assert trade.fee == 0.0


class TestUpdatePosition:
    @pytest.fixture
    def sim(self) -> OrderSimulator:
        return OrderSimulator(fee_rate=0.001, slippage=0.0)

    def test_dca_updates_average(self, sim):
        pos = _make_position(entry_price=42000.0, amount=0.1)
        action = _make_action("dca_long", amount=0.1)
        trade = sim.simulate_fill(action, 41000.0)
        trade.is_dca = True

        sim.update_position(pos, trade)
        # weighted avg: (42000*0.1 + 41000*0.1) / 0.2 = 41500
        assert pos.entry_price == pytest.approx(41500.0)
        assert pos.amount == pytest.approx(0.2)
        assert pos.dca_count == 1

    def test_close_marks_position_closed(self, sim):
        pos = _make_position(entry_price=42000.0, amount=0.1, leverage=5)
        action = _make_action("close")
        trade = sim.simulate_fill(action, 43000.0)
        trade.is_dca = False

        sim.update_position(pos, trade)
        assert pos.state == PositionState.CLOSED
        assert pos.closed_at is not None
        assert pos.unrealized_pnl == 0.0

    def test_close_calculates_pnl(self, sim):
        pos = _make_position(
            side=OrderSide.BUY, entry_price=42000.0, amount=0.1, leverage=5
        )
        action = _make_action("close")
        trade = sim.simulate_fill(action, 43000.0)
        trade.is_dca = False

        sim.update_position(pos, trade)
        # PnL = (43000 - 42000) * 0.1 * 5 - fee
        raw_pnl = (43000.0 - 42000.0) * 0.1 * 5
        assert pos.realized_pnl == pytest.approx(raw_pnl - trade.fee)


class TestCalculatePnl:
    def test_long_profitable(self):
        sim = OrderSimulator()
        pos = _make_position(
            OrderSide.BUY, entry_price=42000.0, amount=0.1, leverage=5
        )
        pnl = sim.calculate_pnl(pos, 43000.0)
        assert pnl == pytest.approx(500.0)  # (43000-42000)*0.1*5

    def test_long_losing(self):
        sim = OrderSimulator()
        pos = _make_position(
            OrderSide.BUY, entry_price=42000.0, amount=0.1, leverage=5
        )
        pnl = sim.calculate_pnl(pos, 41000.0)
        assert pnl == pytest.approx(-500.0)

    def test_short_profitable(self):
        sim = OrderSimulator()
        pos = _make_position(
            OrderSide.SELL, entry_price=42000.0, amount=0.1, leverage=5
        )
        pnl = sim.calculate_pnl(pos, 41000.0)
        assert pnl == pytest.approx(500.0)

    def test_short_losing(self):
        sim = OrderSimulator()
        pos = _make_position(
            OrderSide.SELL, entry_price=42000.0, amount=0.1, leverage=5
        )
        pnl = sim.calculate_pnl(pos, 43000.0)
        assert pnl == pytest.approx(-500.0)

    def test_no_leverage(self):
        sim = OrderSimulator()
        pos = _make_position(
            OrderSide.BUY, entry_price=42000.0, amount=0.1, leverage=1
        )
        pnl = sim.calculate_pnl(pos, 43000.0)
        assert pnl == pytest.approx(100.0)  # (1000)*0.1*1

    def test_breakeven(self):
        sim = OrderSimulator()
        pos = _make_position(
            OrderSide.BUY, entry_price=42000.0, amount=0.1, leverage=5
        )
        pnl = sim.calculate_pnl(pos, 42000.0)
        assert pnl == pytest.approx(0.0)
