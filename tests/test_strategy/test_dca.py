"""Tests for the DCA calculator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config.constants import OrderSide, PositionState
from src.data.models import Position
from src.strategy.dca import DCACalculator


def _make_position(
    side: OrderSide = OrderSide.BUY,
    amount: float = 0.1,
    dca_count: int = 0,
) -> Position:
    return Position(
        symbol="BTC/USDT:USDT",
        side=side,
        state=PositionState.OPEN,
        entry_price=42000.0,
        amount=amount,
        dca_count=dca_count,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestDCACalculator:
    def setup_method(self):
        self.calc = DCACalculator(multiplier=2, max_layers=3)

    def test_first_dca_order(self):
        pos = _make_position(amount=0.1, dca_count=0)
        action = self.calc.calculate_dca_order(pos, 41500.0)
        assert action is not None
        assert action.action == "dca_long"
        assert action.amount == 0.2  # 0.1 * 2
        assert action.price == 41500.0
        assert "layer 1/3" in action.reason

    def test_second_dca_order(self):
        pos = _make_position(amount=0.3, dca_count=1)  # after first DCA, total = 0.3
        action = self.calc.calculate_dca_order(pos, 41000.0)
        assert action is not None
        assert action.amount == 0.6  # 0.3 * 2
        assert "layer 2/3" in action.reason

    def test_max_layers_returns_none(self):
        pos = _make_position(dca_count=3)
        action = self.calc.calculate_dca_order(pos, 41000.0)
        assert action is None

    def test_short_position_dca(self):
        pos = _make_position(side=OrderSide.SELL, amount=0.1, dca_count=0)
        action = self.calc.calculate_dca_order(pos, 43000.0)
        assert action is not None
        assert action.action == "dca_short"

    def test_custom_multiplier(self):
        calc = DCACalculator(multiplier=3, max_layers=2)
        pos = _make_position(amount=0.1, dca_count=0)
        action = calc.calculate_dca_order(pos, 41500.0)
        assert action.amount == pytest.approx(0.3)  # 0.1 * 3

    def test_custom_max_layers(self):
        calc = DCACalculator(multiplier=2, max_layers=1)
        pos = _make_position(dca_count=1)
        assert calc.calculate_dca_order(pos, 41000.0) is None

    def test_includes_symbol(self):
        pos = _make_position()
        action = self.calc.calculate_dca_order(pos, 41500.0)
        assert action.symbol == "BTC/USDT:USDT"


class TestCalculateNewAverage:
    def test_basic_average(self):
        avg = DCACalculator.calculate_new_average(
            entry_price=42000.0,
            entry_amount=0.1,
            dca_price=41000.0,
            dca_amount=0.1,
        )
        assert avg == pytest.approx(41500.0)

    def test_weighted_average(self):
        avg = DCACalculator.calculate_new_average(
            entry_price=42000.0,
            entry_amount=0.1,
            dca_price=41000.0,
            dca_amount=0.2,
        )
        # (42000*0.1 + 41000*0.2) / 0.3 = (4200+8200)/0.3 = 12400/0.3
        expected = (42000.0 * 0.1 + 41000.0 * 0.2) / 0.3
        assert avg == pytest.approx(expected)

    def test_same_price_returns_same(self):
        avg = DCACalculator.calculate_new_average(
            entry_price=42000.0,
            entry_amount=0.1,
            dca_price=42000.0,
            dca_amount=0.2,
        )
        assert avg == pytest.approx(42000.0)

    def test_zero_total_amount(self):
        avg = DCACalculator.calculate_new_average(
            entry_price=42000.0,
            entry_amount=0.0,
            dca_price=41000.0,
            dca_amount=0.0,
        )
        assert avg == 0.0
