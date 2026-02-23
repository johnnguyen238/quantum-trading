"""Tests for the risk manager."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config.constants import OrderSide, PositionState
from src.config.settings import TradingSettings
from src.core.risk_manager import RiskManager
from src.data.models import Position
from src.strategy.base import TradeAction


def _make_position(
    symbol: str = "BTC/USDT:USDT",
    side: OrderSide = OrderSide.BUY,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        state=PositionState.OPEN,
        entry_price=42000.0,
        amount=0.1,
        leverage=5,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _make_action(
    action: str = "open_long",
    amount: float = 0.01,
    price: float = 42000.0,
    leverage: int = 5,
) -> TradeAction:
    return TradeAction(
        action=action,
        symbol="BTC/USDT:USDT",
        amount=amount,
        price=price,
        leverage=leverage,
    )


class TestRiskManager:
    @pytest.fixture
    def settings(self) -> TradingSettings:
        return TradingSettings(
            max_leverage=10,
            risk_per_trade=0.02,
            max_open_positions=3,
        )

    @pytest.fixture
    def risk_mgr(self, settings) -> RiskManager:
        return RiskManager(settings)

    @pytest.mark.asyncio
    async def test_validate_passes_valid_action(self, risk_mgr):
        action = _make_action(amount=0.001, leverage=5)
        result = await risk_mgr.validate(action, balance=10000.0, open_positions=[])
        assert result is not None
        assert result.amount == 0.001

    @pytest.mark.asyncio
    async def test_validate_rejects_zero_balance(self, risk_mgr):
        action = _make_action()
        result = await risk_mgr.validate(action, balance=0.0, open_positions=[])
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_rejects_negative_balance(self, risk_mgr):
        action = _make_action()
        result = await risk_mgr.validate(action, balance=-100.0, open_positions=[])
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_rejects_max_positions_reached(self, risk_mgr):
        positions = [_make_position() for _ in range(3)]
        action = _make_action(action="open_long")
        result = await risk_mgr.validate(
            action, balance=10000.0, open_positions=positions
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_allows_dca_even_at_max_positions(self, risk_mgr):
        positions = [_make_position() for _ in range(3)]
        action = _make_action(action="dca_long", amount=0.001)
        result = await risk_mgr.validate(
            action, balance=10000.0, open_positions=positions
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_validate_allows_close_even_at_max_positions(self, risk_mgr):
        positions = [_make_position() for _ in range(3)]
        action = _make_action(action="close", amount=0.1)
        result = await risk_mgr.validate(
            action, balance=10000.0, open_positions=positions
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_validate_clamps_leverage(self, risk_mgr):
        action = _make_action(leverage=20)
        result = await risk_mgr.validate(action, balance=10000.0, open_positions=[])
        assert result is not None
        assert result.leverage == 10  # max is 10

    @pytest.mark.asyncio
    async def test_validate_caps_amount_at_risk_limit(self, risk_mgr):
        # max_amount = (10000 * 0.02 * 5) / 42000 ≈ 0.0238
        action = _make_action(amount=1.0, leverage=5, price=42000.0)
        result = await risk_mgr.validate(action, balance=10000.0, open_positions=[])
        assert result is not None
        expected_max = (10000.0 * 0.02 * 5) / 42000.0
        assert result.amount == pytest.approx(expected_max, rel=1e-6)

    @pytest.mark.asyncio
    async def test_validate_no_cap_when_no_price(self, risk_mgr):
        action = _make_action(amount=0.5, price=None)
        result = await risk_mgr.validate(action, balance=10000.0, open_positions=[])
        assert result is not None
        assert result.amount == 0.5  # no price → no cap check


class TestCheckMaxPositions:
    def test_under_limit(self):
        mgr = RiskManager(TradingSettings(max_open_positions=3))
        assert mgr.check_max_positions([_make_position()]) is True

    def test_at_limit(self):
        mgr = RiskManager(TradingSettings(max_open_positions=2))
        positions = [_make_position() for _ in range(2)]
        assert mgr.check_max_positions(positions) is False

    def test_empty(self):
        mgr = RiskManager(TradingSettings(max_open_positions=1))
        assert mgr.check_max_positions([]) is True


class TestCalculatePositionSize:
    def test_basic_calculation(self):
        mgr = RiskManager(TradingSettings(risk_per_trade=0.02))
        # risk_capital = 10000 * 0.02 = 200
        # notional = 200 * 5 = 1000
        # max_amount = 1000 / 42000 ≈ 0.0238
        size = mgr.calculate_position_size(
            balance=10000.0, leverage=5, price=42000.0
        )
        expected = (10000.0 * 0.02 * 5) / 42000.0
        assert size == pytest.approx(expected)

    def test_zero_price_returns_zero(self):
        mgr = RiskManager(TradingSettings(risk_per_trade=0.02))
        assert mgr.calculate_position_size(10000.0, 5, 0.0) == 0.0

    def test_higher_leverage_larger_size(self):
        mgr = RiskManager(TradingSettings(risk_per_trade=0.02))
        size_5x = mgr.calculate_position_size(10000.0, 5, 42000.0)
        size_10x = mgr.calculate_position_size(10000.0, 10, 42000.0)
        assert size_10x > size_5x


class TestCheckLeverage:
    def test_within_range(self):
        mgr = RiskManager(TradingSettings(max_leverage=10))
        assert mgr.check_leverage(5) == 5

    def test_above_max(self):
        mgr = RiskManager(TradingSettings(max_leverage=10))
        assert mgr.check_leverage(20) == 10

    def test_below_min(self):
        mgr = RiskManager(TradingSettings(max_leverage=10))
        assert mgr.check_leverage(0) == 1

    def test_negative(self):
        mgr = RiskManager(TradingSettings(max_leverage=10))
        assert mgr.check_leverage(-5) == 1
