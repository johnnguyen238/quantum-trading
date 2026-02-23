"""Tests for the quantum trend strategy (3-scenario logic)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.config.constants import OrderSide, PositionState, Scenario, TrendDirection
from src.data.models import Position
from src.quantum.signal import TrendSignal
from src.strategy.quantum_trend import QuantumTrendStrategy, _signal_matches_position


def _make_signal(
    direction: TrendDirection = TrendDirection.LONG,
    confidence: float = 0.8,
    scenario: Scenario = Scenario.HOLD,
) -> TrendSignal:
    return TrendSignal(
        direction=direction,
        confidence=confidence,
        scenario=scenario,
        timestamp=datetime.now(timezone.utc),
    )


def _make_position(
    side: OrderSide = OrderSide.BUY,
    unrealized_pnl: float = 100.0,
    amount: float = 0.1,
    dca_count: int = 0,
    symbol: str = "BTC/USDT:USDT",
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        state=PositionState.OPEN,
        entry_price=42000.0,
        current_price=42500.0,
        amount=amount,
        leverage=5,
        unrealized_pnl=unrealized_pnl,
        dca_count=dca_count,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _make_df(close: float = 42500.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{"close": close, "high": close + 50, "low": close - 50, "volume": 100.0}]
    )


class TestSignalMatchesPosition:
    def test_long_signal_matches_buy(self):
        signal = _make_signal(TrendDirection.LONG)
        pos = _make_position(OrderSide.BUY)
        assert _signal_matches_position(signal, pos) is True

    def test_short_signal_matches_sell(self):
        signal = _make_signal(TrendDirection.SHORT)
        pos = _make_position(OrderSide.SELL)
        assert _signal_matches_position(signal, pos) is True

    def test_long_signal_does_not_match_sell(self):
        signal = _make_signal(TrendDirection.LONG)
        pos = _make_position(OrderSide.SELL)
        assert _signal_matches_position(signal, pos) is False

    def test_neutral_signal_does_not_match(self):
        signal = _make_signal(TrendDirection.NEUTRAL)
        pos = _make_position(OrderSide.BUY)
        assert _signal_matches_position(signal, pos) is False


class TestClassifyScenario:
    def setup_method(self):
        self.strategy = QuantumTrendStrategy()

    def test_hold_when_signal_matches_and_profitable(self):
        signal = _make_signal(TrendDirection.LONG)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=100.0)
        assert self.strategy._classify_scenario(signal, pos) == Scenario.HOLD

    def test_dca_when_signal_matches_and_losing(self):
        signal = _make_signal(TrendDirection.LONG)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=-50.0)
        assert self.strategy._classify_scenario(signal, pos) == Scenario.DCA

    def test_dca_when_signal_matches_and_breakeven(self):
        signal = _make_signal(TrendDirection.LONG)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=0.0)
        assert self.strategy._classify_scenario(signal, pos) == Scenario.DCA

    def test_reversal_when_signal_flipped(self):
        signal = _make_signal(TrendDirection.SHORT)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=100.0)
        assert self.strategy._classify_scenario(signal, pos) == Scenario.REVERSAL

    def test_reversal_short_to_long(self):
        signal = _make_signal(TrendDirection.LONG)
        pos = _make_position(OrderSide.SELL, unrealized_pnl=50.0)
        assert self.strategy._classify_scenario(signal, pos) == Scenario.REVERSAL

    def test_no_position_returns_hold(self):
        signal = _make_signal(TrendDirection.LONG)
        assert self.strategy._classify_scenario(signal, None) == Scenario.HOLD


class TestEvaluate:
    @pytest.fixture
    def strategy(self) -> QuantumTrendStrategy:
        return QuantumTrendStrategy(
            max_dca_layers=3,
            dca_multiplier=2,
            base_amount=0.01,
            confidence_threshold=0.6,
        )

    @pytest.mark.asyncio
    async def test_no_position_opens_long(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.8)
        result = await strategy.evaluate(_make_df(), [], signal)
        assert len(result.actions) == 1
        assert result.actions[0].action == "open_long"
        assert result.actions[0].amount == 0.01

    @pytest.mark.asyncio
    async def test_no_position_opens_short(self, strategy):
        signal = _make_signal(TrendDirection.SHORT, confidence=0.8)
        result = await strategy.evaluate(_make_df(), [], signal)
        assert len(result.actions) == 1
        assert result.actions[0].action == "open_short"

    @pytest.mark.asyncio
    async def test_neutral_signal_no_action(self, strategy):
        signal = _make_signal(TrendDirection.NEUTRAL)
        result = await strategy.evaluate(_make_df(), [], signal)
        assert len(result.actions) == 0

    @pytest.mark.asyncio
    async def test_low_confidence_no_action(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.3)
        result = await strategy.evaluate(_make_df(), [], signal)
        assert len(result.actions) == 0

    @pytest.mark.asyncio
    async def test_scenario_hold_no_action(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.8)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=100.0)
        result = await strategy.evaluate(_make_df(), [pos], signal)
        assert len(result.actions) == 0
        assert result.signal.scenario == Scenario.HOLD

    @pytest.mark.asyncio
    async def test_scenario_dca_generates_order(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.8)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=-50.0, amount=0.1)
        result = await strategy.evaluate(_make_df(), [pos], signal)
        assert len(result.actions) == 1
        assert result.actions[0].action == "dca_long"
        assert result.actions[0].amount == 0.2  # 0.1 * multiplier(2)
        assert result.signal.scenario == Scenario.DCA

    @pytest.mark.asyncio
    async def test_scenario_dca_short(self, strategy):
        signal = _make_signal(TrendDirection.SHORT, confidence=0.8)
        pos = _make_position(OrderSide.SELL, unrealized_pnl=-30.0, amount=0.05)
        result = await strategy.evaluate(_make_df(), [pos], signal)
        assert len(result.actions) == 1
        assert result.actions[0].action == "dca_short"

    @pytest.mark.asyncio
    async def test_scenario_dca_max_layers_no_action(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.8)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=-50.0, dca_count=3)
        result = await strategy.evaluate(_make_df(), [pos], signal)
        assert len(result.actions) == 0
        assert result.signal.scenario == Scenario.DCA

    @pytest.mark.asyncio
    async def test_scenario_reversal_closes_position(self, strategy):
        signal = _make_signal(TrendDirection.SHORT, confidence=0.8)
        pos = _make_position(OrderSide.BUY, unrealized_pnl=50.0, amount=0.1)
        result = await strategy.evaluate(_make_df(), [pos], signal)
        assert len(result.actions) == 1
        assert result.actions[0].action == "close"
        assert result.actions[0].amount == 0.1
        assert result.signal.scenario == Scenario.REVERSAL

    @pytest.mark.asyncio
    async def test_result_includes_signal(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.8)
        result = await strategy.evaluate(_make_df(), [], signal)
        assert result.signal is signal

    @pytest.mark.asyncio
    async def test_uses_first_matching_position(self, strategy):
        signal = _make_signal(TrendDirection.LONG, confidence=0.8)
        pos1 = _make_position(OrderSide.BUY, unrealized_pnl=100.0)
        pos2 = _make_position(OrderSide.SELL, unrealized_pnl=-10.0, symbol="ETH/USDT:USDT")
        # pos1 is for BTC, pos2 is for ETH — default df has no symbol col
        # so strategy uses BTC/USDT:USDT default, finds pos1 (HOLD)
        result = await strategy.evaluate(_make_df(), [pos1, pos2], signal)
        assert len(result.actions) == 0  # HOLD
