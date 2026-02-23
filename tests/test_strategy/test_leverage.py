"""Tests for the leverage calculator."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.config.constants import Scenario, TrendDirection
from src.config.settings import TradingSettings
from src.quantum.signal import TrendSignal
from src.strategy.leverage import LeverageCalculator


def _make_signal(confidence: float = 0.8) -> TrendSignal:
    return TrendSignal(
        direction=TrendDirection.LONG,
        confidence=confidence,
        scenario=Scenario.HOLD,
        timestamp=datetime.now(timezone.utc),
    )


def _make_df(
    close: float = 42000.0,
    atr: float | None = 420.0,
    high: float = 42200.0,
    low: float = 41800.0,
) -> pd.DataFrame:
    data = {"close": [close], "high": [high], "low": [low]}
    if atr is not None:
        data["atr"] = [atr]
    return pd.DataFrame(data)


class TestLeverageCalculator:
    @pytest.fixture
    def settings(self) -> TradingSettings:
        return TradingSettings(max_leverage=10)

    @pytest.fixture
    def calc(self, settings) -> LeverageCalculator:
        return LeverageCalculator(settings)

    def test_basic_calculation(self, calc):
        # ATR = 420, close = 42000 → atr_pct = 0.01
        # confidence = 0.8 → raw = 0.8 / 0.01 = 80 → capped to 10
        df = _make_df(close=42000.0, atr=420.0)
        signal = _make_signal(confidence=0.8)
        leverage = calc.calculate(df, signal)
        assert leverage == 10  # capped at max

    def test_high_volatility_low_leverage(self, calc):
        # ATR = 4200, close = 42000 → atr_pct = 0.10
        # confidence = 0.8 → raw = 0.8 / 0.10 = 8
        df = _make_df(close=42000.0, atr=4200.0)
        signal = _make_signal(confidence=0.8)
        leverage = calc.calculate(df, signal)
        assert leverage == 8

    def test_low_confidence_reduces_leverage(self, calc):
        # ATR = 4200, close = 42000 → atr_pct = 0.10
        # confidence = 0.4 → raw = 0.4 / 0.10 = 4
        df = _make_df(close=42000.0, atr=4200.0)
        signal = _make_signal(confidence=0.4)
        leverage = calc.calculate(df, signal)
        assert leverage == 4

    def test_very_low_volatility_caps_at_max(self, calc):
        # ATR = 42, close = 42000 → atr_pct = 0.001
        # confidence = 0.8 → raw = 0.8 / 0.001 = 800 → capped at 10
        df = _make_df(close=42000.0, atr=42.0)
        signal = _make_signal(confidence=0.8)
        leverage = calc.calculate(df, signal)
        assert leverage == 10

    def test_minimum_leverage_is_one(self, calc):
        # ATR = 42000, close = 42000 → atr_pct = 1.0
        # confidence = 0.5 → raw = 0.5 / 1.0 = 0.5 → clamped to 1
        df = _make_df(close=42000.0, atr=42000.0)
        signal = _make_signal(confidence=0.5)
        leverage = calc.calculate(df, signal)
        assert leverage == 1

    def test_fallback_to_high_low_range(self, calc):
        # No ATR column → falls back to high - low = 400
        # atr_pct = 400/42000 ≈ 0.0095
        # confidence = 0.8 → raw = 0.8/0.0095 ≈ 84 → capped at 10
        df = _make_df(close=42000.0, atr=None, high=42200.0, low=41800.0)
        signal = _make_signal(confidence=0.8)
        leverage = calc.calculate(df, signal)
        assert leverage == 10

    def test_respects_max_leverage_setting(self):
        settings = TradingSettings(max_leverage=5)
        calc = LeverageCalculator(settings)
        df = _make_df(close=42000.0, atr=42.0)
        signal = _make_signal(confidence=0.9)
        leverage = calc.calculate(df, signal)
        assert leverage == 5

    def test_returns_integer(self, calc):
        df = _make_df(close=42000.0, atr=4200.0)
        signal = _make_signal(confidence=0.75)
        leverage = calc.calculate(df, signal)
        assert isinstance(leverage, int)

    def test_zero_atr_returns_one(self, calc):
        df = _make_df(close=42000.0, atr=0.0)
        signal = _make_signal(confidence=0.8)
        leverage = calc.calculate(df, signal)
        assert leverage == 1

    def test_moderate_volatility_moderate_leverage(self):
        settings = TradingSettings(max_leverage=20)
        calc = LeverageCalculator(settings)
        # ATR = 2000, close = 40000 → atr_pct = 0.05
        # confidence = 0.75 → raw = 0.75 / 0.05 = 15
        df = _make_df(close=40000.0, atr=2000.0)
        signal = _make_signal(confidence=0.75)
        leverage = calc.calculate(df, signal)
        assert leverage == 15
