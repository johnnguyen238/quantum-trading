"""Tests for the trading engine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.constants import OrderSide, PositionState, Scenario, TrendDirection
from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.models import Position
from src.quantum.signal import TrendSignal
from src.strategy.base import StrategyResult, TradeAction


def _make_settings() -> Settings:
    """Create minimal settings for engine tests."""
    return Settings()


def _make_signal(
    direction: TrendDirection = TrendDirection.LONG,
    confidence: float = 0.8,
) -> TrendSignal:
    return TrendSignal(
        direction=direction,
        confidence=confidence,
        scenario=Scenario.HOLD,
        timestamp=datetime.now(timezone.utc),
    )


def _make_df() -> pd.DataFrame:
    """Create a minimal DataFrame for testing."""
    n = 50
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": [42000.0 + i * 10 for i in range(n)],
        "high": [42050.0 + i * 10 for i in range(n)],
        "low": [41950.0 + i * 10 for i in range(n)],
        "close": [42020.0 + i * 10 for i in range(n)],
        "volume": [100.0 + i for i in range(n)],
    }
    return pd.DataFrame(data)


def _make_position() -> Position:
    return Position(
        id=1,
        symbol="BTC/USDT:USDT",
        side=OrderSide.BUY,
        state=PositionState.OPEN,
        entry_price=42000.0,
        current_price=42500.0,
        amount=0.1,
        leverage=5,
        unrealized_pnl=50.0,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestTradingEngineInit:
    def test_creates_with_settings(self):
        settings = _make_settings()
        engine = TradingEngine(settings)
        assert engine._running is False
        assert engine._task is None

    def test_components_initially_none(self):
        engine = TradingEngine(_make_settings())
        assert engine._db is None
        assert engine._client is None
        assert engine._strategy is None


class TestTradingEngineStartStop:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        engine = TradingEngine(_make_settings())

        with patch.object(engine, "_initialize_components", new_callable=AsyncMock):
            with patch.object(engine, "_shutdown_components", new_callable=AsyncMock):
                with patch.object(engine, "_run_loop", new_callable=AsyncMock):
                    await engine.start()
                    assert engine._running is True
                    assert engine._task is not None

                    await engine.stop()
                    assert engine._running is False
                    assert engine._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        engine = TradingEngine(_make_settings())
        with patch.object(engine, "_shutdown_components", new_callable=AsyncMock):
            await engine.stop()  # should not raise
            assert engine._running is False


class TestTick:
    @pytest.fixture
    def engine(self):
        engine = TradingEngine(_make_settings())

        # Mock all components
        engine._market_feed = MagicMock()
        engine._market_feed.fetch_ohlcv = AsyncMock(return_value=_make_df())

        engine._encoder = MagicMock()
        engine._encoder.compute_indicators = MagicMock(return_value=_make_df())
        engine._encoder.encode_single = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0, 4.0])
        )

        engine._detector = MagicMock()
        engine._detector.predict = AsyncMock(return_value=_make_signal())

        engine._position_mgr = MagicMock()
        engine._position_mgr.sync_with_exchange = AsyncMock(return_value=[])
        engine._position_mgr.get_open_positions = AsyncMock(return_value=[])
        engine._position_mgr.open_position = AsyncMock(return_value=_make_position())

        engine._strategy = MagicMock()
        engine._strategy.evaluate = AsyncMock(
            return_value=StrategyResult(
                actions=[
                    TradeAction(
                        action="open_long",
                        symbol="BTC/USDT:USDT",
                        amount=0.01,
                        price=42000.0,
                    )
                ],
                signal=_make_signal(),
            )
        )

        engine._risk_mgr = MagicMock()
        engine._risk_mgr.validate = AsyncMock(
            side_effect=lambda action, balance, open_positions: action
        )

        engine._leverage_calc = MagicMock()
        engine._leverage_calc.calculate = MagicMock(return_value=5)

        engine._client = MagicMock()
        engine._client.fetch_balance = AsyncMock(
            return_value={"USDT": {"free": 10000.0}}
        )
        engine._client.set_leverage = AsyncMock()

        engine._order_mgr = MagicMock()
        from src.data.models import Trade

        mock_trade = Trade(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            order_type=MagicMock(),
            amount=0.01,
            price=42000.0,
            timestamp=datetime.now(timezone.utc),
        )
        engine._order_mgr.submit = AsyncMock(return_value=mock_trade)

        engine._repo = MagicMock()
        engine._repo.save_signal = AsyncMock(return_value=1)

        engine._settings = _make_settings()

        return engine

    @pytest.mark.asyncio
    async def test_tick_full_cycle(self, engine):
        """Test a complete tick: fetch → detect → strategy → execute."""
        await engine._tick("BTC/USDT:USDT")

        engine._market_feed.fetch_ohlcv.assert_called_once()
        engine._encoder.compute_indicators.assert_called_once()
        engine._encoder.encode_single.assert_called_once()
        engine._detector.predict.assert_called_once()
        engine._strategy.evaluate.assert_called_once()
        engine._order_mgr.submit.assert_called_once()
        engine._repo.save_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_skips_on_insufficient_data(self, engine):
        """If fewer than 30 rows, skip the tick."""
        small_df = pd.DataFrame(
            {"close": [42000.0], "high": [42100.0], "low": [41900.0], "volume": [100]}
        )
        engine._market_feed.fetch_ohlcv = AsyncMock(return_value=small_df)
        await engine._tick("BTC/USDT:USDT")
        engine._encoder.compute_indicators.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_skips_on_empty_data(self, engine):
        engine._market_feed.fetch_ohlcv = AsyncMock(
            return_value=pd.DataFrame()
        )
        await engine._tick("BTC/USDT:USDT")
        engine._encoder.compute_indicators.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_no_actions_when_hold(self, engine):
        """When strategy returns HOLD (no actions), no orders submitted."""
        engine._strategy.evaluate = AsyncMock(
            return_value=StrategyResult(actions=[], signal=_make_signal())
        )
        await engine._tick("BTC/USDT:USDT")
        engine._order_mgr.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_risk_rejection_skips_order(self, engine):
        """When risk manager rejects, no order submitted."""
        engine._risk_mgr.validate = AsyncMock(return_value=None)
        await engine._tick("BTC/USDT:USDT")
        engine._order_mgr.submit.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_sets_leverage_on_exchange(self, engine):
        """Leverage is set on exchange for open actions."""
        await engine._tick("BTC/USDT:USDT")
        engine._client.set_leverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_indicator_empty_skips(self, engine):
        """If indicators return empty DataFrame, skip."""
        engine._encoder.compute_indicators = MagicMock(
            return_value=pd.DataFrame()
        )
        await engine._tick("BTC/USDT:USDT")
        engine._detector.predict.assert_not_called()


class TestGetBalance:
    @pytest.mark.asyncio
    async def test_fetches_usdt_balance(self):
        engine = TradingEngine(_make_settings())
        engine._client = MagicMock()
        engine._client.fetch_balance = AsyncMock(
            return_value={"USDT": {"free": 5000.0}}
        )
        balance = await engine._get_balance()
        assert balance == 5000.0

    @pytest.mark.asyncio
    async def test_returns_zero_on_error(self):
        engine = TradingEngine(_make_settings())
        engine._client = MagicMock()
        engine._client.fetch_balance = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        balance = await engine._get_balance()
        assert balance == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_usdt(self):
        engine = TradingEngine(_make_settings())
        engine._client = MagicMock()
        engine._client.fetch_balance = AsyncMock(return_value={})
        balance = await engine._get_balance()
        assert balance == 0.0


class TestFindPosition:
    def test_finds_matching(self):
        pos = _make_position()
        result = TradingEngine._find_position([pos], "BTC/USDT:USDT")
        assert result is pos

    def test_returns_none_when_not_found(self):
        pos = _make_position()
        result = TradingEngine._find_position([pos], "ETH/USDT:USDT")
        assert result is None

    def test_empty_list(self):
        result = TradingEngine._find_position([], "BTC/USDT:USDT")
        assert result is None
