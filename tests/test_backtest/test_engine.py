"""Tests for the backtest engine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.reporter import BacktestReport
from src.config.constants import (
    OrderSide,
    PositionState,
    Scenario,
    TrendDirection,
)
from src.config.settings import Settings
from src.data.models import Position
from src.quantum.signal import TrendSignal
from src.strategy.base import StrategyResult


def _make_settings() -> Settings:
    return Settings()


def _make_df(n: int = 100) -> pd.DataFrame:
    """Create a DataFrame large enough for indicators."""
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": [42000.0 + i * 10 for i in range(n)],
        "high": [42050.0 + i * 10 for i in range(n)],
        "low": [41950.0 + i * 10 for i in range(n)],
        "close": [42020.0 + i * 10 for i in range(n)],
        "volume": [100.0 + i for i in range(n)],
    }
    return pd.DataFrame(data)


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


@pytest.fixture
def mock_data_loader():
    loader = MagicMock()
    loader.load = AsyncMock(return_value=_make_df(100))
    return loader


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.predict = AsyncMock(return_value=_make_signal())
    return detector


class TestBacktestEngineInit:
    def test_creates_with_settings(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        assert engine is not None


class TestBacktestEngineRun:
    @pytest.mark.asyncio
    async def test_run_returns_report(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")
        assert isinstance(report, BacktestReport)
        assert report.symbol == "BTC/USDT:USDT"
        assert report.start_date == "2024-01-01"
        assert report.end_date == "2024-12-31"

    @pytest.mark.asyncio
    async def test_run_with_insufficient_data(
        self, mock_data_loader, mock_detector
    ):
        mock_data_loader.load = AsyncMock(return_value=_make_df(10))
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")
        assert report.total_trades == 0
        assert report.final_balance == 10000.0

    @pytest.mark.asyncio
    async def test_run_with_empty_data(self, mock_data_loader, mock_detector):
        mock_data_loader.load = AsyncMock(return_value=pd.DataFrame())
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")
        assert report.total_trades == 0

    @pytest.mark.asyncio
    async def test_run_calls_detector(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")
        # Detector should be called for each bar after warmup
        assert mock_detector.predict.call_count > 0

    @pytest.mark.asyncio
    async def test_run_neutral_signal_no_trades(
        self, mock_data_loader, mock_detector
    ):
        """Neutral signals should produce no trades."""
        mock_detector.predict = AsyncMock(
            return_value=_make_signal(TrendDirection.NEUTRAL, confidence=0.5)
        )
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")
        assert report.total_trades == 0

    @pytest.mark.asyncio
    async def test_run_balance_preserved_after_report(
        self, mock_data_loader, mock_detector
    ):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")
        assert report.initial_balance == 10000.0


class TestStep:
    @pytest.mark.asyncio
    async def test_step_with_open_position_updates_pnl(
        self, mock_data_loader, mock_detector
    ):
        """Open positions should have unrealized PnL updated each step."""
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        engine._initialize_components()
        engine._balance = 10000.0

        # Add a mock open position
        pos = Position(
            id=1,
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        engine._positions.append(pos)

        # Mock strategy to return no actions
        engine._strategy = MagicMock()
        engine._strategy.evaluate = AsyncMock(
            return_value=StrategyResult(actions=[], signal=_make_signal())
        )

        df = _make_df(50)
        df = engine._encoder.compute_indicators(df)
        window = df.iloc[:40]

        await engine._step(window, "BTC/USDT:USDT")
        # PnL should have been recalculated
        assert pos.unrealized_pnl != 0.0 or pos.current_price != 42000.0


class TestProcessTrade:
    def test_open_long_creates_position(
        self, mock_data_loader, mock_detector
    ):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        engine._initialize_components()
        engine._positions = []

        from src.config.constants import OrderStatus, OrderType
        from src.data.models import Trade

        trade = Trade(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=42000.0,
            timestamp=datetime.now(timezone.utc),
            status=OrderStatus.FILLED,
            filled_amount=0.1,
            filled_price=42000.0,
        )
        trade.id = 1

        engine._process_trade("open_long", "BTC/USDT:USDT", trade, 5)
        assert len(engine._positions) == 1
        assert engine._positions[0].side == OrderSide.BUY
        assert engine._positions[0].leverage == 5

    def test_close_updates_balance(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        engine._initialize_components()
        engine._balance = 10000.0

        # Create an open position
        pos = Position(
            id=1,
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        engine._positions.append(pos)

        from src.config.constants import OrderStatus, OrderType
        from src.data.models import Trade

        trade = Trade(
            symbol="BTC/USDT:USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=43000.0,
            timestamp=datetime.now(timezone.utc),
            status=OrderStatus.FILLED,
            filled_amount=0.1,
            filled_price=43000.0,
            fee=2.58,
        )
        trade.id = 2

        engine._process_trade("close", "BTC/USDT:USDT", trade, 5)
        assert pos.state == PositionState.CLOSED
        assert engine._balance > 10000.0  # should have added PnL


class TestFindOpenPosition:
    def test_finds_matching(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        pos = Position(
            id=1,
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        engine._positions = [pos]
        assert engine._find_open_position("BTC/USDT:USDT") is pos

    def test_returns_none_when_not_found(
        self, mock_data_loader, mock_detector
    ):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        engine._positions = []
        assert engine._find_open_position("BTC/USDT:USDT") is None

    def test_skips_closed_positions(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        pos = Position(
            id=1,
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.CLOSED,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        engine._positions = [pos]
        assert engine._find_open_position("BTC/USDT:USDT") is None


class TestCloseRemainingPositions:
    def test_closes_open_positions(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        engine._initialize_components()
        engine._balance = 10000.0

        pos = Position(
            id=1,
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        engine._positions = [pos]

        engine._close_remaining_positions("BTC/USDT:USDT", 43000.0)
        assert pos.state == PositionState.CLOSED
        assert pos.closed_at is not None
        assert len(engine._all_trades) == 1  # synthetic close trade

    def test_skips_already_closed(self, mock_data_loader, mock_detector):
        engine = BacktestEngine(
            _make_settings(), mock_data_loader, mock_detector
        )
        engine._initialize_components()
        engine._balance = 10000.0

        pos = Position(
            id=1,
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.CLOSED,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        engine._positions = [pos]

        engine._close_remaining_positions("BTC/USDT:USDT", 43000.0)
        assert len(engine._all_trades) == 0
