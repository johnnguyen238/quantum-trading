"""Tests for the repository CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionState,
    Scenario,
    TrendDirection,
)
from src.data.database import Database
from src.data.migrations import run_migrations
from src.data.models import OHLCV, ModelVersion, Performance, Position, Signal, Trade
from src.data.repository import Repository


@pytest.fixture
async def repo(db_path) -> Repository:
    """Create a repository with a migrated database."""
    await run_migrations(db_path)
    db = Database(db_path)
    await db.connect()
    repo = Repository(db)
    yield repo
    await db.disconnect()


class TestOHLCV:
    @pytest.mark.asyncio
    async def test_save_and_get_ohlcv(self, repo):
        candle = OHLCV(
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            open=42000.0,
            high=42100.0,
            low=41900.0,
            close=42050.0,
            volume=150.0,
        )
        row_id = await repo.save_ohlcv(candle)
        assert row_id > 0
        assert candle.id == row_id

        results = await repo.get_ohlcv(
            "BTC/USDT:USDT",
            "15m",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert len(results) == 1
        assert results[0].close == 42050.0

    @pytest.mark.asyncio
    async def test_save_ohlcv_upsert(self, repo):
        candle = OHLCV(
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            open=42000.0,
            high=42100.0,
            low=41900.0,
            close=42050.0,
            volume=150.0,
        )
        await repo.save_ohlcv(candle)

        # Update same candle with new close price
        candle.close = 42100.0
        candle.id = None
        await repo.save_ohlcv(candle)

        results = await repo.get_ohlcv(
            "BTC/USDT:USDT",
            "15m",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert len(results) == 1
        assert results[0].close == 42100.0

    @pytest.mark.asyncio
    async def test_save_ohlcv_batch(self, repo):
        candles = [
            OHLCV(
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                timestamp=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                open=42000.0 + i,
                high=42100.0,
                low=41900.0,
                close=42050.0 + i,
                volume=100.0,
            )
            for i in range(10)
        ]
        count = await repo.save_ohlcv_batch(candles)
        assert count == 10

        results = await repo.get_ohlcv(
            "BTC/USDT:USDT",
            "15m",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_save_ohlcv_batch_empty(self, repo):
        count = await repo.save_ohlcv_batch([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_ohlcv_filters_by_timeframe(self, repo):
        for tf in ["15m", "1h"]:
            await repo.save_ohlcv(
                OHLCV(
                    symbol="BTC/USDT:USDT",
                    timeframe=tf,
                    timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                    open=42000.0,
                    high=42100.0,
                    low=41900.0,
                    close=42050.0,
                    volume=100.0,
                )
            )
        results = await repo.get_ohlcv(
            "BTC/USDT:USDT",
            "15m",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert len(results) == 1


class TestSignals:
    @pytest.mark.asyncio
    async def test_save_and_get_signal(self, repo):
        signal = Signal(
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            direction=TrendDirection.LONG,
            confidence=0.85,
            scenario=Scenario.HOLD,
            model_version="v0.1.0",
            features={"rsi": 0.65, "macd": 0.3},
        )
        row_id = await repo.save_signal(signal)
        assert row_id > 0

        results = await repo.get_signals(
            "BTC/USDT:USDT",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert len(results) == 1
        assert results[0].direction == TrendDirection.LONG
        assert results[0].confidence == 0.85
        assert results[0].features["rsi"] == 0.65


class TestTrades:
    @pytest.mark.asyncio
    async def test_save_and_get_trade(self, repo):
        # First create a position to link to
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
        )
        pos_id = await repo.save_position(pos)

        trade = Trade(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=0.1,
            price=42000.0,
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            status=OrderStatus.FILLED,
            exchange_order_id="exc-123",
            filled_amount=0.1,
            filled_price=42000.0,
            fee=2.52,
            position_id=pos_id,
        )
        row_id = await repo.save_trade(trade)
        assert row_id > 0

        trades = await repo.get_trades_by_position(pos_id)
        assert len(trades) == 1
        assert trades[0].exchange_order_id == "exc-123"
        assert trades[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_update_trade_status(self, repo):
        trade = Trade(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=0.1,
            price=42000.0,
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            status=OrderStatus.OPEN,
            exchange_order_id="exc-456",
        )
        trade_id = await repo.save_trade(trade)

        trade.status = OrderStatus.FILLED
        trade.filled_amount = 0.1
        trade.filled_price = 42000.0
        trade.fee = 2.52
        await repo.update_trade_status(trade_id, trade)

        # Verify via get_trades_by_position (create dummy position first)
        pos = Position(symbol="BTC/USDT:USDT", side=OrderSide.BUY)
        pos_id = await repo.save_position(pos)

        # Manually link the trade to the position for retrieval
        await repo._conn.execute(
            "UPDATE trades SET position_id = ? WHERE id = ?", (pos_id, trade_id)
        )
        await repo._conn.commit()

        trades = await repo.get_trades_by_position(pos_id)
        assert len(trades) == 1
        assert trades[0].status == OrderStatus.FILLED
        assert trades[0].filled_amount == 0.1


class TestPositions:
    @pytest.mark.asyncio
    async def test_save_and_get_open_positions(self, repo):
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
            entry_price=42000.0,
            amount=0.1,
            leverage=5,
            opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        pos_id = await repo.save_position(pos)
        assert pos_id > 0
        assert pos.id == pos_id

        open_positions = await repo.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].entry_price == 42000.0
        assert open_positions[0].leverage == 5

    @pytest.mark.asyncio
    async def test_get_open_positions_filters_by_symbol(self, repo):
        for symbol in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
            await repo.save_position(
                Position(symbol=symbol, side=OrderSide.BUY, state=PositionState.OPEN)
            )

        btc_positions = await repo.get_open_positions("BTC/USDT:USDT")
        assert len(btc_positions) == 1
        assert btc_positions[0].symbol == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_get_open_positions_excludes_closed(self, repo):
        await repo.save_position(
            Position(
                symbol="BTC/USDT:USDT",
                side=OrderSide.BUY,
                state=PositionState.CLOSED,
            )
        )
        open_positions = await repo.get_open_positions()
        assert len(open_positions) == 0

    @pytest.mark.asyncio
    async def test_update_position(self, repo):
        pos = Position(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            state=PositionState.OPEN,
            entry_price=42000.0,
            amount=0.1,
        )
        await repo.save_position(pos)

        pos.current_price = 43000.0
        pos.unrealized_pnl = 100.0
        await repo.update_position(pos)

        open_positions = await repo.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].current_price == 43000.0
        assert open_positions[0].unrealized_pnl == 100.0


class TestModelVersions:
    @pytest.mark.asyncio
    async def test_save_and_get_latest(self, repo):
        v1 = ModelVersion(
            version="v0.1.0",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            parameters={"w0": 0.5, "w1": 0.3},
            accuracy=0.72,
        )
        v2 = ModelVersion(
            version="v0.2.0",
            created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
            parameters={"w0": 0.6, "w1": 0.4},
            accuracy=0.78,
        )
        await repo.save_model_version(v1)
        await repo.save_model_version(v2)

        latest = await repo.get_latest_model_version()
        assert latest is not None
        assert latest.version == "v0.2.0"
        assert latest.accuracy == 0.78
        assert latest.parameters["w0"] == 0.6

    @pytest.mark.asyncio
    async def test_get_latest_returns_none_when_empty(self, repo):
        latest = await repo.get_latest_model_version()
        assert latest is None


class TestPerformance:
    @pytest.mark.asyncio
    async def test_save_and_get_history(self, repo):
        perf = Performance(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            model_version="v0.1.0",
            total_trades=50,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            total_pnl=500.0,
        )
        row_id = await repo.save_performance(perf)
        assert row_id > 0

        history = await repo.get_performance_history()
        assert len(history) == 1
        assert history[0].total_trades == 50
        assert history[0].win_rate == 0.6

    @pytest.mark.asyncio
    async def test_get_history_filters_by_model_version(self, repo):
        for version in ["v0.1.0", "v0.2.0"]:
            await repo.save_performance(
                Performance(
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    model_version=version,
                    total_trades=10,
                )
            )

        history = await repo.get_performance_history("v0.1.0")
        assert len(history) == 1
        assert history[0].model_version == "v0.1.0"
