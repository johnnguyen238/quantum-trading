"""Tests for the market feed (OHLCV fetching and caching)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.data.database import Database
from src.data.migrations import run_migrations
from src.data.repository import Repository
from src.exchange.market_feed import MarketFeed, _raw_to_df, _raw_to_models


@pytest.fixture
async def feed_repo(db_path):
    """Create a repository for market feed tests."""
    await run_migrations(db_path)
    db = Database(db_path)
    await db.connect()
    repo = Repository(db)
    yield repo
    await db.disconnect()


@pytest.fixture
def mock_client():
    """Mock BybitClient."""
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(
        return_value=[
            [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            [1704068100000, 42050.0, 42150.0, 41950.0, 42100.0, 110.0],
            [1704069000000, 42100.0, 42200.0, 42000.0, 42150.0, 120.0],
        ]
    )
    return client


class TestHelpers:
    def test_raw_to_df(self):
        raw = [[1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]]
        df = _raw_to_df(raw)
        assert len(df) == 1
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df.iloc[0]["close"] == 42050.0

    def test_raw_to_models(self):
        raw = [[1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]]
        models = _raw_to_models(raw, "BTC/USDT:USDT", "15m")
        assert len(models) == 1
        assert models[0].symbol == "BTC/USDT:USDT"
        assert models[0].close == 42050.0
        assert models[0].timestamp.tzinfo is not None


class TestMarketFeed:
    @pytest.mark.asyncio
    async def test_fetch_ohlcv(self, mock_client, feed_repo):
        feed = MarketFeed(mock_client, feed_repo)
        df = await feed.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=3)

        assert len(df) == 3
        assert df.iloc[0]["open"] == 42000.0
        mock_client.fetch_ohlcv.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_caches_to_db(self, mock_client, feed_repo):
        feed = MarketFeed(mock_client, feed_repo)
        await feed.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=3)

        # Verify data was persisted
        candles = await feed_repo.get_ohlcv(
            "BTC/USDT:USDT",
            "15m",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert len(candles) == 3

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_empty_response(self, feed_repo):
        client = AsyncMock()
        client.fetch_ohlcv = AsyncMock(return_value=[])
        feed = MarketFeed(client, feed_repo)

        df = await feed.fetch_ohlcv("BTC/USDT:USDT", "15m")
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_get_cached(self, mock_client, feed_repo):
        feed = MarketFeed(mock_client, feed_repo)

        # First populate the cache
        await feed.fetch_ohlcv("BTC/USDT:USDT", "15m", limit=3)

        # Then read from cache
        df = await feed.get_cached("BTC/USDT:USDT", "15m", limit=2)
        assert len(df) == 2

    @pytest.mark.asyncio
    async def test_get_cached_empty(self, mock_client, feed_repo):
        feed = MarketFeed(mock_client, feed_repo)
        df = await feed.get_cached("BTC/USDT:USDT", "15m")
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_fetch_historical_pagination(self, feed_repo):
        """Test that fetch_historical paginates correctly."""
        call_count = 0
        batch1 = [
            [1704067200000 + i * 900000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
            for i in range(200)
        ]
        batch2 = [
            [1704067200000 + (200 + i) * 900000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
            for i in range(50)
        ]

        async def mock_fetch_ohlcv(symbol, timeframe, since=None, limit=200):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return batch1
            return batch2

        client = AsyncMock()
        client.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)

        feed = MarketFeed(client, feed_repo)
        df = await feed.fetch_historical(
            "BTC/USDT:USDT",
            "15m",
            "2024-01-01",
            "2024-03-01",
        )
        assert len(df) == 250
        assert call_count == 2
