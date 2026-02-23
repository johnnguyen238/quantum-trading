"""Tests for the backtest data loader."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.backtest.data_loader import DataLoader
from src.data.models import OHLCV


def _make_candles(n: int = 100) -> list[OHLCV]:
    return [
        OHLCV(
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            timestamp=datetime(2024, 1, 1, i % 24, 0, tzinfo=timezone.utc),
            open=42000.0 + i * 10,
            high=42050.0 + i * 10,
            low=41950.0 + i * 10,
            close=42020.0 + i * 10,
            volume=100.0 + i,
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.get_ohlcv = AsyncMock(return_value=_make_candles(100))
    return repo


@pytest.fixture
def mock_client():
    return MagicMock()


class TestLoad:
    @pytest.mark.asyncio
    async def test_loads_from_cache(self, mock_repo):
        loader = DataLoader(mock_repo)
        df = await loader.load("BTC/USDT:USDT", "15m", "2024-01-01", "2024-12-31")
        assert len(df) == 100
        assert list(df.columns) == [
            "timestamp", "open", "high", "low", "close", "volume"
        ]
        mock_repo.get_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_data(self, mock_repo):
        mock_repo.get_ohlcv = AsyncMock(return_value=[])
        loader = DataLoader(mock_repo)
        df = await loader.load("BTC/USDT:USDT", "15m", "2024-01-01", "2024-12-31")
        assert df.empty

    @pytest.mark.asyncio
    async def test_fetches_from_exchange_when_insufficient(
        self, mock_repo, mock_client
    ):
        # First call returns too few, second call (after ensure_data) returns enough
        mock_repo.get_ohlcv = AsyncMock(
            side_effect=[_make_candles(5), _make_candles(100)]
        )

        # Mock the MarketFeed used inside ensure_data()
        with pytest.MonkeyPatch.context() as mp:
            mock_feed_class = MagicMock()
            mock_feed_instance = MagicMock()
            mock_feed_instance.fetch_historical = AsyncMock(
                return_value=pd.DataFrame({"close": range(100)})
            )
            mock_feed_class.return_value = mock_feed_instance

            mp.setattr(
                "src.exchange.market_feed.MarketFeed", mock_feed_class
            )

            loader = DataLoader(mock_repo, mock_client)
            df = await loader.load(
                "BTC/USDT:USDT", "15m", "2024-01-01", "2024-12-31"
            )
            assert len(df) == 100
            # get_ohlcv called twice: initial check + after fetch
            assert mock_repo.get_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_no_fetch_without_client(self, mock_repo):
        mock_repo.get_ohlcv = AsyncMock(return_value=_make_candles(5))
        loader = DataLoader(mock_repo, client=None)
        df = await loader.load("BTC/USDT:USDT", "15m", "2024-01-01", "2024-12-31")
        # Returns what cache has, even if insufficient
        assert len(df) == 5


class TestEnsureData:
    @pytest.mark.asyncio
    async def test_returns_zero_without_client(self, mock_repo):
        loader = DataLoader(mock_repo, client=None)
        count = await loader.ensure_data(
            "BTC/USDT:USDT", "15m", "2024-01-01", "2024-12-31"
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_fetches_via_market_feed(self, mock_repo, mock_client):
        with pytest.MonkeyPatch.context() as mp:
            mock_feed_class = MagicMock()
            mock_feed_instance = MagicMock()
            fake_df = pd.DataFrame({"close": range(200)})
            mock_feed_instance.fetch_historical = AsyncMock(return_value=fake_df)
            mock_feed_class.return_value = mock_feed_instance

            mp.setattr(
                "src.exchange.market_feed.MarketFeed", mock_feed_class
            )

            loader = DataLoader(mock_repo, mock_client)
            count = await loader.ensure_data(
                "BTC/USDT:USDT", "15m", "2024-01-01", "2024-12-31"
            )
            assert count == 200
            mock_feed_instance.fetch_historical.assert_called_once()
