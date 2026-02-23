"""Load historical OHLCV data for backtesting."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.data.repository import Repository
    from src.exchange.client import BybitClient

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads historical candle data from the database or exchange.

    Parameters
    ----------
    repository:
        Data repository for cached data.
    client:
        Exchange client for fetching missing data.
    """

    def __init__(
        self,
        repository: "Repository",
        client: "BybitClient | None" = None,
    ) -> None:
        self._repo = repository
        self._client = client

    async def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load OHLCV data, fetching from exchange if not cached.

        First attempts to load from the database. If the result is too
        small (< 10 rows), and an exchange client is available, fetches
        from the exchange and caches the data.

        Returns
        -------
        DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

        # Try loading from cache first
        candles = await self._repo.get_ohlcv(symbol, timeframe, start_dt, end_dt)

        if len(candles) < 10 and self._client is not None:
            # Fetch from exchange
            logger.info(
                "Insufficient cached data (%d rows), fetching from exchange...",
                len(candles),
            )
            count = await self.ensure_data(symbol, timeframe, start_date, end_date)
            logger.info("Fetched %d candles from exchange", count)
            candles = await self._repo.get_ohlcv(symbol, timeframe, start_dt, end_dt)

        if not candles:
            logger.warning(
                "No data available for %s %s (%s → %s)",
                symbol,
                timeframe,
                start_date,
                end_date,
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        )

        logger.info(
            "Loaded %d candles for %s %s (%s → %s)",
            len(df),
            symbol,
            timeframe,
            start_date,
            end_date,
        )
        return df

    async def ensure_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """Download and cache any missing candles.

        Uses MarketFeed-style paginated fetching and caches via the
        repository.

        Returns
        -------
        Number of candles fetched and cached.
        """
        if self._client is None:
            logger.warning("No exchange client — cannot fetch data")
            return 0

        from src.exchange.market_feed import MarketFeed

        feed = MarketFeed(self._client, self._repo)
        df = await feed.fetch_historical(symbol, timeframe, start_date, end_date)
        return len(df)
