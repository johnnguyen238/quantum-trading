"""OHLCV data fetching and caching."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from src.data.models import OHLCV

if TYPE_CHECKING:
    from src.data.repository import Repository
    from src.exchange.client import BybitClient

logger = logging.getLogger(__name__)

# Timeframe → milliseconds lookup
_TF_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _raw_to_df(raw: list[list]) -> pd.DataFrame:
    """Convert ccxt OHLCV response to a DataFrame."""
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _raw_to_models(raw: list[list], symbol: str, timeframe: str) -> list[OHLCV]:
    """Convert ccxt OHLCV response to a list of OHLCV dataclass instances."""
    return [
        OHLCV(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            open=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
        )
        for row in raw
    ]


class MarketFeed:
    """Fetches OHLCV candles from Bybit and caches them in SQLite.

    Parameters
    ----------
    client:
        Bybit exchange client.
    repository:
        Data repository for persistence.
    """

    def __init__(self, client: "BybitClient", repository: "Repository") -> None:
        self._client = client
        self._repo = repository

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch recent OHLCV candles and cache to database.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT:USDT"``).
        timeframe:
            Candle timeframe (e.g. ``"15m"``).
        limit:
            Number of candles to fetch.

        Returns
        -------
        DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not raw:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Cache to database
        models = _raw_to_models(raw, symbol, timeframe)
        await self._repo.save_ohlcv_batch(models)
        logger.debug("Fetched and cached %d candles for %s %s", len(raw), symbol, timeframe)

        return _raw_to_df(raw)

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data over a date range.

        Paginates through the exchange API and caches all results.

        Parameters
        ----------
        since:
            Start date (ISO format, e.g. ``"2024-01-01"``).
        until:
            End date (ISO format, e.g. ``"2024-12-31"``).
        """
        since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        until_dt = datetime.fromisoformat(until).replace(tzinfo=timezone.utc)
        since_ms = int(since_dt.timestamp() * 1000)
        until_ms = int(until_dt.timestamp() * 1000)
        tf_ms = _TF_MS.get(timeframe, 900_000)

        all_raw: list[list] = []
        cursor_ms = since_ms
        batch_size = 200  # Bybit max per request

        while cursor_ms < until_ms:
            raw = await self._client.fetch_ohlcv(
                symbol, timeframe, since=cursor_ms, limit=batch_size
            )
            if not raw:
                break

            # Filter out candles beyond the end date
            raw = [r for r in raw if r[0] <= until_ms]
            all_raw.extend(raw)

            # Advance cursor past the last candle
            last_ts = raw[-1][0]
            cursor_ms = last_ts + tf_ms

            # Stop if we got fewer than requested (end of available data)
            if len(raw) < batch_size:
                break

        if not all_raw:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Cache everything
        models = _raw_to_models(all_raw, symbol, timeframe)
        await self._repo.save_ohlcv_batch(models)
        logger.info(
            "Fetched %d historical candles for %s %s (%s → %s)",
            len(all_raw),
            symbol,
            timeframe,
            since,
            until,
        )

        return _raw_to_df(all_raw)

    async def get_cached(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Retrieve candles from the local database cache.

        Returns the most recent *limit* candles.
        """
        # Use a wide date range and take the last `limit` rows
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        end = datetime(2100, 1, 1, tzinfo=timezone.utc)
        candles = await self._repo.get_ohlcv(symbol, timeframe, start, end)

        # Take the last `limit` candles
        candles = candles[-limit:]

        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        return pd.DataFrame(
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
