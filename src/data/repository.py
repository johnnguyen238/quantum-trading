"""CRUD operations for all domain models."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from src.config.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionState,
    Scenario,
    TrendDirection,
)
from src.data.models import OHLCV, ModelVersion, Performance, Position, Signal, Trade

if TYPE_CHECKING:
    from src.data.database import Database

logger = logging.getLogger(__name__)


def _dt_to_str(dt: datetime | None) -> str | None:
    """Convert datetime to ISO string for storage."""
    return dt.isoformat() if dt else None


def _str_to_dt(s: str | None) -> datetime | None:
    """Parse ISO string back to datetime."""
    return datetime.fromisoformat(s) if s else None


class Repository:
    """Data-access layer for the SQLite database."""

    def __init__(self, db: "Database") -> None:
        self._db = db

    @property
    def _conn(self):
        return self._db.connection

    # -- OHLCV ---------------------------------------------------------------

    async def save_ohlcv(self, candle: OHLCV) -> int:
        """Insert or update a single OHLCV candle. Returns the row id."""
        cursor = await self._conn.execute(
            """
            INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, timestamp)
            DO UPDATE SET open=excluded.open, high=excluded.high,
                          low=excluded.low, close=excluded.close, volume=excluded.volume
            """,
            (
                candle.symbol,
                candle.timeframe,
                _dt_to_str(candle.timestamp),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ),
        )
        await self._conn.commit()
        candle.id = cursor.lastrowid
        return cursor.lastrowid

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCV]:
        """Fetch candles for a symbol/timeframe in a time range."""
        cursor = await self._conn.execute(
            """
            SELECT id, symbol, timeframe, timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (symbol, timeframe, _dt_to_str(start), _dt_to_str(end)),
        )
        rows = await cursor.fetchall()
        return [
            OHLCV(
                id=row[0],
                symbol=row[1],
                timeframe=row[2],
                timestamp=_str_to_dt(row[3]),
                open=row[4],
                high=row[5],
                low=row[6],
                close=row[7],
                volume=row[8],
            )
            for row in rows
        ]

    async def save_ohlcv_batch(self, candles: list[OHLCV]) -> int:
        """Bulk-insert candles. Returns number of rows inserted."""
        if not candles:
            return 0
        await self._conn.executemany(
            """
            INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, timestamp)
            DO UPDATE SET open=excluded.open, high=excluded.high,
                          low=excluded.low, close=excluded.close, volume=excluded.volume
            """,
            [
                (
                    c.symbol,
                    c.timeframe,
                    _dt_to_str(c.timestamp),
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                )
                for c in candles
            ],
        )
        await self._conn.commit()
        logger.debug("Saved batch of %d OHLCV candles", len(candles))
        return len(candles)

    # -- Signals --------------------------------------------------------------

    async def save_signal(self, signal: Signal) -> int:
        """Persist a trend signal. Returns the row id."""
        cursor = await self._conn.execute(
            """
            INSERT INTO signals
                (symbol, timeframe, timestamp, direction, confidence, scenario,
                 model_version, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.symbol,
                signal.timeframe,
                _dt_to_str(signal.timestamp),
                signal.direction.value,
                signal.confidence,
                int(signal.scenario),
                signal.model_version,
                json.dumps(signal.features),
            ),
        )
        await self._conn.commit()
        signal.id = cursor.lastrowid
        return cursor.lastrowid

    async def get_signals(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Signal]:
        """Fetch signals in a time range."""
        cursor = await self._conn.execute(
            """
            SELECT id, symbol, timeframe, timestamp, direction, confidence,
                   scenario, model_version, features
            FROM signals
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (symbol, _dt_to_str(start), _dt_to_str(end)),
        )
        rows = await cursor.fetchall()
        return [
            Signal(
                id=row[0],
                symbol=row[1],
                timeframe=row[2],
                timestamp=_str_to_dt(row[3]),
                direction=TrendDirection(row[4]),
                confidence=row[5],
                scenario=Scenario(row[6]),
                model_version=row[7],
                features=json.loads(row[8]) if row[8] else {},
            )
            for row in rows
        ]

    # -- Trades ---------------------------------------------------------------

    async def save_trade(self, trade: Trade) -> int:
        """Persist a trade record. Returns the row id."""
        cursor = await self._conn.execute(
            """
            INSERT INTO trades
                (symbol, side, order_type, amount, price, timestamp, status,
                 exchange_order_id, filled_amount, filled_price, fee,
                 position_id, is_dca, dca_layer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.symbol,
                trade.side.value,
                trade.order_type.value,
                trade.amount,
                trade.price,
                _dt_to_str(trade.timestamp),
                trade.status.value,
                trade.exchange_order_id,
                trade.filled_amount,
                trade.filled_price,
                trade.fee,
                trade.position_id,
                1 if trade.is_dca else 0,
                trade.dca_layer,
            ),
        )
        await self._conn.commit()
        trade.id = cursor.lastrowid
        return cursor.lastrowid

    async def update_trade_status(self, trade_id: int, trade: Trade) -> None:
        """Update trade status and fill information."""
        await self._conn.execute(
            """
            UPDATE trades
            SET status = ?, exchange_order_id = ?, filled_amount = ?,
                filled_price = ?, fee = ?
            WHERE id = ?
            """,
            (
                trade.status.value,
                trade.exchange_order_id,
                trade.filled_amount,
                trade.filled_price,
                trade.fee,
                trade_id,
            ),
        )
        await self._conn.commit()

    async def get_trades_by_position(self, position_id: int) -> list[Trade]:
        """Get all trades associated with a position."""
        cursor = await self._conn.execute(
            """
            SELECT id, symbol, side, order_type, amount, price, timestamp, status,
                   exchange_order_id, filled_amount, filled_price, fee,
                   position_id, is_dca, dca_layer
            FROM trades
            WHERE position_id = ?
            ORDER BY timestamp ASC
            """,
            (position_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(row) for row in rows]

    def _row_to_trade(self, row) -> Trade:
        return Trade(
            id=row[0],
            symbol=row[1],
            side=OrderSide(row[2]),
            order_type=OrderType(row[3]),
            amount=row[4],
            price=row[5],
            timestamp=_str_to_dt(row[6]),
            status=OrderStatus(row[7]),
            exchange_order_id=row[8],
            filled_amount=row[9],
            filled_price=row[10],
            fee=row[11],
            position_id=row[12],
            is_dca=bool(row[13]),
            dca_layer=row[14],
        )

    # -- Positions ------------------------------------------------------------

    async def save_position(self, position: Position) -> int:
        """Create a new position. Returns the row id."""
        cursor = await self._conn.execute(
            """
            INSERT INTO positions
                (symbol, side, state, entry_price, current_price, amount,
                 leverage, unrealized_pnl, realized_pnl, dca_count,
                 opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.symbol,
                position.side.value,
                position.state.value,
                position.entry_price,
                position.current_price,
                position.amount,
                position.leverage,
                position.unrealized_pnl,
                position.realized_pnl,
                position.dca_count,
                _dt_to_str(position.opened_at),
                _dt_to_str(position.closed_at),
            ),
        )
        await self._conn.commit()
        position.id = cursor.lastrowid
        return cursor.lastrowid

    async def update_position(self, position: Position) -> None:
        """Update an existing position."""
        await self._conn.execute(
            """
            UPDATE positions
            SET state = ?, entry_price = ?, current_price = ?, amount = ?,
                leverage = ?, unrealized_pnl = ?, realized_pnl = ?,
                dca_count = ?, opened_at = ?, closed_at = ?
            WHERE id = ?
            """,
            (
                position.state.value,
                position.entry_price,
                position.current_price,
                position.amount,
                position.leverage,
                position.unrealized_pnl,
                position.realized_pnl,
                position.dca_count,
                _dt_to_str(position.opened_at),
                _dt_to_str(position.closed_at),
                position.id,
            ),
        )
        await self._conn.commit()

    async def get_open_positions(self, symbol: str | None = None) -> list[Position]:
        """Get all open positions, optionally filtered by symbol."""
        if symbol:
            cursor = await self._conn.execute(
                """
                SELECT id, symbol, side, state, entry_price, current_price, amount,
                       leverage, unrealized_pnl, realized_pnl, dca_count,
                       opened_at, closed_at
                FROM positions
                WHERE state IN ('OPEN', 'PARTIAL') AND symbol = ?
                ORDER BY opened_at ASC
                """,
                (symbol,),
            )
        else:
            cursor = await self._conn.execute(
                """
                SELECT id, symbol, side, state, entry_price, current_price, amount,
                       leverage, unrealized_pnl, realized_pnl, dca_count,
                       opened_at, closed_at
                FROM positions
                WHERE state IN ('OPEN', 'PARTIAL')
                ORDER BY opened_at ASC
                """
            )
        rows = await cursor.fetchall()
        return [self._row_to_position(row) for row in rows]

    def _row_to_position(self, row) -> Position:
        return Position(
            id=row[0],
            symbol=row[1],
            side=OrderSide(row[2]),
            state=PositionState(row[3]),
            entry_price=row[4],
            current_price=row[5],
            amount=row[6],
            leverage=row[7],
            unrealized_pnl=row[8],
            realized_pnl=row[9],
            dca_count=row[10],
            opened_at=_str_to_dt(row[11]),
            closed_at=_str_to_dt(row[12]),
        )

    # -- Model versions -------------------------------------------------------

    async def save_model_version(self, version: ModelVersion) -> int:
        """Save a model version record. Returns the row id."""
        cursor = await self._conn.execute(
            """
            INSERT INTO model_versions
                (version, created_at, parameters, accuracy, sharpe_ratio, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version.version,
                _dt_to_str(version.created_at),
                json.dumps(version.parameters),
                version.accuracy,
                version.sharpe_ratio,
                version.notes,
            ),
        )
        await self._conn.commit()
        version.id = cursor.lastrowid
        return cursor.lastrowid

    async def get_latest_model_version(self) -> ModelVersion | None:
        """Get the most recent model version."""
        cursor = await self._conn.execute(
            """
            SELECT id, version, created_at, parameters, accuracy, sharpe_ratio, notes
            FROM model_versions
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ModelVersion(
            id=row[0],
            version=row[1],
            created_at=_str_to_dt(row[2]),
            parameters=json.loads(row[3]) if row[3] else {},
            accuracy=row[4],
            sharpe_ratio=row[5],
            notes=row[6],
        )

    # -- Performance ----------------------------------------------------------

    async def save_performance(self, perf: Performance) -> int:
        """Save a performance snapshot. Returns the row id."""
        cursor = await self._conn.execute(
            """
            INSERT INTO performance
                (timestamp, model_version, total_trades, win_rate,
                 profit_factor, sharpe_ratio, max_drawdown, total_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _dt_to_str(perf.timestamp),
                perf.model_version,
                perf.total_trades,
                perf.win_rate,
                perf.profit_factor,
                perf.sharpe_ratio,
                perf.max_drawdown,
                perf.total_pnl,
            ),
        )
        await self._conn.commit()
        perf.id = cursor.lastrowid
        return cursor.lastrowid

    async def get_performance_history(
        self,
        model_version: str | None = None,
    ) -> list[Performance]:
        """Get performance history, optionally filtered by model version."""
        if model_version:
            cursor = await self._conn.execute(
                """
                SELECT id, timestamp, model_version, total_trades, win_rate,
                       profit_factor, sharpe_ratio, max_drawdown, total_pnl
                FROM performance
                WHERE model_version = ?
                ORDER BY timestamp ASC
                """,
                (model_version,),
            )
        else:
            cursor = await self._conn.execute(
                """
                SELECT id, timestamp, model_version, total_trades, win_rate,
                       profit_factor, sharpe_ratio, max_drawdown, total_pnl
                FROM performance
                ORDER BY timestamp ASC
                """
            )
        rows = await cursor.fetchall()
        return [
            Performance(
                id=row[0],
                timestamp=_str_to_dt(row[1]),
                model_version=row[2],
                total_trades=row[3],
                win_rate=row[4],
                profit_factor=row[5],
                sharpe_ratio=row[6],
                max_drawdown=row[7],
                total_pnl=row[8],
            )
            for row in rows
        ]
