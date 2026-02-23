"""SQLite schema creation and migrations."""

from __future__ import annotations

import aiosqlite

SCHEMA_VERSION = 1

TABLES: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS ohlcv (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        timeframe   TEXT    NOT NULL,
        timestamp   TEXT    NOT NULL,
        open        REAL    NOT NULL,
        high        REAL    NOT NULL,
        low         REAL    NOT NULL,
        close       REAL    NOT NULL,
        volume      REAL    NOT NULL,
        UNIQUE(symbol, timeframe, timestamp)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol          TEXT    NOT NULL,
        timeframe       TEXT    NOT NULL,
        timestamp       TEXT    NOT NULL,
        direction       TEXT    NOT NULL,
        confidence      REAL    NOT NULL,
        scenario        INTEGER NOT NULL,
        model_version   TEXT    NOT NULL,
        features        TEXT    DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol              TEXT    NOT NULL,
        side                TEXT    NOT NULL,
        order_type          TEXT    NOT NULL,
        amount              REAL    NOT NULL,
        price               REAL,
        timestamp           TEXT    NOT NULL,
        status              TEXT    NOT NULL DEFAULT 'PENDING',
        exchange_order_id   TEXT    DEFAULT '',
        filled_amount       REAL    DEFAULT 0.0,
        filled_price        REAL    DEFAULT 0.0,
        fee                 REAL    DEFAULT 0.0,
        position_id         INTEGER,
        is_dca              INTEGER DEFAULT 0,
        dca_layer           INTEGER DEFAULT 0,
        FOREIGN KEY (position_id) REFERENCES positions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol          TEXT    NOT NULL,
        side            TEXT    NOT NULL,
        state           TEXT    NOT NULL DEFAULT 'PENDING',
        entry_price     REAL    DEFAULT 0.0,
        current_price   REAL    DEFAULT 0.0,
        amount          REAL    DEFAULT 0.0,
        leverage        INTEGER DEFAULT 1,
        unrealized_pnl  REAL    DEFAULT 0.0,
        realized_pnl    REAL    DEFAULT 0.0,
        dca_count       INTEGER DEFAULT 0,
        opened_at       TEXT,
        closed_at       TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_versions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        version         TEXT    NOT NULL UNIQUE,
        created_at      TEXT    NOT NULL,
        parameters      TEXT    DEFAULT '{}',
        accuracy        REAL    DEFAULT 0.0,
        sharpe_ratio    REAL    DEFAULT 0.0,
        notes           TEXT    DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS performance (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        model_version   TEXT    NOT NULL,
        total_trades    INTEGER DEFAULT 0,
        win_rate        REAL    DEFAULT 0.0,
        profit_factor   REAL    DEFAULT 0.0,
        sharpe_ratio    REAL    DEFAULT 0.0,
        max_drawdown    REAL    DEFAULT 0.0,
        total_pnl       REAL    DEFAULT 0.0
    )
    """,
]

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv(symbol, timeframe, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_trades_position ON trades(position_id)",
    "CREATE INDEX IF NOT EXISTS idx_positions_state ON positions(state)",
]


async def run_migrations(db_path: str) -> None:
    """Create all tables and indexes if they don't exist."""
    async with aiosqlite.connect(db_path) as db:
        for ddl in TABLES:
            await db.execute(ddl)
        for idx in INDEXES:
            await db.execute(idx)
        await db.commit()
