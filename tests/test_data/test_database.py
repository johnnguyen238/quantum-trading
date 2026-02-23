"""Tests for the async SQLite database connection."""

import pytest

from src.data.database import Database


class TestDatabase:
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, db_path):
        db = Database(db_path)
        await db.connect()
        assert db.connection is not None
        await db.disconnect()

    @pytest.mark.asyncio
    async def test_connection_property_raises_when_not_connected(self, db_path):
        db = Database(db_path)
        with pytest.raises(RuntimeError, match="not connected"):
            _ = db.connection

    @pytest.mark.asyncio
    async def test_context_manager(self, db_path):
        async with Database(db_path) as db:
            assert db.connection is not None
            # Verify WAL mode is enabled
            cursor = await db.connection.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] == "wal"

    @pytest.mark.asyncio
    async def test_foreign_keys_enabled(self, db_path):
        async with Database(db_path) as db:
            cursor = await db.connection.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir" / "test.db")
        async with Database(nested) as db:
            assert db.connection is not None

    @pytest.mark.asyncio
    async def test_disconnect_is_idempotent(self, db_path):
        db = Database(db_path)
        await db.connect()
        await db.disconnect()
        await db.disconnect()  # should not raise
