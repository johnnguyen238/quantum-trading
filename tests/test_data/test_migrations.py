"""Tests for database migrations."""

import pytest

from src.data.migrations import run_migrations


class TestMigrations:
    @pytest.mark.asyncio
    async def test_run_migrations(self, db_path):
        await run_migrations(db_path)

        import aiosqlite

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in await cursor.fetchall()]

        assert "ohlcv" in tables
        assert "signals" in tables
        assert "trades" in tables
        assert "positions" in tables
        assert "model_versions" in tables
        assert "performance" in tables
