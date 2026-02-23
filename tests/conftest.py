"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config.constants import OrderSide, PositionState
from src.config.settings import Settings, load_settings
from src.data.models import OHLCV, Position


@pytest.fixture
def settings() -> Settings:
    """Load default settings for testing."""
    return load_settings("default")


@pytest.fixture
def sample_ohlcv() -> list[OHLCV]:
    """Generate sample OHLCV data."""
    return [
        OHLCV(
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            timestamp=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
            open=42000.0 + i * 10,
            high=42050.0 + i * 10,
            low=41950.0 + i * 10,
            close=42020.0 + i * 10,
            volume=100.0 + i,
        )
        for i in range(100)
    ]


@pytest.fixture
def sample_position() -> Position:
    """A sample open long position."""
    return Position(
        symbol="BTC/USDT:USDT",
        side=OrderSide.BUY,
        state=PositionState.OPEN,
        entry_price=42000.0,
        current_price=42500.0,
        amount=0.1,
        leverage=5,
        unrealized_pnl=50.0,
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Temporary database path for test isolation."""
    return str(tmp_path / "test_trading.db")
