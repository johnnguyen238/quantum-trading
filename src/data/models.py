"""Dataclass models representing domain objects persisted to SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionState,
    Scenario,
    TrendDirection,
)


@dataclass
class OHLCV:
    """Single candlestick bar."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    id: int | None = None


@dataclass
class Signal:
    """Output of the quantum trend detector."""

    symbol: str
    timeframe: str
    timestamp: datetime
    direction: TrendDirection
    confidence: float
    scenario: Scenario
    model_version: str
    features: dict[str, float] = field(default_factory=dict)
    id: int | None = None


@dataclass
class Trade:
    """A single order submitted to the exchange."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: float | None
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: str = ""
    filled_amount: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    position_id: int | None = None
    is_dca: bool = False
    dca_layer: int = 0
    id: int | None = None


@dataclass
class Position:
    """An aggregated trading position (may span multiple trades)."""

    symbol: str
    side: OrderSide
    state: PositionState = PositionState.PENDING
    entry_price: float = 0.0
    current_price: float = 0.0
    amount: float = 0.0
    leverage: int = 1
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    dca_count: int = 0
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    id: int | None = None


@dataclass
class ModelVersion:
    """Metadata for a trained quantum model checkpoint."""

    version: str
    created_at: datetime
    parameters: dict[str, float] = field(default_factory=dict)
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    notes: str = ""
    id: int | None = None


@dataclass
class Performance:
    """Snapshot of strategy performance metrics."""

    timestamp: datetime
    model_version: str
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    id: int | None = None
