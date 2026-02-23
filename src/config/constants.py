"""Enums and constants used throughout the trading bot."""

from enum import Enum, IntEnum


class TrendDirection(str, Enum):
    """Market trend direction detected by the quantum circuit."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Scenario(IntEnum):
    """Trading scenario classification.

    Scenario 1: Price follows trend → hold position
    Scenario 2: Trend holds but price hasn't followed → DCA x2
    Scenario 3: Trend reversed → liquidate position
    """

    HOLD = 1
    DCA = 2
    REVERSAL = 3


class PositionState(str, Enum):
    """Lifecycle state of a trading position."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"  # partially filled
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order fill status."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Timeframe(str, Enum):
    """Supported candlestick timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


# Default trading pair format for Bybit futures
DEFAULT_SYMBOL = "BTC/USDT:USDT"

# Feature dimensions for the quantum circuit
N_FEATURES = 4  # returns, RSI, MACD, volume

# SQLite database filename
DB_FILENAME = "trading.db"
