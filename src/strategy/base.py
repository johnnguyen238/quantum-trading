"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.data.models import Position
    from src.quantum.signal import TrendSignal


@dataclass
class TradeAction:
    """A single action the strategy wants the engine to execute."""

    action: str  # "open_long", "open_short", "close", "dca_long", "dca_short"
    symbol: str
    amount: float
    price: float | None = None
    leverage: int = 1
    reason: str = ""


@dataclass
class StrategyResult:
    """Output of a strategy evaluation cycle."""

    actions: list[TradeAction] = field(default_factory=list)
    signal: "TrendSignal | None" = None


class BaseStrategy(ABC):
    """Interface that all trading strategies must implement."""

    @abstractmethod
    async def evaluate(
        self,
        df: "pd.DataFrame",
        positions: list["Position"],
        signal: "TrendSignal",
    ) -> StrategyResult:
        """Evaluate market state and return desired trade actions.

        Parameters
        ----------
        df:
            Recent OHLCV + indicators DataFrame.
        positions:
            Currently open positions for the symbol.
        signal:
            Latest quantum trend signal.

        Returns
        -------
        A ``StrategyResult`` with zero or more ``TradeAction`` items.
        """
        ...
