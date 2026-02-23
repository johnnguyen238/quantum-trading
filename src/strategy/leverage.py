"""Dynamic leverage calculation based on volatility and signal confidence."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config.settings import TradingSettings
    from src.quantum.signal import TrendSignal

logger = logging.getLogger(__name__)


class LeverageCalculator:
    """Compute appropriate leverage for a trade.

    Uses inverse ATR-based volatility scaled by signal confidence,
    capped at the configured maximum.

    Parameters
    ----------
    settings:
        Trading configuration (max_leverage, etc.).
    """

    def __init__(self, settings: "TradingSettings") -> None:
        self._settings = settings

    def calculate(
        self,
        df: pd.DataFrame,
        signal: "TrendSignal",
        atr_period: int = 14,
    ) -> int:
        """Calculate the recommended leverage.

        Formula::

            atr_pct = ATR / close_price
            volatility_factor = 1 / atr_pct
            leverage = int(volatility_factor * confidence)
            leverage = clamp(leverage, 1, max_leverage)

        Lower volatility + higher confidence → higher leverage.

        Parameters
        ----------
        df:
            OHLCV DataFrame. Must have an ``"atr"`` column or ``high/low/close``
            columns from which ATR can be read.
        signal:
            Current trend signal (uses confidence).
        atr_period:
            ATR period (used for fallback ATR computation).

        Returns
        -------
        Integer leverage value (>= 1).
        """
        close = float(df.iloc[-1]["close"])
        atr = self._get_atr(df, close)

        # Guard against division by zero
        if atr < 1e-10 or close < 1e-10:
            return 1

        atr_pct = atr / close

        # Inverse volatility scaled by confidence
        # Higher confidence → more aggressive; higher volatility → less aggressive
        confidence = max(signal.confidence, 0.01)  # avoid zero
        raw_leverage = confidence / atr_pct

        # Clamp to [1, max_leverage]
        leverage = max(1, min(int(raw_leverage), self._settings.max_leverage))

        logger.debug(
            "Leverage calc: ATR=%.2f, ATR%%=%.4f, conf=%.3f → raw=%.1f → %dx (max=%d)",
            atr,
            atr_pct,
            confidence,
            raw_leverage,
            leverage,
            self._settings.max_leverage,
        )
        return leverage

    @staticmethod
    def _get_atr(df: pd.DataFrame, close: float) -> float:
        """Extract the latest ATR value from the DataFrame.

        Falls back to a simple high-low range if the ``atr`` column is missing.
        """
        if "atr" in df.columns:
            val = df.iloc[-1]["atr"]
            if not np.isnan(val):
                return float(val)

        # Fallback: simple range of last candle
        high = float(df.iloc[-1].get("high", close))
        low = float(df.iloc[-1].get("low", close))
        return max(high - low, 1e-10)
