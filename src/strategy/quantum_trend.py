"""Quantum trend strategy — implements the 3-scenario logic.

Scenario 1 (HOLD): signal matches position AND unrealized PnL > 0
Scenario 2 (DCA):  signal matches position AND unrealized PnL <= 0
Scenario 3 (REVERSAL): signal direction flipped from position
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config.constants import OrderSide, Scenario, TrendDirection
from src.strategy.base import BaseStrategy, StrategyResult, TradeAction
from src.strategy.dca import DCACalculator

if TYPE_CHECKING:
    import pandas as pd

    from src.data.models import Position
    from src.quantum.signal import TrendSignal

logger = logging.getLogger(__name__)

# Map trend direction to position side
_DIRECTION_TO_SIDE: dict[TrendDirection, OrderSide] = {
    TrendDirection.LONG: OrderSide.BUY,
    TrendDirection.SHORT: OrderSide.SELL,
}


def _signal_matches_position(signal: "TrendSignal", position: "Position") -> bool:
    """Check whether the signal direction aligns with the position side."""
    expected_side = _DIRECTION_TO_SIDE.get(signal.direction)
    return expected_side == position.side


class QuantumTrendStrategy(BaseStrategy):
    """Three-scenario quantum trend following strategy.

    Parameters
    ----------
    max_dca_layers:
        Maximum number of DCA additions per position.
    dca_multiplier:
        Size multiplier for each DCA layer (default 2x).
    base_amount:
        Default order size when opening a new position.
    confidence_threshold:
        Minimum confidence to act on a signal.
    stop_loss_pct:
        Maximum percentage loss before automatically closing a position.
        E.g. ``0.05`` = 5% stop-loss.  Set to ``0`` to disable.
    """

    def __init__(
        self,
        max_dca_layers: int = 3,
        dca_multiplier: int = 2,
        base_amount: float = 0.001,
        confidence_threshold: float = 0.6,
        stop_loss_pct: float = 0.05,
    ) -> None:
        self._max_dca_layers = max_dca_layers
        self._dca_multiplier = dca_multiplier
        self._base_amount = base_amount
        self._confidence_threshold = confidence_threshold
        self._stop_loss_pct = stop_loss_pct
        self._dca = DCACalculator(
            multiplier=dca_multiplier, max_layers=max_dca_layers
        )

    async def evaluate(
        self,
        df: "pd.DataFrame",
        positions: list["Position"],
        signal: "TrendSignal",
    ) -> StrategyResult:
        """Classify scenario and generate trade actions."""
        actions: list[TradeAction] = []
        if "symbol" in df.columns:
            symbol = df.iloc[-1].get("symbol", "BTC/USDT:USDT")
        else:
            symbol = "BTC/USDT:USDT"

        # If signal is NEUTRAL or below confidence threshold, do nothing
        if signal.direction == TrendDirection.NEUTRAL:
            logger.debug("NEUTRAL signal — no action")
            return StrategyResult(actions=[], signal=signal)

        if signal.confidence < self._confidence_threshold:
            logger.debug(
                "Confidence %.3f below threshold %.3f — no action",
                signal.confidence,
                self._confidence_threshold,
            )
            return StrategyResult(actions=[], signal=signal)

        current_price = float(df.iloc[-1]["close"])

        # Find position for this symbol (use first matching, if any)
        position = self._find_position(positions, symbol)

        # --- Stop-loss check: close immediately if loss exceeds threshold ---
        if position is not None and self._stop_loss_pct > 0:
            entry_notional = position.entry_price * position.amount
            if entry_notional > 0:
                loss_pct = -position.unrealized_pnl / entry_notional
                if loss_pct >= self._stop_loss_pct:
                    actions.append(
                        TradeAction(
                            action="close",
                            symbol=symbol,
                            amount=position.amount,
                            price=current_price,
                            reason=(
                                f"Stop-loss triggered: {loss_pct:.1%} loss "
                                f"(threshold={self._stop_loss_pct:.1%})"
                            ),
                        )
                    )
                    logger.info(
                        "STOP-LOSS: closing %s position at %.2f (loss=%.1f%%)",
                        position.side.value,
                        current_price,
                        loss_pct * 100,
                    )
                    return StrategyResult(actions=actions, signal=signal)

        if position is None:
            # No open position → open new one in signal direction
            side = _DIRECTION_TO_SIDE[signal.direction]
            action_name = "open_long" if side == OrderSide.BUY else "open_short"
            actions.append(
                TradeAction(
                    action=action_name,
                    symbol=symbol,
                    amount=self._base_amount,
                    price=current_price,
                    reason=f"New {signal.direction.value} signal (conf={signal.confidence:.3f})",
                )
            )
            logger.info("No position — opening %s at %.2f", action_name, current_price)
        else:
            scenario = self._classify_scenario(signal, position)
            signal.scenario = scenario

            if scenario == Scenario.HOLD:
                logger.info(
                    "Scenario 1 (HOLD): %s position profitable, PnL=%.2f",
                    position.side.value,
                    position.unrealized_pnl,
                )
                # No action needed

            elif scenario == Scenario.DCA:
                dca_action = self._dca.calculate_dca_order(position, current_price)
                if dca_action is not None:
                    actions.append(dca_action)
                    logger.info(
                        "Scenario 2 (DCA): layer %d, amount=%.4f at %.2f",
                        position.dca_count + 1,
                        dca_action.amount,
                        current_price,
                    )
                else:
                    logger.info(
                        "Scenario 2 (DCA): max layers (%d) reached — holding",
                        self._max_dca_layers,
                    )

            elif scenario == Scenario.REVERSAL:
                # Close the existing position
                actions.append(
                    TradeAction(
                        action="close",
                        symbol=symbol,
                        amount=position.amount,
                        price=current_price,
                        reason=f"Trend reversal: {position.side.value} → {signal.direction.value}",
                    )
                )
                logger.info(
                    "Scenario 3 (REVERSAL): closing %s position (%.4f) at %.2f",
                    position.side.value,
                    position.amount,
                    current_price,
                )

        return StrategyResult(actions=actions, signal=signal)

    def _classify_scenario(
        self,
        signal: "TrendSignal",
        position: "Position | None",
    ) -> Scenario:
        """Determine which scenario applies given signal and position state."""
        if position is None:
            return Scenario.HOLD

        if _signal_matches_position(signal, position):
            # Signal agrees with position direction
            if position.unrealized_pnl > 0:
                return Scenario.HOLD
            else:
                return Scenario.DCA
        else:
            # Signal flipped
            return Scenario.REVERSAL

    @staticmethod
    def _find_position(
        positions: list["Position"], symbol: str
    ) -> "Position | None":
        """Find the first open position for the given symbol."""
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None
