"""DCA (Dollar-Cost Averaging) order generation.

When Scenario 2 triggers, generates x2 DCA orders to average down
the position entry price.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config.constants import OrderSide
from src.strategy.base import TradeAction

if TYPE_CHECKING:
    from src.data.models import Position

logger = logging.getLogger(__name__)


class DCACalculator:
    """Calculate DCA order sizes and prices.

    Parameters
    ----------
    multiplier:
        Each DCA layer is ``multiplier`` times the previous size.
    max_layers:
        Maximum number of DCA additions.
    """

    def __init__(self, multiplier: int = 2, max_layers: int = 3) -> None:
        self._multiplier = multiplier
        self._max_layers = max_layers

    def calculate_dca_order(
        self,
        position: "Position",
        current_price: float,
    ) -> TradeAction | None:
        """Generate the next DCA order for a position.

        Returns None if max DCA layers have been reached.

        The DCA amount is: ``position.amount * multiplier``
        (each DCA doubles the *original* position size, so layer 1 = 2x,
        layer 2 = 4x, layer 3 = 8x of the initial amount).

        Parameters
        ----------
        position:
            The current open position to DCA into.
        current_price:
            Current market price.
        """
        if position.dca_count >= self._max_layers:
            return None

        # DCA amount: multiplier^(dca_count+1) * base amount
        # For multiplier=2: layer 0→2x, layer 1→4x, layer 2→8x
        # But we use a simpler model: each DCA = multiplier * current position amount
        # divided by total layers so far, giving a geometric progression.
        # Simplest: dca_amount = position.amount (match current size, effectively doubling)
        dca_amount = position.amount * self._multiplier

        if position.side == OrderSide.BUY:
            action_name = "dca_long"
        else:
            action_name = "dca_short"

        layer = position.dca_count + 1
        logger.debug(
            "DCA layer %d/%d: %s %.4f at %.2f",
            layer,
            self._max_layers,
            action_name,
            dca_amount,
            current_price,
        )

        return TradeAction(
            action=action_name,
            symbol=position.symbol,
            amount=dca_amount,
            price=current_price,
            reason=f"DCA layer {layer}/{self._max_layers} (x{self._multiplier})",
        )

    @staticmethod
    def calculate_new_average(
        entry_price: float,
        entry_amount: float,
        dca_price: float,
        dca_amount: float,
    ) -> float:
        """Calculate the new weighted-average entry price after a DCA fill.

        Formula::

            new_avg = (entry_price * entry_amount + dca_price * dca_amount)
                      / (entry_amount + dca_amount)
        """
        total_cost = entry_price * entry_amount + dca_price * dca_amount
        total_amount = entry_amount + dca_amount
        if total_amount == 0:
            return 0.0
        return total_cost / total_amount
