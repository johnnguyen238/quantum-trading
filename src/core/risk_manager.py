"""Risk management — leverage limits and exposure validation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings import TradingSettings
    from src.data.models import Position
    from src.strategy.base import TradeAction

logger = logging.getLogger(__name__)


class RiskManager:
    """Validate and constrain trade actions based on risk parameters.

    Parameters
    ----------
    settings:
        Trading configuration (max_leverage, risk_per_trade, etc.).
    """

    def __init__(self, settings: "TradingSettings") -> None:
        self._settings = settings

    async def validate(
        self,
        action: "TradeAction",
        balance: float,
        open_positions: list["Position"],
    ) -> "TradeAction | None":
        """Validate a trade action against risk limits.

        Returns the (possibly adjusted) action, or ``None`` if rejected.

        Checks performed:
        1. Balance must be positive.
        2. New positions must not exceed ``max_open_positions``.
        3. Leverage is clamped to ``max_leverage``.
        4. Amount is capped at max position size for the balance.
        """
        if balance <= 0:
            logger.warning("Rejecting action: zero or negative balance (%.2f)", balance)
            return None

        # For new position opens, check max positions limit
        is_new_position = action.action in ("open_long", "open_short")
        if is_new_position and not self.check_max_positions(open_positions):
            logger.warning(
                "Rejecting %s: max open positions (%d) reached",
                action.action,
                self._settings.max_open_positions,
            )
            return None

        # Clamp leverage
        action.leverage = self.check_leverage(action.leverage)

        # Cap amount at max position size
        if action.price and action.price > 0:
            max_amount = self.calculate_position_size(
                balance, action.leverage, action.price
            )
            if action.amount > max_amount:
                logger.info(
                    "Reducing amount from %.6f to %.6f (risk limit)",
                    action.amount,
                    max_amount,
                )
                action.amount = max_amount

        logger.debug(
            "Risk validated: %s %s amount=%.6f leverage=%dx",
            action.action,
            action.symbol,
            action.amount,
            action.leverage,
        )
        return action

    def check_max_positions(self, open_positions: list["Position"]) -> bool:
        """Return True if a new position can be opened."""
        return len(open_positions) < self._settings.max_open_positions

    def calculate_position_size(
        self,
        balance: float,
        leverage: int,
        price: float,
    ) -> float:
        """Calculate maximum position size based on risk_per_trade.

        Formula::

            risk_capital = balance * risk_per_trade
            notional = risk_capital * leverage
            max_amount = notional / price

        For example, with balance=10000, risk_per_trade=0.02, leverage=5,
        price=42000:  risk_capital=200, notional=1000, amount=0.0238.
        """
        if price <= 0:
            return 0.0
        risk_capital = balance * self._settings.risk_per_trade
        notional = risk_capital * leverage
        return notional / price

    def check_leverage(self, leverage: int) -> int:
        """Clamp leverage to [1, max_leverage]."""
        return max(1, min(leverage, self._settings.max_leverage))
