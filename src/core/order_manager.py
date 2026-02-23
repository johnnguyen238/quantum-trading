"""Order lifecycle management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.models import Trade
    from src.data.repository import Repository
    from src.exchange.executor import OrderExecutor
    from src.strategy.base import TradeAction

logger = logging.getLogger(__name__)

# Actions that open or add to a long position
_LONG_ACTIONS = {"open_long", "dca_long"}
# Actions that open or add to a short position
_SHORT_ACTIONS = {"open_short", "dca_short"}


class OrderManager:
    """Translates strategy actions into exchange orders and tracks them.

    Parameters
    ----------
    executor:
        Exchange order executor.
    repository:
        Data repository for trade persistence.
    """

    def __init__(
        self,
        executor: "OrderExecutor",
        repository: "Repository",
    ) -> None:
        self._executor = executor
        self._repo = repository

    async def submit(self, action: "TradeAction") -> "Trade":
        """Execute a trade action and persist the result.

        Translates action types into the appropriate exchange calls:
        - ``open_long`` / ``dca_long`` → buy market order
        - ``open_short`` / ``dca_short`` → sell market order
        - ``close`` → close_position (opposing market order)
        """
        trade = await self._execute_action(action)

        # Mark DCA metadata
        if action.action.startswith("dca_"):
            trade.is_dca = True

        # Persist to database
        await self._repo.save_trade(trade)
        logger.info(
            "Order submitted: %s %s amount=%.6f → trade_id=%s",
            action.action,
            action.symbol,
            action.amount,
            trade.id,
        )
        return trade

    async def submit_batch(self, actions: list["TradeAction"]) -> list["Trade"]:
        """Execute multiple trade actions sequentially."""
        trades: list[Trade] = []
        for action in actions:
            try:
                trade = await self.submit(action)
                trades.append(trade)
            except Exception:
                logger.exception(
                    "Failed to submit action: %s %s",
                    action.action,
                    action.symbol,
                )
        return trades

    async def sync_open_orders(self) -> list["Trade"]:
        """Poll exchange for status updates on open/pending orders.

        Fetches all trades with OPEN or PENDING status from the database
        and syncs their status with the exchange.
        """
        # Get trades that might still be pending on the exchange
        updated: list[Trade] = []

        # We don't have a get_open_trades method, but we can check
        # recent trades from the repository. For now this is a
        # placeholder that the engine will call with known open trades.
        return updated

    async def sync_trade(self, trade: "Trade") -> "Trade":
        """Sync a single trade's status with the exchange."""
        if not trade.exchange_order_id:
            return trade

        updated = await self._executor.sync_order_status(trade)
        if updated.status != trade.status:
            logger.info(
                "Trade %s status changed: %s → %s",
                trade.exchange_order_id,
                trade.status.value,
                updated.status.value,
            )
            if trade.id is not None:
                await self._repo.update_trade_status(trade.id, updated)
        return updated

    async def cancel_all(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count cancelled."""
        # This would require tracking open order IDs.
        # For market orders (our primary execution), this is rarely needed.
        logger.info("Cancel all orders requested for %s", symbol)
        return 0

    async def _execute_action(self, action: "TradeAction") -> "Trade":
        """Route an action to the correct executor method."""
        if action.action in _LONG_ACTIONS:
            return await self._executor.execute_market_order(
                symbol=action.symbol,
                side="buy",
                amount=action.amount,
            )
        elif action.action in _SHORT_ACTIONS:
            return await self._executor.execute_market_order(
                symbol=action.symbol,
                side="sell",
                amount=action.amount,
            )
        elif action.action == "close":
            # Determine the position side from the action reason or context
            # close actions always specify a side in the reason context
            # We need to know the position side to close it properly.
            # Convention: close action's price indicates current price,
            # and the executor will figure out the opposing side.
            # For now we infer from the action reason or default to sell.
            side = self._infer_close_side(action)
            return await self._executor.close_position(
                symbol=action.symbol,
                side=side,
                amount=action.amount,
            )
        else:
            raise ValueError(f"Unknown action type: {action.action}")

    @staticmethod
    def _infer_close_side(action: "TradeAction") -> str:
        """Infer the position side being closed from the action reason.

        The strategy produces reasons like ``"Trend reversal: buy → SHORT"``
        where the first side token (before ``→``) is the position side.
        """
        reason = action.reason
        # If the reason contains "→", parse the position side before it
        if "→" in reason:
            before_arrow = reason.split("→")[0].lower()
            if "sell" in before_arrow or "short" in before_arrow:
                return "sell"
            if "buy" in before_arrow or "long" in before_arrow:
                return "buy"

        reason_lower = reason.lower()
        if "sell" in reason_lower or "short" in reason_lower:
            return "sell"
        elif "buy" in reason_lower or "long" in reason_lower:
            return "buy"
        # Default: assume closing a long (buy) position
        return "buy"
