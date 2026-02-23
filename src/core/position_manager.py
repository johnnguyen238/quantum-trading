"""Position tracking and aggregation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.config.constants import OrderSide, PositionState
from src.data.models import Position
from src.strategy.dca import DCACalculator

if TYPE_CHECKING:
    from src.data.models import Trade
    from src.data.repository import Repository
    from src.exchange.client import BybitClient

logger = logging.getLogger(__name__)


class PositionManager:
    """Track and manage open trading positions.

    Aggregates trades into positions and syncs with exchange state.

    Parameters
    ----------
    repository:
        Data repository for position persistence.
    client:
        Exchange client for live position queries.
    """

    def __init__(
        self,
        repository: "Repository",
        client: "BybitClient",
    ) -> None:
        self._repo = repository
        self._client = client

    async def get_open_positions(
        self,
        symbol: str | None = None,
    ) -> list[Position]:
        """Get all open positions, optionally filtered by symbol."""
        return await self._repo.get_open_positions(symbol)

    async def open_position(self, trade: "Trade", leverage: int) -> Position:
        """Create a new position from an initial trade.

        Parameters
        ----------
        trade:
            The opening trade (must be filled or at least submitted).
        leverage:
            Leverage to use for this position.

        Returns
        -------
        The persisted Position with its database ID set.
        """
        price = trade.filled_price if trade.filled_price > 0 else (trade.price or 0.0)

        position = Position(
            symbol=trade.symbol,
            side=trade.side,
            state=PositionState.OPEN,
            entry_price=price,
            current_price=price,
            amount=trade.filled_amount if trade.filled_amount > 0 else trade.amount,
            leverage=leverage,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            dca_count=0,
            opened_at=datetime.now(timezone.utc),
        )

        position_id = await self._repo.save_position(position)
        position.id = position_id  # ensure id is set even with mocks

        # Link the trade to this position
        trade.position_id = position.id
        if trade.id is not None:
            await self._repo.update_trade_status(trade.id, trade)

        logger.info(
            "Opened %s position: %s amount=%.6f @ %.2f leverage=%dx (id=%s)",
            position.side.value,
            position.symbol,
            position.amount,
            position.entry_price,
            position.leverage,
            position.id,
        )
        return position

    async def update_from_trade(
        self,
        position: Position,
        trade: "Trade",
    ) -> Position:
        """Update a position after a DCA trade fill.

        Recalculates the weighted average entry price and increases
        the total amount and DCA count.
        """
        trade_price = (
            trade.filled_price if trade.filled_price > 0 else (trade.price or 0.0)
        )
        trade_amount = (
            trade.filled_amount if trade.filled_amount > 0 else trade.amount
        )

        # Calculate new weighted average entry
        new_entry = DCACalculator.calculate_new_average(
            entry_price=position.entry_price,
            entry_amount=position.amount,
            dca_price=trade_price,
            dca_amount=trade_amount,
        )

        position.entry_price = new_entry
        position.amount += trade_amount
        position.dca_count += 1
        position.state = PositionState.OPEN

        # Link the trade to this position
        trade.position_id = position.id
        trade.is_dca = True
        trade.dca_layer = position.dca_count
        if trade.id is not None:
            await self._repo.update_trade_status(trade.id, trade)

        await self._repo.update_position(position)

        logger.info(
            "DCA layer %d on %s: new avg=%.2f, total amount=%.6f",
            position.dca_count,
            position.symbol,
            position.entry_price,
            position.amount,
        )
        return position

    async def close_position(
        self, position: Position, trade: "Trade"
    ) -> Position:
        """Mark a position as closed after a closing trade.

        Calculates realized PnL from the closing trade price.
        """
        close_price = (
            trade.filled_price if trade.filled_price > 0 else (trade.price or 0.0)
        )

        # Calculate realized PnL
        if position.side == OrderSide.BUY:
            pnl = (close_price - position.entry_price) * position.amount
        else:
            pnl = (position.entry_price - close_price) * position.amount

        # Factor in leverage for PnL
        pnl *= position.leverage

        position.state = PositionState.CLOSED
        position.realized_pnl = pnl
        position.unrealized_pnl = 0.0
        position.closed_at = datetime.now(timezone.utc)
        position.current_price = close_price

        # Link the trade
        trade.position_id = position.id
        if trade.id is not None:
            await self._repo.update_trade_status(trade.id, trade)

        await self._repo.update_position(position)

        logger.info(
            "Closed %s position %s: entry=%.2f, exit=%.2f, PnL=%.2f",
            position.side.value,
            position.symbol,
            position.entry_price,
            close_price,
            pnl,
        )
        return position

    async def sync_with_exchange(self, symbol: str) -> list[Position]:
        """Reconcile local position state with exchange data.

        Fetches live positions from the exchange and updates unrealized
        PnL and current price for local open positions.
        """
        local_positions = await self.get_open_positions(symbol)

        if not local_positions:
            return local_positions

        exchange_positions = await self._client.fetch_positions(symbol)

        # Build a lookup of exchange positions by side
        exchange_by_side: dict[str, dict] = {}
        for ep in exchange_positions:
            contracts = float(ep.get("contracts", 0) or 0)
            if contracts > 0:
                side = ep.get("side", "")
                exchange_by_side[side] = ep

        for pos in local_positions:
            side_key = "long" if pos.side == OrderSide.BUY else "short"
            ep = exchange_by_side.get(side_key)

            if ep:
                pos.current_price = float(ep.get("markPrice", 0) or 0)
                pos.unrealized_pnl = float(
                    ep.get("unrealizedPnl", 0) or 0
                )
                await self._repo.update_position(pos)
                logger.debug(
                    "Synced %s %s: price=%.2f, uPnL=%.2f",
                    pos.symbol,
                    pos.side.value,
                    pos.current_price,
                    pos.unrealized_pnl,
                )
            else:
                # Position exists locally but not on exchange — may be closed
                logger.warning(
                    "Local position %s %s not found on exchange",
                    pos.symbol,
                    pos.side.value,
                )

        return local_positions
