"""Simulated order fills for backtesting."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.config.constants import OrderSide, OrderStatus, OrderType, PositionState
from src.data.models import Position, Trade
from src.strategy.dca import DCACalculator

if TYPE_CHECKING:
    from src.strategy.base import TradeAction

logger = logging.getLogger(__name__)

# Actions that correspond to buy-side orders
_BUY_ACTIONS = {"open_long", "dca_long"}
_SELL_ACTIONS = {"open_short", "dca_short"}


class OrderSimulator:
    """Simulates order execution against historical data.

    Parameters
    ----------
    fee_rate:
        Taker fee rate (e.g. 0.0006 for 0.06%).
    slippage:
        Simulated slippage fraction (e.g. 0.0001 for 0.01%).
    """

    def __init__(self, fee_rate: float = 0.0006, slippage: float = 0.0001) -> None:
        self._fee_rate = fee_rate
        self._slippage = slippage

    def simulate_fill(
        self,
        action: "TradeAction",
        market_price: float,
    ) -> Trade:
        """Simulate filling a trade action at the given market price.

        Applies slippage: buys fill slightly above market, sells slightly below.
        Fee is computed as ``fee_rate * notional``.
        """
        if action.action in _BUY_ACTIONS:
            side = OrderSide.BUY
            fill_price = market_price * (1 + self._slippage)
        elif action.action in _SELL_ACTIONS:
            side = OrderSide.SELL
            fill_price = market_price * (1 - self._slippage)
        elif action.action == "close":
            # Infer position side from action reason.
            # Reason format: "Trend reversal: {position_side} → {direction}"
            pos_side = self._infer_position_side(action.reason)
            if pos_side == "sell":
                # Closing a short → buy to close
                side = OrderSide.BUY
                fill_price = market_price * (1 + self._slippage)
            else:
                # Closing a long → sell to close
                side = OrderSide.SELL
                fill_price = market_price * (1 - self._slippage)
        else:
            raise ValueError(f"Unknown action type: {action.action}")

        notional = action.amount * fill_price
        fee = notional * self._fee_rate

        trade = Trade(
            symbol=action.symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=action.amount,
            price=fill_price,
            timestamp=datetime.now(timezone.utc),
            status=OrderStatus.FILLED,
            exchange_order_id=f"sim-{id(action)}",
            filled_amount=action.amount,
            filled_price=fill_price,
            fee=fee,
            is_dca=action.action.startswith("dca_"),
        )

        logger.debug(
            "Simulated fill: %s %s %.6f @ %.2f (fee=%.4f)",
            side.value,
            action.symbol,
            action.amount,
            fill_price,
            fee,
        )
        return trade

    def update_position(
        self,
        position: Position,
        trade: Trade,
    ) -> Position:
        """Update a position with a simulated trade fill.

        Handles DCA (weighted average entry) and closing.
        """
        if trade.is_dca:
            new_entry = DCACalculator.calculate_new_average(
                entry_price=position.entry_price,
                entry_amount=position.amount,
                dca_price=trade.filled_price,
                dca_amount=trade.filled_amount,
            )
            position.entry_price = new_entry
            position.amount += trade.filled_amount
            position.dca_count += 1
        else:
            # Close action
            position.state = PositionState.CLOSED
            close_pnl = self.calculate_pnl(position, trade.filled_price)
            position.realized_pnl = close_pnl - trade.fee
            position.unrealized_pnl = 0.0
            position.closed_at = trade.timestamp

        return position

    @staticmethod
    def _infer_position_side(reason: str) -> str:
        """Infer the position side from the close action reason.

        The strategy produces reasons like ``"Trend reversal: buy → SHORT"``
        where the first side token (before ``→``) is the position side.
        """
        if "→" in reason:
            before_arrow = reason.split("→")[0].lower()
            if "sell" in before_arrow or "short" in before_arrow:
                return "sell"
            if "buy" in before_arrow or "long" in before_arrow:
                return "buy"

        reason_lower = reason.lower()
        if "sell" in reason_lower or "short" in reason_lower:
            return "sell"
        return "buy"

    def calculate_pnl(
        self,
        position: Position,
        current_price: float,
    ) -> float:
        """Calculate unrealized PnL for a position at current price.

        For long: ``(current - entry) * amount * leverage``
        For short: ``(entry - current) * amount * leverage``
        """
        if position.side == OrderSide.BUY:
            return (
                (current_price - position.entry_price)
                * position.amount
                * position.leverage
            )
        else:
            return (
                (position.entry_price - current_price)
                * position.amount
                * position.leverage
            )
