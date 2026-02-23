"""Order placement and lifecycle management on the exchange."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.config.constants import OrderSide, OrderStatus, OrderType
from src.data.models import Trade

if TYPE_CHECKING:
    from src.exchange.client import BybitClient

logger = logging.getLogger(__name__)


def _map_ccxt_status(status: str) -> OrderStatus:
    """Map ccxt order status string to our OrderStatus enum."""
    mapping = {
        "open": OrderStatus.OPEN,
        "closed": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "cancelled": OrderStatus.CANCELLED,
        "expired": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
    }
    return mapping.get(status, OrderStatus.PENDING)


def _order_to_trade(order: dict, symbol: str, side: str, order_type: str) -> Trade:
    """Convert a ccxt order response dict to a Trade model."""
    return Trade(
        symbol=symbol,
        side=OrderSide(side),
        order_type=OrderType(order_type),
        amount=order.get("amount", 0.0),
        price=order.get("price"),
        timestamp=datetime.now(timezone.utc),
        status=_map_ccxt_status(order.get("status", "open")),
        exchange_order_id=str(order.get("id", "")),
        filled_amount=order.get("filled", 0.0) or 0.0,
        filled_price=order.get("average", 0.0) or 0.0,
        fee=order.get("fee", {}).get("cost", 0.0) if order.get("fee") else 0.0,
    )


class OrderExecutor:
    """Places and tracks orders on Bybit.

    Parameters
    ----------
    client:
        Bybit exchange client.
    """

    def __init__(self, client: "BybitClient") -> None:
        self._client = client

    async def execute_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> Trade:
        """Place a market order and return the resulting Trade."""
        order = await self._client.create_order(
            symbol=symbol,
            side=side,
            order_type="market",
            amount=amount,
        )
        trade = _order_to_trade(order, symbol, side, "market")
        logger.info(
            "Market order executed: %s %s %.4f %s → %s",
            side,
            symbol,
            amount,
            trade.exchange_order_id,
            trade.status.value,
        )
        return trade

    async def execute_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> Trade:
        """Place a limit order and return the resulting Trade."""
        order = await self._client.create_order(
            symbol=symbol,
            side=side,
            order_type="limit",
            amount=amount,
            price=price,
        )
        trade = _order_to_trade(order, symbol, side, "limit")
        logger.info(
            "Limit order placed: %s %s %.4f @ %.2f %s → %s",
            side,
            symbol,
            amount,
            price,
            trade.exchange_order_id,
            trade.status.value,
        )
        return trade

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order. Returns True if successfully cancelled."""
        try:
            await self._client.cancel_order(order_id, symbol)
            logger.info("Order %s cancelled on %s", order_id, symbol)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s on %s", order_id, symbol)
            return False

    async def sync_order_status(self, trade: Trade) -> Trade:
        """Poll the exchange for the current order status and update the Trade."""
        order = await self._client.fetch_order(trade.exchange_order_id, trade.symbol)
        trade.status = _map_ccxt_status(order.get("status", "open"))
        trade.filled_amount = order.get("filled", 0.0) or 0.0
        trade.filled_price = order.get("average", 0.0) or 0.0
        fee = order.get("fee")
        if fee:
            trade.fee = fee.get("cost", 0.0)
        return trade

    async def close_position(self, symbol: str, side: str, amount: float) -> Trade:
        """Close a position by placing an opposing market order.

        Parameters
        ----------
        side:
            The side of the *position* being closed (``"buy"`` or ``"sell"``).
            The closing order will be on the opposite side.
        """
        close_side = "sell" if side == "buy" else "buy"
        return await self.execute_market_order(symbol, close_side, amount)
