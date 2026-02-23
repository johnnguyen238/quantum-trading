"""Backtesting engine — replays historical data through the strategy."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.backtest.reporter import BacktestReport, BacktestReporter
from src.backtest.simulator import OrderSimulator
from src.config.constants import OrderSide, PositionState
from src.data.models import Position, Trade
from src.quantum.feature_encoding import FeatureEncoder
from src.strategy.leverage import LeverageCalculator
from src.strategy.quantum_trend import QuantumTrendStrategy

if TYPE_CHECKING:
    import pandas as pd

    from src.backtest.data_loader import DataLoader
    from src.config.settings import Settings
    from src.quantum.trend_detector import TrendDetector

logger = logging.getLogger(__name__)

# Minimum rows of indicator warm-up data before we start trading
_MIN_WARMUP_ROWS = 30


class BacktestEngine:
    """Replay historical OHLCV data and simulate trading.

    Parameters
    ----------
    settings:
        Full application settings.
    data_loader:
        Loader for historical candle data.
    detector:
        Quantum trend detector (must be initialized).
    """

    def __init__(
        self,
        settings: "Settings",
        data_loader: "DataLoader",
        detector: "TrendDetector",
    ) -> None:
        self._settings = settings
        self._data_loader = data_loader
        self._detector = detector

        # Components initialized in run()
        self._encoder: FeatureEncoder | None = None
        self._strategy: QuantumTrendStrategy | None = None
        self._simulator: OrderSimulator | None = None
        self._leverage_calc: LeverageCalculator | None = None
        self._reporter: BacktestReporter | None = None

        # State
        self._positions: list[Position] = []
        self._all_trades: list[Trade] = []
        self._balance: float = 0.0
        self._trade_counter: int = 0

    async def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> BacktestReport:
        """Execute a full backtest over the given date range.

        Steps:
        1. Load historical data
        2. Compute indicators
        3. Walk forward bar-by-bar: encode → predict → strategy → simulate
        4. Close any remaining open positions at end
        5. Compute metrics and return report
        """
        self._initialize_components()
        self._balance = self._settings.backtest.initial_balance

        # 1. Load data
        df = await self._data_loader.load(
            symbol,
            self._settings.trading.timeframe,
            start_date,
            end_date,
        )
        if df.empty or len(df) < _MIN_WARMUP_ROWS:
            logger.warning(
                "Insufficient data for backtest (%d rows)", len(df)
            )
            report = BacktestReport(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self._balance,
                final_balance=self._balance,
            )
            return report

        # 2. Compute indicators
        df = self._encoder.compute_indicators(df)
        if df.empty:
            logger.warning("No valid rows after indicator computation")
            return BacktestReport(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self._balance,
                final_balance=self._balance,
            )

        # 3. Walk forward
        logger.info(
            "Starting backtest: %s %s → %s (%d bars)",
            symbol,
            start_date,
            end_date,
            len(df),
        )

        for i in range(_MIN_WARMUP_ROWS, len(df)):
            window = df.iloc[: i + 1]
            await self._step(window, symbol)

        # 4. Force-close any open positions at last price
        if df is not None and not df.empty:
            last_price = float(df.iloc[-1]["close"])
            self._close_remaining_positions(symbol, last_price)

        # 5. Compute report
        report = self._reporter.compute_metrics(
            self._all_trades,
            self._settings.backtest.initial_balance,
        )
        report.symbol = symbol
        report.start_date = start_date
        report.end_date = end_date

        logger.info(
            "Backtest complete: %d trades, PnL=%.2f, Win rate=%.1f%%",
            report.total_trades,
            report.total_pnl,
            report.win_rate * 100,
        )

        return report

    async def _step(self, window: "pd.DataFrame", symbol: str) -> dict[str, Any]:
        """Process a single bar in the backtest.

        Uses the full window up to the current bar for indicator context,
        but only generates signals from the latest row.
        """
        current_price = float(window.iloc[-1]["close"])

        # Update unrealized PnL for open positions
        for pos in self._positions:
            if pos.state == PositionState.OPEN:
                pos.unrealized_pnl = self._simulator.calculate_pnl(
                    pos, current_price
                )
                pos.current_price = current_price

        # Encode features from the latest row
        try:
            features = self._encoder.encode_single(window)
        except (ValueError, KeyError):
            return {}

        # Quantum prediction
        signal = await self._detector.predict(features)

        # Get open positions for this symbol
        open_positions = [
            p
            for p in self._positions
            if p.symbol == symbol and p.state == PositionState.OPEN
        ]

        # Strategy evaluation
        result = await self._strategy.evaluate(window, open_positions, signal)

        # Calculate leverage
        leverage = self._leverage_calc.calculate(window, signal)

        # Execute actions
        step_trades: list[Trade] = []
        for action in result.actions:
            action.leverage = leverage

            trade = self._simulator.simulate_fill(action, current_price)
            self._trade_counter += 1
            trade.id = self._trade_counter
            step_trades.append(trade)
            self._all_trades.append(trade)

            # Update position state
            self._process_trade(action.action, symbol, trade, leverage)

            # Update balance with fees
            self._balance -= trade.fee

        return {"trades": step_trades, "signal": signal}

    def _process_trade(
        self,
        action: str,
        symbol: str,
        trade: Trade,
        leverage: int,
    ) -> None:
        """Update local position state based on the executed trade."""
        if action in ("open_long", "open_short"):
            pos = Position(
                symbol=symbol,
                side=trade.side,
                state=PositionState.OPEN,
                entry_price=trade.filled_price,
                current_price=trade.filled_price,
                amount=trade.filled_amount,
                leverage=leverage,
                opened_at=trade.timestamp,
            )
            self._trade_counter += 1
            pos.id = self._trade_counter
            trade.position_id = pos.id
            self._positions.append(pos)

        elif action in ("dca_long", "dca_short"):
            pos = self._find_open_position(symbol)
            if pos:
                trade.position_id = pos.id
                self._simulator.update_position(pos, trade)

        elif action == "close":
            pos = self._find_open_position(symbol)
            if pos:
                trade.position_id = pos.id
                self._simulator.update_position(pos, trade)
                self._balance += pos.realized_pnl

    def _close_remaining_positions(
        self, symbol: str, last_price: float
    ) -> None:
        """Force-close any remaining open positions at the last price."""
        for pos in self._positions:
            if pos.state == PositionState.OPEN and pos.symbol == symbol:
                pnl = self._simulator.calculate_pnl(pos, last_price)
                fee = pos.amount * last_price * self._simulator._fee_rate
                pos.state = PositionState.CLOSED
                pos.realized_pnl = pnl - fee
                pos.unrealized_pnl = 0.0
                pos.closed_at = datetime.now(timezone.utc)
                self._balance += pos.realized_pnl

                # Create a synthetic closing trade
                close_side = (
                    OrderSide.SELL
                    if pos.side == OrderSide.BUY
                    else OrderSide.BUY
                )
                from src.config.constants import OrderStatus, OrderType

                close_trade = Trade(
                    symbol=symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    amount=pos.amount,
                    price=last_price,
                    timestamp=datetime.now(timezone.utc),
                    status=OrderStatus.FILLED,
                    exchange_order_id=f"sim-close-{pos.id}",
                    filled_amount=pos.amount,
                    filled_price=last_price,
                    fee=fee,
                    position_id=pos.id,
                )
                self._trade_counter += 1
                close_trade.id = self._trade_counter
                self._all_trades.append(close_trade)

                logger.debug(
                    "Force-closed %s position at %.2f, PnL=%.2f",
                    pos.side.value,
                    last_price,
                    pos.realized_pnl,
                )

    def _find_open_position(self, symbol: str) -> Position | None:
        """Find the first open position for a symbol."""
        for pos in self._positions:
            if pos.symbol == symbol and pos.state == PositionState.OPEN:
                return pos
        return None

    def _initialize_components(self) -> None:
        """Set up strategy, simulator, encoder, etc."""
        self._encoder = FeatureEncoder(self._settings.strategy)
        self._strategy = QuantumTrendStrategy(
            max_dca_layers=self._settings.trading.max_dca_layers,
            dca_multiplier=self._settings.trading.dca_multiplier,
            base_amount=0.01,
            confidence_threshold=self._settings.quantum.confidence_threshold,
        )
        self._simulator = OrderSimulator(
            fee_rate=self._settings.backtest.fee_rate,
            slippage=self._settings.backtest.slippage,
        )
        self._leverage_calc = LeverageCalculator(self._settings.trading)
        self._reporter = BacktestReporter()

        # Reset state
        self._positions = []
        self._all_trades = []
        self._trade_counter = 0
