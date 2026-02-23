"""Main trading loop — the orchestrator."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.config.constants import Scenario
from src.core.order_manager import OrderManager
from src.core.position_manager import PositionManager
from src.core.risk_manager import RiskManager
from src.data.database import Database
from src.data.migrations import run_migrations
from src.data.models import Signal
from src.data.repository import Repository
from src.exchange.client import BybitClient
from src.exchange.executor import OrderExecutor
from src.exchange.market_feed import MarketFeed
from src.quantum.feature_encoding import FeatureEncoder
from src.quantum.trend_detector import TrendDetector
from src.strategy.leverage import LeverageCalculator
from src.strategy.quantum_trend import QuantumTrendStrategy

if TYPE_CHECKING:
    from src.config.settings import Settings

logger = logging.getLogger(__name__)

# Timeframe → seconds for sleep interval between ticks
_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class TradingEngine:
    """Orchestrates the full trading cycle.

    Data flow per tick::

        market_feed → feature_encoding → trend_detector → strategy
        → risk_manager → order_manager → executor → repository

    Parameters
    ----------
    settings:
        Full application settings.
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._running = False
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

        # Components — initialized in _initialize_components
        self._db: Database | None = None
        self._repo: Repository | None = None
        self._client: BybitClient | None = None
        self._market_feed: MarketFeed | None = None
        self._executor: OrderExecutor | None = None
        self._order_mgr: OrderManager | None = None
        self._position_mgr: PositionManager | None = None
        self._risk_mgr: RiskManager | None = None
        self._encoder: FeatureEncoder | None = None
        self._detector: TrendDetector | None = None
        self._strategy: QuantumTrendStrategy | None = None
        self._leverage_calc: LeverageCalculator | None = None

    async def start(self) -> None:
        """Initialize all components and start the trading loop."""
        logger.info("Starting trading engine...")
        await self._initialize_components()
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Trading engine started")

    async def stop(self) -> None:
        """Gracefully stop the trading loop."""
        logger.info("Stopping trading engine...")
        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        await self._shutdown_components()
        logger.info("Trading engine stopped")

    async def _run_loop(self) -> None:
        """Main async loop: fetch data, detect trend, execute strategy."""
        interval = _TF_SECONDS.get(self._settings.trading.timeframe, 900)

        while self._running:
            for symbol in self._settings.trading.symbols:
                try:
                    await self._tick(symbol)
                except Exception:
                    logger.exception("Error in tick for %s", symbol)

            # Sleep until the next candle interval
            await asyncio.sleep(interval)

    async def _tick(self, symbol: str) -> None:
        """Process a single trading cycle for one symbol.

        Steps:
        1. Fetch fresh OHLCV data
        2. Compute technical indicators
        3. Encode features for quantum circuit
        4. Run quantum trend detection
        5. Evaluate strategy (scenario classification)
        6. Validate actions through risk manager
        7. Execute validated actions
        8. Update positions
        9. Persist signal to database
        """
        logger.debug("Tick: %s", symbol)

        # 1. Fetch OHLCV data
        df = await self._market_feed.fetch_ohlcv(
            symbol,
            self._settings.trading.timeframe,
            limit=self._settings.strategy.lookback_period,
        )
        if df.empty or len(df) < 30:
            logger.warning("Insufficient data for %s (%d rows)", symbol, len(df))
            return

        # 2. Compute technical indicators
        df = self._encoder.compute_indicators(df)
        if df.empty:
            logger.warning("No valid rows after indicator computation for %s", symbol)
            return

        # 3. Encode features
        features = self._encoder.encode_single(df)

        # 4. Quantum trend detection
        signal = await self._detector.predict(features)
        logger.info(
            "[%s] Signal: %s (conf=%.3f)",
            symbol,
            signal.direction.value,
            signal.confidence,
        )

        # 5. Sync positions with exchange and evaluate strategy
        await self._position_mgr.sync_with_exchange(symbol)
        positions = await self._position_mgr.get_open_positions(symbol)

        result = await self._strategy.evaluate(df, positions, signal)

        # 6. Get account balance for risk checks
        balance = await self._get_balance()
        all_open = await self._position_mgr.get_open_positions()

        # Calculate leverage for new position actions
        leverage = self._leverage_calc.calculate(df, signal)

        # 7. Validate and execute each action
        for action in result.actions:
            action.leverage = leverage

            validated = await self._risk_mgr.validate(action, balance, all_open)
            if validated is None:
                logger.info("Action rejected by risk manager: %s", action.action)
                continue

            # Set leverage on exchange before placing order
            if validated.action in ("open_long", "open_short"):
                try:
                    await self._client.set_leverage(symbol, validated.leverage)
                except Exception:
                    logger.warning(
                        "Failed to set leverage to %dx on %s",
                        validated.leverage,
                        symbol,
                    )

            # Execute the order
            trade = await self._order_mgr.submit(validated)

            # 8. Update position state
            await self._update_position(
                validated.action, symbol, trade, positions, validated.leverage
            )

        # 9. Persist signal
        await self._persist_signal(signal, symbol)

    async def _update_position(
        self, action: str, symbol: str, trade, positions, leverage: int
    ) -> None:
        """Update position state based on the executed action."""
        if action in ("open_long", "open_short"):
            await self._position_mgr.open_position(trade, leverage)

        elif action in ("dca_long", "dca_short"):
            pos = self._find_position(positions, symbol)
            if pos:
                await self._position_mgr.update_from_trade(pos, trade)

        elif action == "close":
            pos = self._find_position(positions, symbol)
            if pos:
                await self._position_mgr.close_position(pos, trade)

    async def _persist_signal(self, signal, symbol: str) -> None:
        """Save the trend signal to the database."""
        db_signal = Signal(
            symbol=symbol,
            timeframe=self._settings.trading.timeframe,
            timestamp=signal.timestamp,
            direction=signal.direction,
            confidence=signal.confidence,
            scenario=signal.scenario or Scenario.HOLD,
            model_version=signal.model_version,
            features=signal.features,
        )
        await self._repo.save_signal(db_signal)

    async def _get_balance(self) -> float:
        """Fetch current USDT balance from the exchange."""
        try:
            balance = await self._client.fetch_balance()
            usdt = balance.get("USDT", {})
            return float(usdt.get("free", 0.0) or 0.0)
        except Exception:
            logger.exception("Failed to fetch balance")
            return 0.0

    @staticmethod
    def _find_position(positions, symbol: str):
        """Find the first position matching a symbol."""
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def _initialize_components(self) -> None:
        """Set up database, exchange client, strategy, etc."""
        # Database
        db_path = self._settings.database.path
        await run_migrations(db_path)
        self._db = Database(db_path)
        await self._db.connect()
        self._repo = Repository(self._db)

        # Exchange
        self._client = BybitClient(self._settings.exchange)
        await self._client.connect()
        self._executor = OrderExecutor(self._client)
        self._market_feed = MarketFeed(self._client, self._repo)

        # Core managers
        self._order_mgr = OrderManager(self._executor, self._repo)
        self._position_mgr = PositionManager(self._repo, self._client)
        self._risk_mgr = RiskManager(self._settings.trading)
        self._leverage_calc = LeverageCalculator(self._settings.trading)

        # Quantum
        self._encoder = FeatureEncoder(self._settings.strategy)
        self._detector = TrendDetector(
            quantum_settings=self._settings.quantum,
            strategy_settings=self._settings.strategy,
        )
        await self._detector.initialize()

        # Strategy
        self._strategy = QuantumTrendStrategy(
            max_dca_layers=self._settings.trading.max_dca_layers,
            dca_multiplier=self._settings.trading.dca_multiplier,
            base_amount=self._risk_mgr.calculate_position_size(
                balance=10000.0,  # will be recalculated per tick
                leverage=1,
                price=1.0,
            ),
            confidence_threshold=self._settings.quantum.confidence_threshold,
        )

        logger.info("All components initialized")

    async def _shutdown_components(self) -> None:
        """Clean up all resources."""
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

        if self._db is not None:
            await self._db.disconnect()
            self._db = None

        logger.info("All components shut down")
