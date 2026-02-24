"""Train the 6-qubit model and run an improved backtest."""

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.backtest.reporter import BacktestReporter
from src.config.settings import load_settings
from src.data.database import Database
from src.data.migrations import run_migrations
from src.data.repository import Repository
from src.exchange.client import BybitClient
from src.ml.feature_engineering import FeatureEngineer
from src.ml.trainer import QuantumTrainer
from src.quantum.trend_detector import TrendDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("optimize")

SYMBOL = "BTC/USDT:USDT"
START = "2025-11-24"
END = "2026-02-24"


async def main():
    settings = load_settings("testnet")

    await run_migrations(settings.database.path)
    db = Database(settings.database.path)
    await db.connect()

    client = BybitClient(settings.exchange)
    await client.connect()

    try:
        repo = Repository(db)
        data_loader = DataLoader(repo, client)

        # --- Load data ---
        logger.info("Loading historical data...")
        df = await data_loader.load(SYMBOL, settings.trading.timeframe, START, END)
        logger.info("Loaded %d candles", len(df))

        # --- Use first 70% for training ---
        split_idx = int(len(df) * 0.7)
        df_train = df.iloc[:split_idx].copy()
        logger.info("Training on first %d candles", len(df_train))

        # --- Create labeled dataset ---
        engineer = FeatureEngineer(
            settings.strategy, forward_period=8, threshold=0.008
        )
        train_features, train_labels = engineer.create_dataset(df_train)

        logger.info(
            "Train set: %d samples — LONG=%d, SHORT=%d, NEUTRAL=%d",
            len(train_labels),
            int((train_labels == 2).sum()),
            int((train_labels == 0).sum()),
            int((train_labels == 1).sum()),
        )

        # --- Train ---
        logger.info("=== Training 6-qubit model: COBYLA, 200 iterations ===")
        q_settings = settings.quantum.model_copy()
        q_settings.max_iterations = 200

        trainer = QuantumTrainer(q_settings, batch_size=600)
        model_version = await trainer.train(train_features, train_labels)

        logger.info(
            "Training done: accuracy=%.2f%%, version=%s",
            model_version.accuracy * 100,
            model_version.version,
        )

        # Save
        save_path = f"data/models/{model_version.version}"
        trainer.save_model(save_path)
        await repo.save_model_version(model_version)
        weights_path = f"{save_path}.npy"

        # --- Backtest with trained model ---
        logger.info("\n=== Running backtest ===")
        logger.info(
            "Config: confidence=%.2f, stop_loss=5%%, max_dca=1, dca_mult=1",
            settings.quantum.confidence_threshold,
        )

        detector = TrendDetector(
            quantum_settings=settings.quantum,
            strategy_settings=settings.strategy,
            model_weights_path=weights_path,
        )
        await detector.initialize()

        engine = BacktestEngine(settings, data_loader, detector)
        report = await engine.run(SYMBOL, START, END)

        reporter = BacktestReporter()
        reporter.print_summary(report)
        reporter.export_csv(report, "data/backtest_optimized.csv")
        logger.info("Results exported to data/backtest_optimized.csv")

    finally:
        await client.disconnect()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
