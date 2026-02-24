"""CLI interface for the Quantum Trading Bot."""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="quantum-bot",
    help="Quantum Algorithm Trading Bot — Bybit futures trading with VQC trend detection.",
    add_completion=False,
)
console = Console()


def _configure_logging(level: str = "INFO") -> None:
    """Set up root logger with the specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def _run_trading(settings) -> None:
    """Run the trading engine with graceful shutdown on signals."""
    from src.core.engine import TradingEngine

    engine = TradingEngine(settings)
    loop = asyncio.get_running_loop()

    def _shutdown():
        console.print("\n[yellow]Shutdown signal received, stopping...[/]")
        asyncio.ensure_future(engine.stop())

    # Register signal handlers (not available on Windows for all signals)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            pass

    try:
        await engine.start()
        # Keep running until the engine stops itself or a signal is received
        while engine._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]KeyboardInterrupt, stopping...[/]")
    finally:
        if engine._running:
            await engine.stop()


@app.command()
def run(
    profile: str = typer.Option("testnet", help="Config profile (default/testnet/backtest)"),
    symbols: Optional[list[str]] = typer.Option(None, help="Trading pairs to trade"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Start the live trading bot."""
    from src.config.settings import load_settings

    _configure_logging(log_level)
    settings = load_settings(profile)
    if symbols:
        settings.trading.symbols = symbols

    console.print(f"[bold green]Starting trading bot[/] (profile={profile})")
    console.print(f"  Symbols: {settings.trading.symbols}")
    console.print(f"  Testnet: {settings.exchange.testnet}")
    console.print(f"  Timeframe: {settings.trading.timeframe}")

    try:
        asyncio.run(_run_trading(settings))
    except KeyboardInterrupt:
        pass

    console.print("[green]Bot stopped.[/]")


def _find_latest_model() -> str | None:
    """Find the most recently saved model weights file in data/models/."""
    from pathlib import Path

    models_dir = Path("data/models")
    if not models_dir.exists():
        return None
    npy_files = sorted(models_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime)
    if npy_files:
        return str(npy_files[-1])
    return None


async def _run_backtest(settings, symbol: str, start: str, end: str, export: str | None) -> None:
    """Run a backtest asynchronously with full component setup."""
    from src.backtest.data_loader import DataLoader
    from src.backtest.engine import BacktestEngine
    from src.backtest.reporter import BacktestReporter
    from src.data.database import Database
    from src.data.migrations import run_migrations
    from src.data.repository import Repository
    from src.exchange.client import BybitClient
    from src.quantum.trend_detector import TrendDetector

    # Database
    db_path = settings.database.path
    await run_migrations(db_path)
    db = Database(db_path)
    await db.connect()

    try:
        repo = Repository(db)

        # Exchange client (optional, for fetching missing data)
        client: BybitClient | None = None
        if settings.exchange.api_key:
            try:
                client = BybitClient(settings.exchange)
                await client.connect()
            except Exception:
                console.print("[yellow]Could not connect to exchange, using cached data only.[/]")
                client = None

        # Data loader
        data_loader = DataLoader(repo, client)

        # Quantum detector — load latest trained model if available
        model_weights_path = _find_latest_model()
        if model_weights_path:
            console.print(f"[dim]Loading trained model: {model_weights_path}[/]")
        else:
            console.print("[yellow]No trained model found, using random weights.[/]")

        detector = TrendDetector(
            quantum_settings=settings.quantum,
            strategy_settings=settings.strategy,
            model_weights_path=model_weights_path,
        )
        await detector.initialize()

        # Run backtest
        engine = BacktestEngine(settings, data_loader, detector)
        report = await engine.run(symbol, start, end)

        # Print results
        reporter = BacktestReporter()
        reporter.print_summary(report)

        # Export if requested
        if export:
            reporter.export_csv(report, export)
            console.print(f"[green]Trades exported to {export}[/]")

    finally:
        if client is not None:
            await client.disconnect()
        await db.disconnect()


@app.command()
def backtest(
    symbol: str = typer.Option("BTC/USDT:USDT", help="Symbol to backtest"),
    start: str = typer.Option("2024-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)"),
    profile: str = typer.Option("backtest", help="Config profile"),
    export: Optional[str] = typer.Option(None, help="Export trades to CSV path"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Run a backtest on historical data."""
    from src.config.settings import load_settings

    _configure_logging(log_level)
    settings = load_settings(profile)
    console.print(f"[bold blue]Running backtest[/] {symbol} from {start} to {end}")

    asyncio.run(_run_backtest(settings, symbol, start, end, export))


async def _run_train(
    settings, symbol: str, start: str, end: str, output: str | None
) -> None:
    """Train the quantum model asynchronously."""
    from src.backtest.data_loader import DataLoader
    from src.data.database import Database
    from src.data.migrations import run_migrations
    from src.data.repository import Repository
    from src.exchange.client import BybitClient
    from src.ml.feature_engineering import FeatureEngineer
    from src.ml.trainer import QuantumTrainer

    # Database
    db_path = settings.database.path
    await run_migrations(db_path)
    db = Database(db_path)
    await db.connect()

    try:
        repo = Repository(db)

        # Exchange client (optional, for fetching missing data)
        client: BybitClient | None = None
        if settings.exchange.api_key:
            try:
                client = BybitClient(settings.exchange)
                await client.connect()
            except Exception:
                console.print("[yellow]Could not connect to exchange, using cached data only.[/]")
                client = None

        # Load historical data
        data_loader = DataLoader(repo, client)
        console.print("[dim]Loading historical data...[/]")
        df = await data_loader.load(symbol, settings.trading.timeframe, start, end)

        if df.empty or len(df) < 50:
            console.print(
                f"[red]Insufficient data for training ({len(df)} rows, need >= 50).[/]"
            )
            return

        console.print(f"[dim]Loaded {len(df)} candles.[/]")

        # Create labeled dataset
        engineer = FeatureEngineer(
            settings.strategy,
            forward_period=5,
            threshold=0.01,
        )
        features, labels = engineer.create_dataset(df)

        if len(features) < 20:
            console.print(
                f"[red]Insufficient labeled samples ({len(features)}, need >= 20).[/]"
            )
            return

        console.print(
            f"[dim]Created dataset: {len(features)} samples, "
            f"LONG={int((labels == 2).sum())}, "
            f"SHORT={int((labels == 0).sum())}, "
            f"NEUTRAL={int((labels == 1).sum())}[/]"
        )

        # Train
        console.print("[bold magenta]Training quantum model...[/]")
        trainer = QuantumTrainer(settings.quantum)
        model_version = await trainer.train(features, labels)

        console.print(
            f"[green]Training complete![/] "
            f"Accuracy: {model_version.accuracy:.2%}, "
            f"Version: {model_version.version}"
        )

        # Save model
        save_path = output or f"data/models/{model_version.version}"
        trainer.save_model(save_path)
        console.print(f"[green]Model saved to {save_path}.npy[/]")

        # Persist version to database
        await repo.save_model_version(model_version)
        console.print("[green]Model version recorded in database.[/]")

    finally:
        if client is not None:
            await client.disconnect()
        await db.disconnect()


@app.command()
def train(
    symbol: str = typer.Option("BTC/USDT:USDT", help="Symbol for training data"),
    start: str = typer.Option("2024-01-01", help="Training data start date"),
    end: str = typer.Option("2024-12-31", help="Training data end date"),
    profile: str = typer.Option("default", help="Config profile"),
    output: Optional[str] = typer.Option(None, help="Path to save model weights"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Train / retrain the quantum model on historical data."""
    from src.config.settings import load_settings

    _configure_logging(log_level)
    settings = load_settings(profile)
    console.print(
        f"[bold magenta]Training quantum model[/] on {symbol} ({start} → {end})"
    )
    console.print(f"  Qubits: {settings.quantum.n_qubits}, Reps: {settings.quantum.reps}")
    console.print(f"  Optimizer: {settings.quantum.optimizer}")

    asyncio.run(_run_train(settings, symbol, start, end, output))


async def _run_evaluate(settings, symbol: str, start: str, end: str) -> None:
    """Run walk-forward evaluation asynchronously."""
    from src.backtest.data_loader import DataLoader
    from src.data.database import Database
    from src.data.migrations import run_migrations
    from src.data.repository import Repository
    from src.exchange.client import BybitClient
    from src.ml.evaluator import WalkForwardEvaluator
    from src.ml.feature_engineering import FeatureEngineer

    # Database
    db_path = settings.database.path
    await run_migrations(db_path)
    db = Database(db_path)
    await db.connect()

    try:
        repo = Repository(db)

        # Exchange client (optional)
        client: BybitClient | None = None
        if settings.exchange.api_key:
            try:
                client = BybitClient(settings.exchange)
                await client.connect()
            except Exception:
                console.print("[yellow]Could not connect to exchange, using cached data only.[/]")
                client = None

        # Load data
        data_loader = DataLoader(repo, client)
        console.print("[dim]Loading historical data...[/]")
        df = await data_loader.load(symbol, settings.trading.timeframe, start, end)

        if df.empty or len(df) < 50:
            console.print(
                f"[red]Insufficient data for evaluation ({len(df)} rows, need >= 50).[/]"
            )
            return

        # Create labeled dataset
        engineer = FeatureEngineer(settings.strategy, forward_period=5, threshold=0.01)
        features, labels = engineer.create_dataset(df)

        if len(features) < 20:
            console.print(f"[red]Insufficient samples ({len(features)}).[/]")
            return

        console.print(f"[dim]Dataset: {len(features)} samples[/]")

        # Evaluate
        console.print("[bold cyan]Running walk-forward evaluation...[/]")
        evaluator = WalkForwardEvaluator(
            settings.quantum, n_folds=5, train_ratio=0.8
        )
        results = await evaluator.evaluate(features, labels)

        if not results:
            console.print("[yellow]No evaluation results (data may be too small).[/]")
            return

        # Print per-fold results
        table = Table(title="Walk-Forward Evaluation Results", show_header=True)
        table.add_column("Fold", style="cyan", justify="center")
        table.add_column("Accuracy", style="green", justify="right")
        table.add_column("Precision", style="green", justify="right")
        table.add_column("Recall", style="green", justify="right")
        table.add_column("F1 Score", style="green", justify="right")

        for r in results:
            table.add_row(
                str(r.fold_index + 1),
                f"{r.accuracy:.2%}",
                f"{r.precision:.2%}",
                f"{r.recall:.2%}",
                f"{r.f1_score:.2%}",
            )

        # Aggregate
        agg = evaluator.aggregate(results)
        table.add_row(
            "[bold]Mean[/]",
            f"[bold]{agg['mean_accuracy']:.2%}[/]",
            f"[bold]{agg['mean_precision']:.2%}[/]",
            f"[bold]{agg['mean_recall']:.2%}[/]",
            f"[bold]{agg['mean_f1_score']:.2%}[/]",
        )

        console.print(table)

    finally:
        if client is not None:
            await client.disconnect()
        await db.disconnect()


@app.command()
def evaluate(
    symbol: str = typer.Option("BTC/USDT:USDT", help="Symbol for evaluation data"),
    start: str = typer.Option("2024-01-01", help="Data start date"),
    end: str = typer.Option("2024-12-31", help="Data end date"),
    profile: str = typer.Option("default", help="Config profile"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Evaluate current model with walk-forward validation."""
    from src.config.settings import load_settings

    _configure_logging(log_level)
    settings = load_settings(profile)
    console.print(f"[bold cyan]Evaluating model[/] on {symbol} ({start} → {end})")

    asyncio.run(_run_evaluate(settings, symbol, start, end))


async def _run_status(settings) -> None:
    """Fetch and display bot status asynchronously."""
    from src.data.database import Database
    from src.data.migrations import run_migrations
    from src.data.repository import Repository
    from src.exchange.client import BybitClient

    # Database
    db_path = settings.database.path
    await run_migrations(db_path)
    db = Database(db_path)
    await db.connect()

    try:
        repo = Repository(db)

        # Latest model version
        model_ver = await repo.get_latest_model_version()

        # Exchange status
        balance_str = "N/A"
        position_count = "N/A"
        client: BybitClient | None = None

        if settings.exchange.api_key:
            try:
                client = BybitClient(settings.exchange)
                await client.connect()

                # Balance
                balance = await client.fetch_balance()
                usdt = balance.get("USDT", {})
                free = float(usdt.get("free", 0.0) or 0.0)
                total = float(usdt.get("total", 0.0) or 0.0)
                balance_str = f"${free:,.2f} free / ${total:,.2f} total"

                # Positions
                positions = await client.fetch_positions()
                open_positions = [
                    p for p in positions
                    if float(p.get("contracts", 0) or 0) > 0
                ]
                position_count = str(len(open_positions))

            except Exception as e:
                balance_str = f"Error: {e}"
                position_count = "Error"
        else:
            balance_str = "No API key configured"
            position_count = "No API key configured"

        # Build table
        table = Table(title="Bot Status", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row(
            "Exchange",
            f"Bybit {'testnet' if settings.exchange.testnet else 'mainnet'}",
        )
        table.add_row("Symbols", ", ".join(settings.trading.symbols))
        table.add_row("Timeframe", settings.trading.timeframe)
        table.add_row("Balance (USDT)", balance_str)
        table.add_row("Open Positions", position_count)
        table.add_row(
            "Model Version",
            model_ver.version if model_ver else "No model trained yet",
        )
        table.add_row(
            "Model Accuracy",
            f"{model_ver.accuracy:.2%}" if model_ver else "N/A",
        )
        table.add_row("Max Leverage", str(settings.trading.max_leverage))
        table.add_row("Risk Per Trade", f"{settings.trading.risk_per_trade:.1%}")

        console.print(table)

    finally:
        if client is not None:
            await client.disconnect()
        await db.disconnect()


@app.command()
def status(
    profile: str = typer.Option("testnet", help="Config profile"),
    log_level: str = typer.Option("WARNING", help="Log level"),
) -> None:
    """Show current bot status (positions, balance, connection)."""
    from src.config.settings import load_settings

    _configure_logging(log_level)
    settings = load_settings(profile)

    asyncio.run(_run_status(settings))


@app.command()
def migrate(
    profile: str = typer.Option("default", help="Config profile"),
) -> None:
    """Initialize or migrate the database schema."""
    from src.config.settings import load_settings
    from src.data.migrations import run_migrations

    settings = load_settings(profile)
    console.print(f"[bold]Running migrations[/] → {settings.database.path}")
    asyncio.run(run_migrations(settings.database.path))
    console.print("[green]Migrations complete.[/]")


if __name__ == "__main__":
    app()
