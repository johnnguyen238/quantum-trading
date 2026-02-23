# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Quantum Algorithm Trading Bot** — an automated cryptocurrency trading bot that uses a Variational Quantum Classifier (VQC) to identify market trends and execute Long/Short trades on Bybit.

## Core Requirements (from require.md)

- **Trend Detection:** VQC using ZZFeatureMap + RealAmplitudes ansatz; encodes RSI, MACD, returns, volume → LONG/SHORT/NEUTRAL + confidence
- **Position Management:**
  - Scenario 1 (HOLD): Signal matches position AND unrealized PnL > 0
  - Scenario 2 (DCA): Signal matches position AND unrealized PnL ≤ 0 → place x2 DCA orders
  - Scenario 3 (REVERSAL): Signal flipped from position → close immediately
- **Leverage:** Inverse ATR-based volatility × signal confidence, capped at configurable max
- **Backtesting:** SQLite storage of OHLCV, signals, trades, positions, model versions, performance
- **AI Retraining:** Walk-forward validation with COBYLA/SPSA optimizers
- **Exchange:** Bybit testnet (credentials in `.env`, NOT in code)

## Tech Stack

- Python 3.11+ with `pyproject.toml` (PEP 621)
- Qiskit (qiskit, qiskit-aer, qiskit-machine-learning) — quantum circuits
- ccxt (async) — Bybit exchange connectivity
- pandas / numpy / ta — data processing and technical indicators
- aiosqlite — async SQLite persistence
- Pydantic Settings — config management (YAML + .env merge)
- Typer + Rich — CLI interface
- pytest + pytest-asyncio — testing
- ruff — linting

## Project Structure

```
src/
├── main.py              # CLI (typer) — all commands fully wired ✅ implemented
├── config/
│   ├── settings.py      # Pydantic Settings (YAML + .env merge) ✅ implemented
│   └── constants.py     # Enums (TrendDirection, Scenario, etc.) ✅ implemented
├── core/
│   ├── engine.py        # Main trading loop (orchestrator) ✅ implemented
│   ├── order_manager.py # Order lifecycle ✅ implemented
│   ├── position_manager.py # Position tracking ✅ implemented
│   └── risk_manager.py  # Leverage/exposure limits ✅ implemented
├── quantum/
│   ├── circuits.py      # VQC: ZZFeatureMap + RealAmplitudes + EstimatorQNN ✅ implemented
│   ├── feature_encoding.py # RSI/MACD/ATR/returns → [0, 2π] encoding ✅ implemented
│   ├── trend_detector.py   # Quantum inference → TrendSignal (softmax 3-class) ✅ implemented
│   └── signal.py        # TrendSignal dataclass ✅ implemented
├── exchange/
│   ├── client.py        # Async ccxt Bybit wrapper ✅ implemented
│   ├── market_feed.py   # OHLCV fetching + cache ✅ implemented
│   └── executor.py      # Order placement ✅ implemented
├── strategy/
│   ├── base.py          # Abstract strategy + TradeAction/StrategyResult ✅ implemented
│   ├── quantum_trend.py # 3-scenario HOLD/DCA/REVERSAL logic ✅ implemented
│   ├── dca.py           # x2 DCA order generation + avg price calc ✅ implemented
│   └── leverage.py      # Inverse ATR × confidence leverage calc ✅ implemented
├── backtest/
│   ├── engine.py        # Replay historical data ✅ implemented
│   ├── data_loader.py   # Load historical OHLCV ✅ implemented
│   ├── simulator.py     # Simulated order fills ✅ implemented
│   └── reporter.py      # Performance metrics + BacktestReport dataclass ✅ implemented
├── ml/
│   ├── trainer.py       # Quantum circuit retraining (COBYLA/SPSA) ✅ implemented
│   ├── optimizer.py     # Hyperparameter tuning (random search) ✅ implemented
│   ├── evaluator.py     # Walk-forward validation ✅ implemented
│   └── feature_engineering.py # Labeled dataset creation ✅ implemented
└── data/
    ├── database.py      # SQLite connection (aiosqlite) ✅ implemented
    ├── models.py        # Dataclass models ✅ implemented
    ├── repository.py    # CRUD operations ✅ implemented
    └── migrations.py    # Schema creation ✅ implemented
```

## Build & Run Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# CLI help
python -m src.main --help

# Run live trading bot (testnet)
python -m src.main run --profile testnet

# Run backtest
python -m src.main backtest --symbol "BTC/USDT:USDT" --start 2024-01-01 --end 2024-12-31

# Train quantum model
python -m src.main train --symbol "BTC/USDT:USDT"

# Initialize database
python -m src.main migrate

# Lint
ruff check src/

# Test
pytest tests/
```

## Project Status

**Phase 1 COMPLETE** — Full skeleton with all modules stubbed.
**Phase 2 COMPLETE** — Data layer and exchange connectivity.
**Phase 3 COMPLETE** — Quantum module (circuits, encoding, detector).
**Phase 4 COMPLETE** — Strategy module (quantum trend, DCA, leverage).
**Phase 5 COMPLETE** — Core engine module fully implemented:
- TradingEngine: Full orchestrator with async loop, per-symbol ticks, graceful start/stop
- OrderManager: Translates TradeAction → exchange orders, handles all action types, batch submission
- PositionManager: Open/DCA/close positions, weighted average entry, PnL calculation, exchange sync
- RiskManager: Balance validation, max positions check, leverage clamping, position size limits

**Phase 6 COMPLETE** — Backtest module fully implemented:
- BacktestEngine: Walk-forward replay with bar-by-bar stepping, warmup period, force-close at end
- DataLoader: Cache-first loading with exchange fallback when data insufficient
- OrderSimulator: Slippage simulation, fee calculation, DCA position updates, PnL tracking
- BacktestReporter: Win rate, profit factor, Sharpe ratio, max drawdown, Rich summary, CSV export

**Phase 7 COMPLETE** — ML retraining module fully implemented:
- FeatureEngineer: Forward-looking label generation (LONG/SHORT/NEUTRAL), FeatureEncoder integration for [0, 2π] features
- QuantumTrainer: TrendCircuit + scipy.optimize.minimize (COBYLA/Nelder-Mead/Powell), cross-entropy loss, model save/load (.npy + .json)
- WalkForwardEvaluator: Chronological train/val splits, per-fold accuracy/precision/recall/F1, macro-averaged metrics
- HyperparameterOptimizer: Random search over n_qubits/reps/optimizer/max_iterations, feature dimension adjustment

**Phase 8 COMPLETE** — Final integration and polish:
- All CLI commands fully wired: run (with graceful SIGINT shutdown), backtest (DB + detector init, Rich summary, CSV export), train (data load → feature engineering → QuantumTrainer → save model + DB persist), evaluate (walk-forward with Rich table output), status (exchange balance, positions, model version), migrate
- Integration tests for all CLI commands and end-to-end pipelines (backtest, train, evaluate, status)
- 334 tests passing, 0 skipped, ruff clean

**ALL PHASES COMPLETE** — The project is fully implemented from skeleton to production-ready CLI.

## Architecture Notes

- **Bybit futures symbols** use `BTC/USDT:USDT` format (not `BTC/USDT`)
- **Async throughout** — ccxt async for parallel multi-pair fetching/execution
- **Data flow**: Bybit → market_feed → feature_encoding → trend_detector → strategy → risk_manager → order_manager → executor → repository

## Security Note

Bybit testnet API credentials are stored in `require.md`. These must never be committed to a public repository. Use `.env` file (gitignored) for all API keys and secrets.
