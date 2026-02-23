"""Application settings — merges config YAML files with .env overrides via Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_yaml(profile: str = "default") -> dict[str, Any]:
    """Load and merge YAML config files.

    Loads ``default.yaml`` first, then overlays the requested profile.
    """
    base: dict[str, Any] = {}
    default_path = _CONFIG_DIR / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            base = yaml.safe_load(f) or {}

    if profile != "default":
        overlay_path = _CONFIG_DIR / f"{profile}.yaml"
        if overlay_path.exists():
            with open(overlay_path) as f:
                overlay = yaml.safe_load(f) or {}
            base = _deep_merge(base, overlay)
    return base


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ExchangeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BYBIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = "bybit"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    rate_limit: int = 50


class TradingSettings(BaseSettings):
    symbols: list[str] = ["BTC/USDT:USDT"]
    timeframe: str = "15m"
    max_leverage: int = 10
    risk_per_trade: float = 0.02
    max_open_positions: int = 3
    dca_multiplier: int = 2
    max_dca_layers: int = 3


class QuantumSettings(BaseSettings):
    n_qubits: int = 4
    reps: int = 2
    feature_map: str = "ZZFeatureMap"
    ansatz: str = "RealAmplitudes"
    optimizer: str = "COBYLA"
    max_iterations: int = 100
    confidence_threshold: float = 0.6


class StrategySettings(BaseSettings):
    lookback_period: int = 100
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14


class BacktestSettings(BaseSettings):
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000.0
    fee_rate: float = 0.0006
    slippage: float = 0.0001


class DatabaseSettings(BaseSettings):
    path: str = "data/trading.db"


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Top-level application settings.

    Build order:
    1. Load ``config/default.yaml``
    2. Overlay profile YAML (e.g. ``testnet.yaml``)
    3. Override with environment variables / ``.env``
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    quantum: QuantumSettings = Field(default_factory=QuantumSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(profile: str = "default") -> Settings:
    """Create a ``Settings`` instance from YAML + env vars.

    Parameters
    ----------
    profile:
        Config profile name (maps to ``config/<profile>.yaml``).
        Use ``"testnet"`` or ``"backtest"``.
    """
    yaml_data = _load_yaml(profile)

    return Settings(
        exchange=ExchangeSettings(**(yaml_data.get("exchange", {}))),
        trading=TradingSettings(**(yaml_data.get("trading", {}))),
        quantum=QuantumSettings(**(yaml_data.get("quantum", {}))),
        strategy=StrategySettings(**(yaml_data.get("strategy", {}))),
        backtest=BacktestSettings(**(yaml_data.get("backtest", {}))),
        database=DatabaseSettings(**(yaml_data.get("database", {}))),
        logging=LoggingSettings(**(yaml_data.get("logging", {}))),
    )
