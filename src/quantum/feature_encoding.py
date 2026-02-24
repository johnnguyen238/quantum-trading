"""Encode market data features into qubit-compatible values.

Maps raw indicator values (returns, RSI, MACD, volume) into the ``[0, 2*pi]``
range suitable for quantum feature map rotation gates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import ta

if TYPE_CHECKING:
    from src.config.settings import StrategySettings

logger = logging.getLogger(__name__)

# Feature columns extracted for the quantum circuit
FEATURE_COLS = [
    "returns",
    "rsi_norm",
    "macd_norm",
    "volume_norm",
    "ema_ratio",
    "bb_position",
]


class FeatureEncoder:
    """Transforms a DataFrame of OHLCV data into quantum-ready feature vectors.

    Parameters
    ----------
    settings:
        Strategy configuration (lookback periods, indicator params).
    """

    def __init__(self, settings: "StrategySettings") -> None:
        self._settings = settings

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, ATR, EMA ratio, Bollinger Band, and returns columns.

        Parameters
        ----------
        df:
            Must contain columns: open, high, low, close, volume.

        Returns
        -------
        DataFrame with added indicator columns.
        """
        out = df.copy()

        # Log returns
        out["returns"] = np.log(out["close"] / out["close"].shift(1))

        # RSI
        out["rsi"] = ta.momentum.rsi(
            out["close"], window=self._settings.rsi_period, fillna=True
        )

        # MACD histogram
        macd = ta.trend.MACD(
            out["close"],
            window_fast=self._settings.macd_fast,
            window_slow=self._settings.macd_slow,
            window_sign=self._settings.macd_signal,
            fillna=True,
        )
        out["macd_hist"] = macd.macd_diff()

        # ATR (used by leverage calculator, included for completeness)
        out["atr"] = ta.volatility.average_true_range(
            out["high"],
            out["low"],
            out["close"],
            window=self._settings.atr_period,
            fillna=True,
        )

        # EMA ratio: short EMA / long EMA (captures trend direction)
        ema_fast = out["close"].ewm(span=self._settings.macd_fast, adjust=False).mean()
        ema_slow = out["close"].ewm(span=self._settings.macd_slow, adjust=False).mean()
        out["ema_ratio"] = ema_fast / ema_slow

        # Bollinger Band %B: (price - lower) / (upper - lower)
        bb = ta.volatility.BollingerBands(
            out["close"], window=20, window_dev=2, fillna=True
        )
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = bb_upper - bb_lower
        out["bb_pctb"] = np.where(
            bb_range > 0, (out["close"] - bb_lower) / bb_range, 0.5
        )

        # Drop rows with NaN from indicator warm-up
        out = out.dropna(
            subset=["returns", "rsi", "macd_hist", "ema_ratio", "bb_pctb"]
        ).reset_index(drop=True)

        return out

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize features to ``[0, 2*pi]``.

        Parameters
        ----------
        df:
            DataFrame with indicator columns (call ``compute_indicators`` first).

        Returns
        -------
        Array of shape ``(n_samples, n_features)`` with values in ``[0, 2*pi]``.
        """
        features = self._extract_raw(df)
        return self._normalize(features)

    def encode_single(self, df: pd.DataFrame) -> np.ndarray:
        """Encode only the latest row (for live inference).

        Returns
        -------
        Array of shape ``(n_features,)``.
        """
        encoded = self.encode(df)
        if len(encoded) == 0:
            raise ValueError("No valid rows to encode after indicator computation")
        return encoded[-1]

    def _extract_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Pull raw feature values from the indicator DataFrame.

        Returns array of shape ``(n_samples, 6)`` with columns:
        [returns, rsi, macd, volume, ema_ratio, bb_position].
        All values scaled to ``[0, 1]``.
        """
        returns = df["returns"].values
        rsi = df["rsi"].values / 100.0  # RSI is [0, 100] → [0, 1]

        # MACD histogram: scale to [-1, 1] using tanh-like clipping
        macd = df["macd_hist"].values
        macd_max = np.abs(macd).max()
        if macd_max > 0:
            macd_scaled = macd / macd_max  # → [-1, 1]
        else:
            macd_scaled = np.zeros_like(macd)

        # Volume: normalize using rolling z-score → sigmoid to [0, 1]
        vol = df["volume"].values.astype(float)
        vol_mean = np.mean(vol) if len(vol) > 0 else 1.0
        vol_std = np.std(vol) if len(vol) > 1 else 1.0
        if vol_std < 1e-10:
            vol_std = 1.0
        vol_z = (vol - vol_mean) / vol_std
        vol_scaled = 1.0 / (1.0 + np.exp(-vol_z))  # sigmoid → [0, 1]

        # Returns: clip to [-0.1, 0.1] then scale to [0, 1]
        ret_clipped = np.clip(returns, -0.1, 0.1)
        ret_scaled = (ret_clipped + 0.1) / 0.2  # → [0, 1]

        # EMA ratio: clip to [0.95, 1.05] then scale to [0, 1]
        ema = df["ema_ratio"].values
        ema_clipped = np.clip(ema, 0.95, 1.05)
        ema_scaled = (ema_clipped - 0.95) / 0.10  # → [0, 1]

        # Bollinger Band %B: already roughly in [0, 1], clip for safety
        bb = np.clip(df["bb_pctb"].values, 0.0, 1.0)

        return np.column_stack([
            ret_scaled, rsi, macd_scaled * 0.5 + 0.5, vol_scaled,
            ema_scaled, bb,
        ])

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        """Map features from [0, 1] to [0, 2*pi]."""
        return features * (2.0 * np.pi)
