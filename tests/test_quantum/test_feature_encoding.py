"""Tests for feature encoding (market data to quantum-ready features)."""

import numpy as np
import pandas as pd
import pytest

from src.config.settings import StrategySettings
from src.quantum.feature_encoding import FEATURE_COLS, FeatureEncoder


@pytest.fixture
def strategy_settings() -> StrategySettings:
    return StrategySettings(
        lookback_period=100,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_period=14,
    )


@pytest.fixture
def encoder(strategy_settings) -> FeatureEncoder:
    return FeatureEncoder(strategy_settings)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate a realistic OHLCV DataFrame with 200 rows."""
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 42000.0 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    open_ = close + np.random.randn(n) * 20
    volume = np.abs(np.random.randn(n) * 100 + 500)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestComputeIndicators:
    def test_adds_indicator_columns(self, encoder, sample_df):
        result = encoder.compute_indicators(sample_df)
        assert "returns" in result.columns
        assert "rsi" in result.columns
        assert "macd_hist" in result.columns
        assert "atr" in result.columns

    def test_drops_nan_rows(self, encoder, sample_df):
        result = encoder.compute_indicators(sample_df)
        assert not result["returns"].isna().any()
        assert not result["rsi"].isna().any()
        # Result should be shorter than input due to warm-up
        assert len(result) < len(sample_df)
        assert len(result) > 100  # should still have most rows

    def test_rsi_in_range(self, encoder, sample_df):
        result = encoder.compute_indicators(sample_df)
        assert (result["rsi"] >= 0).all()
        assert (result["rsi"] <= 100).all()

    def test_does_not_modify_input(self, encoder, sample_df):
        original_cols = set(sample_df.columns)
        encoder.compute_indicators(sample_df)
        assert set(sample_df.columns) == original_cols


class TestEncode:
    def test_output_shape(self, encoder, sample_df):
        df = encoder.compute_indicators(sample_df)
        encoded = encoder.encode(df)
        assert encoded.ndim == 2
        assert encoded.shape[1] == 4  # returns, rsi, macd, volume
        assert encoded.shape[0] == len(df)

    def test_values_in_range(self, encoder, sample_df):
        df = encoder.compute_indicators(sample_df)
        encoded = encoder.encode(df)
        assert np.all(encoded >= 0)
        assert np.all(encoded <= 2 * np.pi + 1e-10)

    def test_no_nans(self, encoder, sample_df):
        df = encoder.compute_indicators(sample_df)
        encoded = encoder.encode(df)
        assert not np.any(np.isnan(encoded))

    def test_feature_columns_constant(self):
        assert len(FEATURE_COLS) == 4
        assert "returns" in FEATURE_COLS[0]


class TestEncodeSingle:
    def test_returns_1d_array(self, encoder, sample_df):
        df = encoder.compute_indicators(sample_df)
        single = encoder.encode_single(df)
        assert single.ndim == 1
        assert single.shape == (4,)

    def test_values_in_range(self, encoder, sample_df):
        df = encoder.compute_indicators(sample_df)
        single = encoder.encode_single(df)
        assert np.all(single >= 0)
        assert np.all(single <= 2 * np.pi + 1e-10)

    def test_matches_last_row_of_batch(self, encoder, sample_df):
        df = encoder.compute_indicators(sample_df)
        batch = encoder.encode(df)
        single = encoder.encode_single(df)
        np.testing.assert_array_almost_equal(single, batch[-1])


class TestNormalize:
    def test_zero_maps_to_zero(self):
        result = FeatureEncoder._normalize(np.array([0.0]))
        assert result[0] == 0.0

    def test_one_maps_to_two_pi(self):
        result = FeatureEncoder._normalize(np.array([1.0]))
        np.testing.assert_almost_equal(result[0], 2 * np.pi)

    def test_half_maps_to_pi(self):
        result = FeatureEncoder._normalize(np.array([0.5]))
        np.testing.assert_almost_equal(result[0], np.pi)
