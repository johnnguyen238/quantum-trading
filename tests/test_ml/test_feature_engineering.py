"""Tests for the ML feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import StrategySettings
from src.ml.feature_engineering import FeatureEngineer


def _make_df(n: int = 100) -> pd.DataFrame:
    """Create a realistic OHLCV DataFrame."""
    base_price = 42000.0
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "open": [base_price + i * 10 for i in range(n)],
        "high": [base_price + i * 10 + 50 for i in range(n)],
        "low": [base_price + i * 10 - 50 for i in range(n)],
        "close": [base_price + i * 10 + 20 for i in range(n)],
        "volume": [100.0 + i for i in range(n)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def settings() -> StrategySettings:
    return StrategySettings()


@pytest.fixture
def engineer(settings: StrategySettings) -> FeatureEngineer:
    return FeatureEngineer(settings, forward_period=5, threshold=0.01)


class TestCreateDataset:
    def test_returns_features_and_labels(self, engineer):
        df = _make_df(100)
        features, labels = engineer.create_dataset(df)
        assert features.ndim == 2
        assert labels.ndim == 1
        assert len(features) == len(labels)
        assert len(features) > 0

    def test_features_shape(self, engineer):
        df = _make_df(100)
        features, labels = engineer.create_dataset(df)
        assert features.shape[1] == 6  # returns, rsi, macd, volume, ema_ratio, bb

    def test_features_in_valid_range(self, engineer):
        df = _make_df(100)
        features, _ = engineer.create_dataset(df)
        # Features should be in [0, 2*pi]
        assert np.all(features >= 0)
        assert np.all(features <= 2 * np.pi + 0.01)

    def test_labels_are_valid_classes(self, engineer):
        df = _make_df(100)
        _, labels = engineer.create_dataset(df)
        assert set(labels).issubset({0, 1, 2})

    def test_empty_dataframe(self, engineer):
        df = pd.DataFrame()
        features, labels = engineer.create_dataset(df)
        assert len(features) == 0
        assert len(labels) == 0

    def test_too_few_rows(self, engineer):
        df = _make_df(3)  # forward_period=5, needs at least 7
        features, labels = engineer.create_dataset(df)
        assert len(features) == 0

    def test_custom_threshold(self, settings):
        # Very low threshold → more extreme labels
        eng_low = FeatureEngineer(settings, forward_period=5, threshold=0.0001)
        df = _make_df(100)
        _, labels_low = eng_low.create_dataset(df)

        # Very high threshold → mostly neutral
        eng_high = FeatureEngineer(settings, forward_period=5, threshold=0.5)
        _, labels_high = eng_high.create_dataset(df)

        # More extremes with low threshold
        neutral_low = np.sum(labels_low == 1)
        neutral_high = np.sum(labels_high == 1)
        assert neutral_high >= neutral_low

    def test_forward_period_affects_label_count(self, settings):
        df = _make_df(100)
        eng5 = FeatureEngineer(settings, forward_period=5, threshold=0.01)
        eng10 = FeatureEngineer(settings, forward_period=10, threshold=0.01)

        feat5, _ = eng5.create_dataset(df)
        feat10, _ = eng10.create_dataset(df)

        # More forward period → fewer valid labels (more trimmed)
        assert len(feat5) >= len(feat10)


class TestGenerateLabels:
    def test_uptrend_labels_long(self, settings):
        """Strongly trending up should produce LONG labels."""
        n = 50
        prices = [42000 + i * 100 for i in range(n)]  # +100 per bar
        df = pd.DataFrame({
            "close": prices,
            "open": prices,
            "high": [p + 50 for p in prices],
            "low": [p - 50 for p in prices],
            "volume": [100.0] * n,
        })
        eng = FeatureEngineer(settings, forward_period=5, threshold=0.005)
        labels = eng._generate_labels(df)
        # Most labels should be LONG (2) given strong uptrend
        long_count = np.sum(labels == 2)
        assert long_count > len(labels) * 0.5

    def test_downtrend_labels_short(self, settings):
        """Strongly trending down should produce SHORT labels."""
        n = 50
        prices = [50000 - i * 100 for i in range(n)]
        df = pd.DataFrame({
            "close": prices,
            "open": prices,
            "high": [p + 50 for p in prices],
            "low": [p - 50 for p in prices],
            "volume": [100.0] * n,
        })
        eng = FeatureEngineer(settings, forward_period=5, threshold=0.005)
        labels = eng._generate_labels(df)
        short_count = np.sum(labels == 0)
        assert short_count > len(labels) * 0.5

    def test_flat_labels_neutral(self, settings):
        """Flat price should produce NEUTRAL labels."""
        n = 50
        prices = [42000.0] * n
        df = pd.DataFrame({
            "close": prices,
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "volume": [100.0] * n,
        })
        eng = FeatureEngineer(settings, forward_period=5, threshold=0.01)
        labels = eng._generate_labels(df)
        neutral_count = np.sum(labels == 1)
        assert neutral_count == len(labels)

    def test_label_count_matches_forward_period(self, settings):
        """Labels should trim the last forward_period rows."""
        df = _make_df(50)
        eng = FeatureEngineer(settings, forward_period=5, threshold=0.01)
        labels = eng._generate_labels(df)
        assert len(labels) == 50 - 5
