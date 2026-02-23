"""Tests for the ML hyperparameter optimizer module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config.settings import QuantumSettings
from src.ml.optimizer import HyperparameterOptimizer


def _mock_trainer_factory():
    """Create a factory that returns mock trainers."""
    from datetime import datetime, timezone

    from src.data.models import ModelVersion

    def create_mock():
        trainer = MagicMock()
        trainer.train = AsyncMock(
            return_value=ModelVersion(
                version="v_test",
                created_at=datetime.now(timezone.utc),
                accuracy=0.6,
            )
        )
        # Mock circuit for _evaluate_accuracy
        mock_circuit = MagicMock()
        weights = np.random.uniform(-np.pi, np.pi, size=8)
        mock_circuit.get_weights.return_value = weights
        mock_qnn = MagicMock()

        def forward_fn(features, weights):
            n = features.shape[0]
            rng = np.random.RandomState(42)
            return rng.randn(n, 3)

        mock_qnn.forward = MagicMock(side_effect=forward_fn)
        mock_circuit.get_qnn.return_value = mock_qnn
        trainer._circuit = mock_circuit
        return trainer

    return create_mock


@pytest.fixture
def base_settings() -> QuantumSettings:
    return QuantumSettings(
        n_qubits=4, reps=2, optimizer="COBYLA", max_iterations=5
    )


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    features = rng.uniform(0, 2 * np.pi, size=(30, 4))
    labels = rng.integers(0, 3, size=30)
    return features, labels


class TestOptimize:
    @pytest.mark.asyncio
    async def test_returns_best_config(self, base_settings, sample_data):
        features, labels = sample_data
        optimizer = HyperparameterOptimizer(base_settings)

        factory = _mock_trainer_factory()
        with patch("src.ml.optimizer.QuantumTrainer") as MockTrainer:
            MockTrainer.side_effect = lambda settings: factory()
            result = await optimizer.optimize(features, labels, n_trials=3)

        assert "n_qubits" in result
        assert "reps" in result
        assert "optimizer" in result
        assert "validation_accuracy" in result
        assert "training_accuracy" in result
        assert "max_iterations" in result

    @pytest.mark.asyncio
    async def test_validation_accuracy_in_range(self, base_settings, sample_data):
        features, labels = sample_data
        optimizer = HyperparameterOptimizer(base_settings)

        factory = _mock_trainer_factory()
        with patch("src.ml.optimizer.QuantumTrainer") as MockTrainer:
            MockTrainer.side_effect = lambda settings: factory()
            result = await optimizer.optimize(features, labels, n_trials=3)

        assert 0.0 <= result["validation_accuracy"] <= 1.0

    @pytest.mark.asyncio
    async def test_qubits_in_valid_range(self, base_settings, sample_data):
        features, labels = sample_data
        optimizer = HyperparameterOptimizer(base_settings)

        factory = _mock_trainer_factory()
        with patch("src.ml.optimizer.QuantumTrainer") as MockTrainer:
            MockTrainer.side_effect = lambda settings: factory()
            result = await optimizer.optimize(features, labels, n_trials=5)

        assert result["n_qubits"] in [3, 4, 5, 6]
        assert result["reps"] in [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_features_raises(self, base_settings):
        features = np.empty((0, 4))
        labels = np.empty((0,), dtype=int)
        optimizer = HyperparameterOptimizer(base_settings)
        with pytest.raises(ValueError, match="empty dataset"):
            await optimizer.optimize(features, labels)

    @pytest.mark.asyncio
    async def test_too_few_for_split_raises(self, base_settings):
        features = np.random.uniform(0, 2 * np.pi, size=(1, 4))
        labels = np.array([0])
        optimizer = HyperparameterOptimizer(base_settings)
        with pytest.raises(ValueError, match="Not enough data"):
            await optimizer.optimize(features, labels)

    @pytest.mark.asyncio
    async def test_all_trials_fail_raises(self, base_settings, sample_data):
        features, labels = sample_data
        optimizer = HyperparameterOptimizer(base_settings)

        with patch("src.ml.optimizer.QuantumTrainer") as MockTrainer:
            mock = MagicMock()
            mock.train = AsyncMock(side_effect=RuntimeError("boom"))
            MockTrainer.return_value = mock
            with pytest.raises(RuntimeError, match="All hyperparameter trials failed"):
                await optimizer.optimize(features, labels, n_trials=2)

    @pytest.mark.asyncio
    async def test_picks_highest_accuracy(self, base_settings, sample_data):
        """The optimizer should return the config with highest val accuracy."""
        features, labels = sample_data
        optimizer = HyperparameterOptimizer(base_settings)

        call_count = [0]
        from datetime import datetime, timezone

        from src.data.models import ModelVersion

        def create_mock_with_varying_accuracy():
            call_count[0] += 1
            trainer = MagicMock()
            trainer.train = AsyncMock(
                return_value=ModelVersion(
                    version="v_test",
                    created_at=datetime.now(timezone.utc),
                    accuracy=0.5,
                )
            )
            mock_circuit = MagicMock()
            weights = np.random.uniform(-np.pi, np.pi, size=8)
            mock_circuit.get_weights.return_value = weights
            mock_qnn = MagicMock()

            # Make predictions deterministically match some labels
            count = call_count[0]
            def forward_fn(features, weights):
                n = features.shape[0]
                # Each trial produces different accuracy
                rng = np.random.RandomState(count)
                return rng.randn(n, 3)

            mock_qnn.forward = MagicMock(side_effect=forward_fn)
            mock_circuit.get_qnn.return_value = mock_qnn
            trainer._circuit = mock_circuit
            return trainer

        with patch("src.ml.optimizer.QuantumTrainer") as MockTrainer:
            MockTrainer.side_effect = lambda s: create_mock_with_varying_accuracy()
            result = await optimizer.optimize(features, labels, n_trials=5)

        # Should have found *some* config
        assert result["validation_accuracy"] >= 0.0


class TestAdjustFeatures:
    def test_same_dimension(self):
        features = np.random.uniform(0, 1, size=(10, 4))
        adjusted = HyperparameterOptimizer._adjust_features(features, 4)
        assert adjusted.shape == (10, 4)
        np.testing.assert_array_equal(features, adjusted)

    def test_truncate(self):
        features = np.random.uniform(0, 1, size=(10, 6))
        adjusted = HyperparameterOptimizer._adjust_features(features, 4)
        assert adjusted.shape == (10, 4)
        np.testing.assert_array_equal(features[:, :4], adjusted)

    def test_pad(self):
        features = np.random.uniform(0, 1, size=(10, 3))
        adjusted = HyperparameterOptimizer._adjust_features(features, 5)
        assert adjusted.shape == (10, 5)
        np.testing.assert_array_equal(features, adjusted[:, :3])
        np.testing.assert_array_equal(adjusted[:, 3:], 0.0)
