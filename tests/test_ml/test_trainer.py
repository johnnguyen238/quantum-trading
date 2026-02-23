"""Tests for the ML quantum trainer module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.settings import QuantumSettings
from src.ml.trainer import QuantumTrainer


def _mock_qnn(n_classes: int = 3):
    """Create a mock QNN that returns random logits."""
    qnn = MagicMock()

    def forward_fn(features, weights):
        n_samples = features.shape[0]
        # Produce stable outputs based on weights sum for reproducibility
        rng = np.random.RandomState(int(abs(weights.sum()) * 100) % 2**31)
        return rng.randn(n_samples, n_classes)

    qnn.forward = MagicMock(side_effect=forward_fn)
    qnn.num_inputs = 4
    qnn.num_weights = 8
    qnn.output_shape = (n_classes,)
    return qnn


def _mock_circuit(n_weights: int = 8):
    """Create a mock TrendCircuit."""
    circuit = MagicMock()
    weights = np.random.uniform(-np.pi, np.pi, size=n_weights)
    circuit.get_weights.return_value = weights.copy()
    circuit.num_parameters = n_weights
    circuit.build.return_value = MagicMock()
    circuit.build_qnn.return_value = _mock_qnn()
    circuit.get_qnn.return_value = _mock_qnn()

    def set_weights_fn(w):
        weights[:] = w
        circuit.get_weights.return_value = w.copy()

    circuit.set_weights = MagicMock(side_effect=set_weights_fn)
    return circuit


@pytest.fixture
def settings() -> QuantumSettings:
    return QuantumSettings(
        n_qubits=4,
        reps=2,
        optimizer="COBYLA",
        max_iterations=5,  # low for fast tests
    )


@pytest.fixture
def sample_data():
    """Create small sample training data."""
    rng = np.random.default_rng(42)
    features = rng.uniform(0, 2 * np.pi, size=(20, 4))
    labels = rng.integers(0, 3, size=20)
    return features, labels


class TestTrain:
    @pytest.mark.asyncio
    async def test_returns_model_version(self, settings, sample_data):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            result = await trainer.train(features, labels)

        from src.data.models import ModelVersion

        assert isinstance(result, ModelVersion)
        assert result.version.startswith("v")
        assert 0.0 <= result.accuracy <= 1.0

    @pytest.mark.asyncio
    async def test_empty_features_raises(self, settings):
        features = np.empty((0, 4))
        labels = np.empty((0,), dtype=int)
        trainer = QuantumTrainer(settings)
        with pytest.raises(ValueError, match="empty dataset"):
            await trainer.train(features, labels)

    @pytest.mark.asyncio
    async def test_stores_best_weights(self, settings, sample_data):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            await trainer.train(features, labels)

        weights = trainer.get_trained_weights()
        assert isinstance(weights, np.ndarray)
        assert len(weights) > 0

    @pytest.mark.asyncio
    async def test_model_version_has_metadata(self, settings, sample_data):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            result = await trainer.train(features, labels)

        assert "optimizer=COBYLA" in result.notes
        assert len(result.parameters) > 0

    @pytest.mark.asyncio
    async def test_uses_configured_optimizer(self, sample_data):
        features, labels = sample_data
        settings = QuantumSettings(
            n_qubits=4, reps=2, optimizer="Nelder-Mead", max_iterations=3
        )
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            result = await trainer.train(features, labels)

        assert "Nelder-Mead" in result.notes


class TestObjective:
    @pytest.mark.asyncio
    async def test_objective_returns_float(self, settings, sample_data):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            mock_circuit = _mock_circuit()
            MockCircuit.return_value = mock_circuit
            trainer = QuantumTrainer(settings)

            # Set up internal state like train() would
            trainer._circuit = mock_circuit
            trainer._train_features = features
            trainer._train_labels = labels
            trainer._iteration = 0
            trainer._best_loss = float("inf")
            trainer._best_weights = np.zeros(8)

            weights = np.random.uniform(-np.pi, np.pi, size=8)
            loss = trainer._objective(weights)

        assert isinstance(loss, float)
        assert loss > 0  # cross-entropy is positive

    @pytest.mark.asyncio
    async def test_objective_tracks_iteration(self, settings, sample_data):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            mock_circuit = _mock_circuit()
            MockCircuit.return_value = mock_circuit
            trainer = QuantumTrainer(settings)
            trainer._circuit = mock_circuit
            trainer._train_features = features
            trainer._train_labels = labels
            trainer._iteration = 0
            trainer._best_loss = float("inf")
            trainer._best_weights = np.zeros(8)

            weights = np.random.uniform(-np.pi, np.pi, size=8)
            trainer._objective(weights)
            trainer._objective(weights)

        assert trainer._iteration == 2


class TestGetTrainedWeights:
    def test_raises_before_training(self, settings):
        trainer = QuantumTrainer(settings)
        with pytest.raises(RuntimeError, match="No trained weights"):
            trainer.get_trained_weights()

    @pytest.mark.asyncio
    async def test_returns_copy(self, settings, sample_data):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            await trainer.train(features, labels)

        w1 = trainer.get_trained_weights()
        w2 = trainer.get_trained_weights()
        assert w1 is not w2  # should be a copy
        np.testing.assert_array_equal(w1, w2)


class TestSaveModel:
    def test_raises_before_training(self, settings, tmp_path):
        trainer = QuantumTrainer(settings)
        with pytest.raises(RuntimeError, match="No trained weights"):
            trainer.save_model(str(tmp_path / "model"))

    @pytest.mark.asyncio
    async def test_saves_weights_and_metadata(self, settings, sample_data, tmp_path):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            await trainer.train(features, labels)

        model_path = str(tmp_path / "model")
        trainer.save_model(model_path)

        import json
        from pathlib import Path

        # Check .npy exists
        assert Path(model_path).with_suffix(".npy").exists()
        # Check .json exists
        meta_path = Path(model_path).with_suffix(".json")
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["n_qubits"] == 4
        assert meta["optimizer"] == "COBYLA"

    @pytest.mark.asyncio
    async def test_saves_with_custom_metadata(
        self, settings, sample_data, tmp_path
    ):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            await trainer.train(features, labels)

        model_path = str(tmp_path / "model")
        trainer.save_model(model_path, metadata={"custom_key": "custom_value"})

        import json
        from pathlib import Path

        with open(Path(model_path).with_suffix(".json")) as f:
            meta = json.load(f)
        assert meta["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(
        self, settings, sample_data, tmp_path
    ):
        features, labels = sample_data
        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = _mock_circuit()
            trainer = QuantumTrainer(settings)
            await trainer.train(features, labels)

        from pathlib import Path

        model_path = str(tmp_path / "nested" / "dir" / "model")
        trainer.save_model(model_path)
        assert Path(model_path).with_suffix(".npy").exists()
