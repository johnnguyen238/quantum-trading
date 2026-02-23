"""Tests for the quantum trend detector."""

import numpy as np
import pytest

from src.config.constants import TrendDirection
from src.config.settings import QuantumSettings, StrategySettings
from src.quantum.signal import TrendSignal
from src.quantum.trend_detector import TrendDetector


@pytest.fixture
def quantum_settings() -> QuantumSettings:
    """Use small circuit for fast tests."""
    return QuantumSettings(n_qubits=2, reps=1, confidence_threshold=0.4)


@pytest.fixture
def strategy_settings() -> StrategySettings:
    return StrategySettings()


@pytest.fixture
async def detector(quantum_settings, strategy_settings) -> TrendDetector:
    """Create and initialize a TrendDetector."""
    td = TrendDetector(quantum_settings, strategy_settings)
    await td.initialize()
    return td


class TestTrendDetector:
    @pytest.mark.asyncio
    async def test_initialize(self, quantum_settings, strategy_settings):
        td = TrendDetector(quantum_settings, strategy_settings)
        await td.initialize()
        assert td._initialized is True
        assert td._qnn is not None
        assert td._circuit is not None

    @pytest.mark.asyncio
    async def test_predict_returns_trend_signal(self, detector):
        features = np.random.uniform(0, 2 * np.pi, size=(2,))
        signal = await detector.predict(features)
        assert isinstance(signal, TrendSignal)
        assert signal.direction in list(TrendDirection)
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_predict_confidence_sums_to_one(self, detector):
        """The softmax probabilities behind the prediction should sum to ~1."""
        # We test indirectly: confidence must be at least 1/3 (random chance)
        features = np.random.uniform(0, 2 * np.pi, size=(2,))
        signal = await detector.predict(features)
        assert signal.confidence >= 1.0 / 3.0 - 0.01

    @pytest.mark.asyncio
    async def test_predict_includes_features(self, detector):
        features = np.array([1.0, 2.0])
        signal = await detector.predict(features)
        assert len(signal.features) > 0

    @pytest.mark.asyncio
    async def test_predict_includes_timestamp(self, detector):
        features = np.random.uniform(0, 2 * np.pi, size=(2,))
        signal = await detector.predict(features)
        assert signal.timestamp is not None

    @pytest.mark.asyncio
    async def test_predict_not_initialized_raises(self, quantum_settings, strategy_settings):
        td = TrendDetector(quantum_settings, strategy_settings)
        with pytest.raises(RuntimeError, match="not initialized"):
            await td.predict(np.array([1.0, 2.0]))

    @pytest.mark.asyncio
    async def test_predict_batch(self, detector):
        features = np.random.uniform(0, 2 * np.pi, size=(5, 2))
        signals = await detector.predict_batch(features)
        assert len(signals) == 5
        for s in signals:
            assert isinstance(s, TrendSignal)
            assert s.direction in list(TrendDirection)

    @pytest.mark.asyncio
    async def test_predict_deterministic_with_same_weights(
        self, quantum_settings, strategy_settings
    ):
        """Same features + same weights should give same prediction."""
        td1 = TrendDetector(quantum_settings, strategy_settings)
        await td1.initialize()
        weights = td1._circuit.get_weights()

        td2 = TrendDetector(quantum_settings, strategy_settings)
        await td2.initialize()
        td2._circuit.set_weights(weights)

        features = np.array([1.0, 2.0])
        s1 = await td1.predict(features)
        s2 = await td2.predict(features)
        assert s1.direction == s2.direction
        assert abs(s1.confidence - s2.confidence) < 1e-6


class TestWeightsIO:
    @pytest.mark.asyncio
    async def test_save_and_load_weights(self, detector, tmp_path):
        path = str(tmp_path / "test_weights.npy")
        detector.save_weights(path)

        original_weights = detector._circuit.get_weights()

        # Modify weights
        detector._circuit.set_weights(np.zeros(detector._circuit.num_parameters))

        # Load saved weights
        detector.load_weights(path)
        loaded_weights = detector._circuit.get_weights()
        np.testing.assert_array_almost_equal(loaded_weights, original_weights)

    @pytest.mark.asyncio
    async def test_save_creates_directory(self, detector, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "weights.npy")
        detector.save_weights(path)
        assert (tmp_path / "sub" / "dir" / "weights.npy").exists()

    @pytest.mark.asyncio
    async def test_load_weights_from_init(self, quantum_settings, strategy_settings, tmp_path):
        """Test that model_weights_path loads on initialize."""
        # First, create a detector and save its weights
        td1 = TrendDetector(quantum_settings, strategy_settings)
        await td1.initialize()
        path = str(tmp_path / "init_weights.npy")
        td1.save_weights(path)
        saved_weights = td1._circuit.get_weights()

        # Create new detector with the weights path
        td2 = TrendDetector(quantum_settings, strategy_settings, model_weights_path=path)
        await td2.initialize()
        loaded_weights = td2._circuit.get_weights()
        np.testing.assert_array_almost_equal(loaded_weights, saved_weights)
