"""Tests for quantum circuits."""

import numpy as np
import pytest

from src.config.settings import QuantumSettings
from src.quantum.circuits import TrendCircuit, _build_observables


@pytest.fixture
def quantum_settings() -> QuantumSettings:
    return QuantumSettings(n_qubits=4, reps=2)


@pytest.fixture
def small_settings() -> QuantumSettings:
    """Smaller circuit for faster tests."""
    return QuantumSettings(n_qubits=2, reps=1)


class TestBuildObservables:
    def test_returns_three_observables(self):
        obs = _build_observables(4)
        assert len(obs) == 3

    def test_observables_are_valid(self):
        obs = _build_observables(4)
        for o in obs:
            assert o.num_qubits == 4


class TestTrendCircuit:
    def test_build_creates_circuit(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        circuit = tc.build()
        assert circuit is not None
        assert circuit.num_qubits == 4

    def test_circuit_has_correct_param_counts(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        tc.build()
        # ZZFeatureMap(4, reps=2) = 4 input params
        # RealAmplitudes(4, reps=2) = 12 trainable params
        assert tc.num_parameters == 12
        assert tc.num_features == 4

    def test_get_circuit_builds_lazily(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        circuit = tc.get_circuit()
        assert circuit is not None
        # Total params = feature_map(4) + ansatz(12) = 16
        assert circuit.num_parameters == 16

    def test_weights_initialized_after_build(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        tc.build()
        weights = tc.get_weights()
        assert weights.shape == (12,)
        assert np.all(np.abs(weights) <= np.pi)

    def test_set_and_get_weights(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        tc.build()
        new_weights = np.ones(12) * 0.5
        tc.set_weights(new_weights)
        retrieved = tc.get_weights()
        np.testing.assert_array_almost_equal(retrieved, new_weights)

    def test_set_weights_wrong_shape_raises(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        tc.build()
        with pytest.raises(ValueError, match="Expected weights"):
            tc.set_weights(np.ones(5))

    def test_get_weights_returns_copy(self, quantum_settings):
        tc = TrendCircuit(quantum_settings)
        tc.build()
        w1 = tc.get_weights()
        w1[:] = 999.0
        w2 = tc.get_weights()
        assert not np.array_equal(w1, w2)

    def test_build_qnn(self, small_settings):
        tc = TrendCircuit(small_settings)
        qnn = tc.build_qnn()
        assert qnn.num_inputs == 2
        assert qnn.output_shape == (3,)

    def test_get_qnn_builds_lazily(self, small_settings):
        tc = TrendCircuit(small_settings)
        qnn = tc.get_qnn()
        assert qnn is not None
        assert qnn.num_inputs == 2

    def test_qnn_forward_pass(self, small_settings):
        tc = TrendCircuit(small_settings)
        qnn = tc.build_qnn()
        x = np.random.uniform(0, 2 * np.pi, size=(1, 2))
        w = tc.get_weights()
        result = qnn.forward(x, w)
        assert result.shape == (1, 3)

    def test_different_reps_changes_params(self):
        s1 = QuantumSettings(n_qubits=4, reps=1)
        s2 = QuantumSettings(n_qubits=4, reps=3)
        tc1 = TrendCircuit(s1)
        tc2 = TrendCircuit(s2)
        tc1.build()
        tc2.build()
        assert tc1.num_parameters < tc2.num_parameters
