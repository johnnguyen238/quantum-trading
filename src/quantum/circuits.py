"""Parameterized quantum circuits for trend classification.

Uses Qiskit 2.x function-based API: ``zz_feature_map`` for data encoding
and ``real_amplitudes`` as the trainable ansatz, combined into a VQC.
The ``EstimatorQNN`` maps expectation values from 3 observables to
LONG / SHORT / NEUTRAL class logits.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

if TYPE_CHECKING:
    from src.config.settings import QuantumSettings

logger = logging.getLogger(__name__)


def _build_observables(n_qubits: int) -> list[SparsePauliOp]:
    """Build 3 single-qubit Z observables for the 3 output classes.

    Each observable measures a different qubit so the network can
    learn to route class information to separate qubits.
    """
    labels = []
    for i in range(3):
        qubit_idx = i % n_qubits
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - qubit_idx] = "Z"
        labels.append("".join(pauli))
    return [SparsePauliOp.from_list([(label, 1.0)]) for label in labels]


class TrendCircuit:
    """Builds and manages the parameterized quantum circuit.

    Parameters
    ----------
    settings:
        Quantum configuration (n_qubits, reps, feature_map, ansatz).
    """

    def __init__(self, settings: "QuantumSettings") -> None:
        self._settings = settings
        self._circuit: QuantumCircuit | None = None
        self._feature_map: QuantumCircuit | None = None
        self._ansatz: QuantumCircuit | None = None
        self._qnn: EstimatorQNN | None = None
        self._weights: np.ndarray | None = None

    def build(self) -> QuantumCircuit:
        """Construct the full VQC circuit (feature map + ansatz).

        Returns the composed ``QuantumCircuit``.
        """
        n_qubits = self._settings.n_qubits
        reps = self._settings.reps

        self._feature_map = zz_feature_map(feature_dimension=n_qubits, reps=reps)
        self._ansatz = real_amplitudes(num_qubits=n_qubits, reps=reps)
        self._circuit = self._feature_map.compose(self._ansatz)

        # Initialize random weights
        self._weights = np.random.uniform(
            -np.pi, np.pi, size=self._ansatz.num_parameters
        )

        logger.info(
            "Built VQC circuit: %d qubits, %d feature params, %d trainable params",
            n_qubits,
            self._feature_map.num_parameters,
            self._ansatz.num_parameters,
        )
        return self._circuit

    def get_circuit(self) -> QuantumCircuit:
        """Return the built circuit, building it if necessary."""
        if self._circuit is None:
            self.build()
        return self._circuit

    def build_qnn(self) -> EstimatorQNN:
        """Build the EstimatorQNN with 3 observables for trend classification.

        Returns
        -------
        An ``EstimatorQNN`` ready for forward/backward passes.
        """
        circuit = self.get_circuit()
        observables = _build_observables(self._settings.n_qubits)

        self._qnn = EstimatorQNN(
            circuit=circuit,
            input_params=list(self._feature_map.parameters),
            weight_params=list(self._ansatz.parameters),
            observables=observables,
        )
        logger.info(
            "Built EstimatorQNN: %d inputs, %d weights, %d outputs",
            self._qnn.num_inputs,
            self._qnn.num_weights,
            self._qnn.output_shape[0],
        )
        return self._qnn

    def get_qnn(self) -> EstimatorQNN:
        """Return the QNN, building it if necessary."""
        if self._qnn is None:
            self.build_qnn()
        return self._qnn

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters in the ansatz."""
        if self._ansatz is None:
            self.build()
        return self._ansatz.num_parameters

    @property
    def num_features(self) -> int:
        """Number of input features (= n_qubits)."""
        return self._settings.n_qubits

    def set_weights(self, weights: np.ndarray) -> None:
        """Load trained weights into the circuit."""
        expected = self.num_parameters
        if weights.shape != (expected,):
            raise ValueError(
                f"Expected weights of shape ({expected},), got {weights.shape}"
            )
        self._weights = weights.copy()

    def get_weights(self) -> np.ndarray:
        """Return current circuit weights."""
        if self._weights is None:
            # Ensure circuit is built so weights get initialized
            self.get_circuit()
        return self._weights.copy()
