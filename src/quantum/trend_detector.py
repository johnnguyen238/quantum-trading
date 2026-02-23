"""Quantum inference — runs the VQC circuit and produces a TrendSignal."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import softmax

from src.config.constants import Scenario, TrendDirection
from src.quantum.circuits import TrendCircuit
from src.quantum.feature_encoding import FEATURE_COLS
from src.quantum.signal import TrendSignal

if TYPE_CHECKING:
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    from src.config.settings import QuantumSettings, StrategySettings

logger = logging.getLogger(__name__)

# Maps class indices to TrendDirection
_CLASS_MAP: dict[int, TrendDirection] = {
    0: TrendDirection.SHORT,
    1: TrendDirection.NEUTRAL,
    2: TrendDirection.LONG,
}


class TrendDetector:
    """Run quantum circuit inference on market features to detect trends.

    Combines ``TrendCircuit`` for the quantum model and ``FeatureEncoder``
    for data preparation.

    Parameters
    ----------
    quantum_settings:
        Quantum circuit configuration.
    strategy_settings:
        Indicator / feature configuration.
    model_weights_path:
        Optional path to a saved ``.npy`` weights file.
    """

    def __init__(
        self,
        quantum_settings: "QuantumSettings",
        strategy_settings: "StrategySettings",
        model_weights_path: str | None = None,
    ) -> None:
        self._quantum_settings = quantum_settings
        self._strategy_settings = strategy_settings
        self._model_weights_path = model_weights_path
        self._circuit: TrendCircuit | None = None
        self._qnn: "EstimatorQNN | None" = None
        self._initialized = False

    async def initialize(self) -> None:
        """Build the circuit and load weights."""
        self._circuit = TrendCircuit(self._quantum_settings)
        self._circuit.build()
        self._qnn = self._circuit.build_qnn()

        if self._model_weights_path:
            self.load_weights(self._model_weights_path)

        self._initialized = True
        logger.info(
            "TrendDetector initialized: %d inputs, %d weights",
            self._qnn.num_inputs,
            self._qnn.num_weights,
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("TrendDetector not initialized. Call initialize() first.")

    async def predict(self, features: np.ndarray) -> TrendSignal:
        """Run inference on a single feature vector.

        Parameters
        ----------
        features:
            Encoded feature vector of shape ``(n_features,)``.

        Returns
        -------
        A ``TrendSignal`` with direction, confidence, and scenario.
        """
        self._ensure_initialized()

        # Reshape to batch of 1
        if features.ndim == 1:
            features = features.reshape(1, -1)

        weights = self._circuit.get_weights()
        raw_output = self._qnn.forward(features, weights)  # shape (1, 3)

        logits = raw_output[0]  # shape (3,)
        probs = softmax(logits)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        direction = _CLASS_MAP[class_idx]

        # Build features dict for logging
        feature_dict = {}
        for i, col in enumerate(FEATURE_COLS):
            if i < features.shape[1]:
                feature_dict[col] = float(features[0, i])

        signal = TrendSignal(
            direction=direction,
            confidence=confidence,
            scenario=Scenario.HOLD,  # default; strategy layer classifies scenario
            timestamp=datetime.now(timezone.utc),
            features=feature_dict,
            model_version=self._get_model_version(),
        )

        logger.debug(
            "Prediction: %s (conf=%.3f) from logits=%s",
            direction.value,
            confidence,
            logits.tolist(),
        )
        return signal

    async def predict_batch(self, features: np.ndarray) -> list[TrendSignal]:
        """Run inference on multiple feature vectors (for backtesting).

        Parameters
        ----------
        features:
            Array of shape ``(n_samples, n_features)``.
        """
        self._ensure_initialized()

        weights = self._circuit.get_weights()
        raw_output = self._qnn.forward(features, weights)  # shape (n_samples, 3)

        signals = []
        now = datetime.now(timezone.utc)
        model_ver = self._get_model_version()

        for i in range(raw_output.shape[0]):
            logits = raw_output[i]
            probs = softmax(logits)
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            direction = _CLASS_MAP[class_idx]

            feature_dict = {}
            for j, col in enumerate(FEATURE_COLS):
                if j < features.shape[1]:
                    feature_dict[col] = float(features[i, j])

            signals.append(
                TrendSignal(
                    direction=direction,
                    confidence=confidence,
                    scenario=Scenario.HOLD,
                    timestamp=now,
                    features=feature_dict,
                    model_version=model_ver,
                )
            )

        logger.info("Batch prediction: %d signals generated", len(signals))
        return signals

    def load_weights(self, path: str) -> None:
        """Load model weights from disk."""
        weights = np.load(path)
        self._circuit.set_weights(weights)
        logger.info("Loaded weights from %s (%d parameters)", path, len(weights))

    def save_weights(self, path: str) -> None:
        """Save current model weights to disk."""
        self._ensure_initialized()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._circuit.get_weights())
        logger.info("Saved weights to %s", path)

    def _get_model_version(self) -> str:
        """Derive model version from weights path or default."""
        if self._model_weights_path:
            return Path(self._model_weights_path).stem
        return "v0.1.0"
