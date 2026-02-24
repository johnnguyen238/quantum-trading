"""Quantum circuit retraining using classical optimizers (COBYLA / SPSA)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from src.data.models import ModelVersion
from src.quantum.circuits import TrendCircuit

if TYPE_CHECKING:
    from src.config.settings import QuantumSettings

logger = logging.getLogger(__name__)


class QuantumTrainer:
    """Train the VQC circuit parameters on labeled market data.

    Parameters
    ----------
    settings:
        Quantum configuration (optimizer, max_iterations, etc.).
    batch_size:
        Number of samples per optimizer iteration. ``0`` = use all samples.
        Mini-batching dramatically speeds up training for large datasets.
    """

    def __init__(
        self, settings: "QuantumSettings", *, batch_size: int = 500
    ) -> None:
        self._settings = settings
        self._batch_size = batch_size
        self._circuit: TrendCircuit | None = None
        self._best_weights: np.ndarray | None = None
        self._best_loss: float = float("inf")
        self._iteration = 0
        self._rng = np.random.default_rng(42)

        # Stored during training for _objective access
        self._train_features: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None

    async def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> ModelVersion:
        """Train the quantum circuit on the provided dataset.

        Parameters
        ----------
        features:
            Shape ``(n_samples, n_features)``, values in ``[0, 2*pi]``.
        labels:
            Shape ``(n_samples,)``, integer labels (0=SHORT, 1=NEUTRAL, 2=LONG).

        Returns
        -------
        ``ModelVersion`` with trained parameters and accuracy.
        """
        if len(features) == 0:
            raise ValueError("Cannot train on empty dataset")

        # Build circuit and QNN
        self._circuit = TrendCircuit(self._settings)
        self._circuit.build()
        self._circuit.build_qnn()

        # Store for _objective
        self._train_features = features
        self._train_labels = labels
        self._iteration = 0
        self._best_loss = float("inf")

        # Initial weights
        initial_weights = self._circuit.get_weights()
        self._best_weights = initial_weights.copy()

        logger.info(
            "Starting training: %d samples, %d weights, optimizer=%s, max_iter=%d",
            len(features),
            len(initial_weights),
            self._settings.optimizer,
            self._settings.max_iterations,
        )

        # Run classical optimizer
        result = minimize(
            self._objective,
            initial_weights,
            method=self._settings.optimizer,
            options={"maxiter": self._settings.max_iterations, "disp": False},
        )

        # Use the best weights found (not necessarily the last iteration)
        if result.fun < self._best_loss:
            self._best_weights = result.x.copy()
            self._best_loss = result.fun

        self._circuit.set_weights(self._best_weights)

        # Compute final accuracy
        accuracy = self._compute_accuracy(features, labels)

        logger.info(
            "Training complete: loss=%.4f, accuracy=%.4f after %d iterations",
            self._best_loss,
            accuracy,
            self._iteration,
        )

        # Build ModelVersion
        version_str = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        param_dict = {
            str(i): float(w) for i, w in enumerate(self._best_weights)
        }

        return ModelVersion(
            version=version_str,
            created_at=datetime.now(timezone.utc),
            parameters=param_dict,
            accuracy=accuracy,
            notes=f"optimizer={self._settings.optimizer}, iterations={self._iteration}",
        )

    def _objective(self, weights: np.ndarray) -> float:
        """Cost function for the optimizer.

        Computes cross-entropy loss on a mini-batch of training data.
        If ``batch_size`` is 0 or >= n_samples, uses the full dataset.
        """
        self._iteration += 1
        qnn = self._circuit.get_qnn()

        n_total = len(self._train_labels)
        if self._batch_size > 0 and self._batch_size < n_total:
            idx = self._rng.choice(n_total, self._batch_size, replace=False)
            features = self._train_features[idx]
            labels = self._train_labels[idx]
        else:
            features = self._train_features
            labels = self._train_labels

        # Forward pass
        raw_output = qnn.forward(features, weights)  # (batch, 3)

        # Softmax to get probabilities
        probs = softmax(raw_output, axis=1)

        # Cross-entropy loss (vectorised)
        n_samples = len(labels)
        eps = 1e-10
        label_ints = labels.astype(int)
        loss = -np.mean(np.log(probs[np.arange(n_samples), label_ints] + eps))

        # Track best
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_weights = weights.copy()

        if self._iteration % 10 == 0:
            logger.info("Iteration %d: loss=%.4f", self._iteration, loss)

        return float(loss)

    def _compute_accuracy(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute classification accuracy with current weights."""
        qnn = self._circuit.get_qnn()
        weights = self._circuit.get_weights()
        raw_output = qnn.forward(features, weights)
        predictions = np.argmax(raw_output, axis=1)
        return float(np.mean(predictions == labels))

    def get_trained_weights(self) -> np.ndarray:
        """Return the latest trained weights."""
        if self._best_weights is None:
            raise RuntimeError("No trained weights available. Call train() first.")
        return self._best_weights.copy()

    def save_model(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """Save trained weights and metadata to disk."""
        if self._best_weights is None:
            raise RuntimeError("No trained weights to save. Call train() first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights as .npy
        weights_path = save_path.with_suffix(".npy")
        np.save(str(weights_path), self._best_weights)

        # Save metadata as .json alongside
        meta = {
            "n_qubits": self._settings.n_qubits,
            "reps": self._settings.reps,
            "optimizer": self._settings.optimizer,
            "max_iterations": self._settings.max_iterations,
            "best_loss": self._best_loss,
            "iterations_run": self._iteration,
        }
        if metadata:
            meta.update(metadata)

        meta_path = save_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved model to %s (weights + metadata)", save_path)
