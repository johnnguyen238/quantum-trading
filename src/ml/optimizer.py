"""Hyperparameter tuning for the quantum circuit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.settings import QuantumSettings
from src.ml.trainer import QuantumTrainer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Search space boundaries
_QUBIT_OPTIONS = [3, 4, 5, 6]
_REPS_OPTIONS = [1, 2, 3]
_OPTIMIZER_OPTIONS = ["COBYLA", "Nelder-Mead", "Powell"]


class HyperparameterOptimizer:
    """Search for optimal quantum circuit hyperparameters.

    Tunes n_qubits, reps, optimizer choice, and learning rate.

    Parameters
    ----------
    base_settings:
        Base quantum configuration to vary from.
    """

    def __init__(self, base_settings: QuantumSettings) -> None:
        self._base_settings = base_settings

    async def optimize(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_trials: int = 20,
    ) -> dict[str, Any]:
        """Run hyperparameter search and return the best configuration.

        Parameters
        ----------
        features:
            Training features.
        labels:
            Training labels.
        n_trials:
            Number of configurations to try.

        Returns
        -------
        Dict with the best hyperparameters and their validation score.
        """
        if len(features) == 0:
            raise ValueError("Cannot optimize on empty dataset")

        # Split into train/validation (80/20 chronological)
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]

        if len(train_features) == 0 or len(val_features) == 0:
            raise ValueError("Not enough data for train/validation split")

        best_config: dict[str, Any] = {}
        best_score: float = -1.0
        rng = np.random.default_rng()

        logger.info("Starting hyperparameter search: %d trials", n_trials)

        for trial in range(n_trials):
            # Sample hyperparameters
            n_qubits = int(rng.choice(_QUBIT_OPTIONS))
            reps = int(rng.choice(_REPS_OPTIONS))
            optimizer_name = str(rng.choice(_OPTIMIZER_OPTIONS))
            max_iterations = int(rng.integers(50, 200))

            # Feature dimension must match n_qubits — truncate or pad features
            trial_train = self._adjust_features(train_features, n_qubits)
            trial_val = self._adjust_features(val_features, n_qubits)

            # Create trial settings
            trial_settings = QuantumSettings(
                n_qubits=n_qubits,
                reps=reps,
                optimizer=optimizer_name,
                max_iterations=max_iterations,
                confidence_threshold=self._base_settings.confidence_threshold,
            )

            try:
                trainer = QuantumTrainer(trial_settings)
                model_version = await trainer.train(trial_train, train_labels)

                # Evaluate on validation set
                accuracy = self._evaluate_accuracy(trainer, trial_val, val_labels)

                logger.info(
                    "Trial %d/%d: qubits=%d, reps=%d, opt=%s, iter=%d → acc=%.4f",
                    trial + 1,
                    n_trials,
                    n_qubits,
                    reps,
                    optimizer_name,
                    max_iterations,
                    accuracy,
                )

                if accuracy > best_score:
                    best_score = accuracy
                    best_config = {
                        "n_qubits": n_qubits,
                        "reps": reps,
                        "optimizer": optimizer_name,
                        "max_iterations": max_iterations,
                        "validation_accuracy": accuracy,
                        "training_accuracy": model_version.accuracy,
                    }

            except Exception:
                logger.exception(
                    "Trial %d/%d failed (qubits=%d, reps=%d, opt=%s)",
                    trial + 1,
                    n_trials,
                    n_qubits,
                    reps,
                    optimizer_name,
                )
                continue

        if not best_config:
            raise RuntimeError("All hyperparameter trials failed")

        logger.info(
            "Best config: qubits=%d, reps=%d, opt=%s → acc=%.4f",
            best_config["n_qubits"],
            best_config["reps"],
            best_config["optimizer"],
            best_config["validation_accuracy"],
        )
        return best_config

    @staticmethod
    def _adjust_features(
        features: np.ndarray, n_qubits: int
    ) -> np.ndarray:
        """Adjust feature dimension to match n_qubits.

        Truncates if features have more columns, pads with zeros if fewer.
        """
        n_features = features.shape[1]
        if n_features == n_qubits:
            return features
        elif n_features > n_qubits:
            return features[:, :n_qubits]
        else:
            padding = np.zeros((features.shape[0], n_qubits - n_features))
            return np.hstack([features, padding])

    @staticmethod
    def _evaluate_accuracy(
        trainer: QuantumTrainer,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute accuracy of the trained model on a feature set."""
        from scipy.special import softmax

        circuit = trainer._circuit
        qnn = circuit.get_qnn()
        weights = circuit.get_weights()
        raw_output = qnn.forward(features, weights)
        predictions = np.argmax(softmax(raw_output, axis=1), axis=1)
        return float(np.mean(predictions == labels))
