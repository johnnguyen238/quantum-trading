"""Walk-forward validation for model evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.ml.trainer import QuantumTrainer

if TYPE_CHECKING:
    from src.config.settings import QuantumSettings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Metrics from a single validation fold."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fold_index: int = 0


class WalkForwardEvaluator:
    """Evaluate model performance using walk-forward cross-validation.

    Splits data chronologically: train on past, validate on future.

    Parameters
    ----------
    settings:
        Quantum configuration.
    n_folds:
        Number of walk-forward folds.
    train_ratio:
        Fraction of each fold used for training.
    """

    def __init__(
        self,
        settings: "QuantumSettings",
        n_folds: int = 5,
        train_ratio: float = 0.8,
    ) -> None:
        self._settings = settings
        self._n_folds = n_folds
        self._train_ratio = train_ratio

    async def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> list[EvaluationResult]:
        """Run walk-forward evaluation across all folds.

        Each fold trains on a growing window of past data and validates
        on the immediately following segment.

        Returns
        -------
        List of ``EvaluationResult`` for each fold.
        """
        n_samples = len(features)
        if n_samples < self._n_folds + 1:
            logger.warning(
                "Not enough data (%d samples) for %d folds",
                n_samples,
                self._n_folds,
            )
            return []

        results: list[EvaluationResult] = []

        # Walk-forward: divide data into expanding train + fixed-size val windows
        fold_size = n_samples // (self._n_folds + 1)

        for fold_idx in range(self._n_folds):
            # Train: everything up to the fold boundary
            train_end = fold_size * (fold_idx + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples or train_end < 2:
                continue

            train_features = features[:train_end]
            train_labels = labels[:train_end]
            val_features = features[val_start:val_end]
            val_labels = labels[val_start:val_end]

            if len(val_features) == 0:
                continue

            logger.info(
                "Fold %d/%d: train=%d, val=%d",
                fold_idx + 1,
                self._n_folds,
                len(train_features),
                len(val_features),
            )

            # Train on this fold
            trainer = QuantumTrainer(self._settings)
            await trainer.train(train_features, train_labels)

            # Predict on validation set
            predictions = self._predict_with_trainer(trainer, val_features)

            # Compute metrics
            result = self._compute_metrics(val_labels, predictions, fold_idx)
            results.append(result)

            logger.info(
                "Fold %d: accuracy=%.4f, f1=%.4f",
                fold_idx + 1,
                result.accuracy,
                result.f1_score,
            )

        return results

    def _predict_with_trainer(
        self, trainer: QuantumTrainer, features: np.ndarray
    ) -> np.ndarray:
        """Get predictions from a trained model."""
        from scipy.special import softmax

        circuit = trainer._circuit
        qnn = circuit.get_qnn()
        weights = circuit.get_weights()
        raw_output = qnn.forward(features, weights)
        return np.argmax(softmax(raw_output, axis=1), axis=1)

    @staticmethod
    def _compute_metrics(
        labels: np.ndarray,
        predictions: np.ndarray,
        fold_index: int,
    ) -> EvaluationResult:
        """Compute accuracy, precision, recall, and F1 score.

        Uses macro-averaging across the 3 classes.
        """
        n_classes = 3
        accuracy = float(np.mean(predictions == labels))

        # Per-class precision, recall, F1
        precisions = []
        recalls = []
        f1s = []

        for cls in range(n_classes):
            tp = int(np.sum((predictions == cls) & (labels == cls)))
            fp = int(np.sum((predictions == cls) & (labels != cls)))
            fn = int(np.sum((predictions != cls) & (labels == cls)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return EvaluationResult(
            accuracy=accuracy,
            precision=float(np.mean(precisions)),
            recall=float(np.mean(recalls)),
            f1_score=float(np.mean(f1s)),
            fold_index=fold_index,
        )

    def aggregate(self, results: list[EvaluationResult]) -> dict[str, float]:
        """Compute mean metrics across all folds."""
        if not results:
            return {
                "mean_accuracy": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_f1_score": 0.0,
                "n_folds": 0,
            }

        return {
            "mean_accuracy": float(np.mean([r.accuracy for r in results])),
            "mean_precision": float(np.mean([r.precision for r in results])),
            "mean_recall": float(np.mean([r.recall for r in results])),
            "mean_f1_score": float(np.mean([r.f1_score for r in results])),
            "n_folds": len(results),
        }
