"""Tests for the ML walk-forward evaluator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config.settings import QuantumSettings
from src.ml.evaluator import EvaluationResult, WalkForwardEvaluator


def _mock_trainer():
    """Create a mock QuantumTrainer that 'trains' instantly."""
    from datetime import datetime, timezone

    from src.data.models import ModelVersion

    trainer = MagicMock()
    trainer.train = AsyncMock(
        return_value=ModelVersion(
            version="v_test",
            created_at=datetime.now(timezone.utc),
            accuracy=0.6,
        )
    )

    # Mock circuit for predictions
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


@pytest.fixture
def settings() -> QuantumSettings:
    return QuantumSettings(
        n_qubits=4, reps=2, optimizer="COBYLA", max_iterations=3
    )


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    features = rng.uniform(0, 2 * np.pi, size=(60, 4))
    labels = rng.integers(0, 3, size=60)
    return features, labels


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_returns_results_per_fold(self, settings, sample_data):
        features, labels = sample_data
        evaluator = WalkForwardEvaluator(settings, n_folds=3, train_ratio=0.8)

        with patch("src.ml.evaluator.QuantumTrainer") as MockTrainer:
            MockTrainer.return_value = _mock_trainer()
            results = await evaluator.evaluate(features, labels)

        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.asyncio
    async def test_fold_indices_are_sequential(self, settings, sample_data):
        features, labels = sample_data
        evaluator = WalkForwardEvaluator(settings, n_folds=3, train_ratio=0.8)

        with patch("src.ml.evaluator.QuantumTrainer") as MockTrainer:
            MockTrainer.return_value = _mock_trainer()
            results = await evaluator.evaluate(features, labels)

        for i, r in enumerate(results):
            assert r.fold_index == i

    @pytest.mark.asyncio
    async def test_metrics_in_valid_range(self, settings, sample_data):
        features, labels = sample_data
        evaluator = WalkForwardEvaluator(settings, n_folds=3, train_ratio=0.8)

        with patch("src.ml.evaluator.QuantumTrainer") as MockTrainer:
            MockTrainer.return_value = _mock_trainer()
            results = await evaluator.evaluate(features, labels)

        for r in results:
            assert 0.0 <= r.accuracy <= 1.0
            assert 0.0 <= r.precision <= 1.0
            assert 0.0 <= r.recall <= 1.0
            assert 0.0 <= r.f1_score <= 1.0

    @pytest.mark.asyncio
    async def test_too_few_samples(self, settings):
        features = np.random.uniform(0, 2 * np.pi, size=(2, 4))
        labels = np.array([0, 1])
        evaluator = WalkForwardEvaluator(settings, n_folds=5, train_ratio=0.8)

        results = await evaluator.evaluate(features, labels)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_single_fold(self, settings, sample_data):
        features, labels = sample_data
        evaluator = WalkForwardEvaluator(settings, n_folds=1, train_ratio=0.8)

        with patch("src.ml.evaluator.QuantumTrainer") as MockTrainer:
            MockTrainer.return_value = _mock_trainer()
            results = await evaluator.evaluate(features, labels)

        assert len(results) == 1


class TestComputeMetrics:
    def test_perfect_predictions(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        labels = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 2, 0, 1, 2])

        result = evaluator._compute_metrics(labels, predictions, fold_index=0)
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0

    def test_all_wrong_predictions(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        labels = np.array([0, 0, 0, 0])
        predictions = np.array([1, 1, 1, 1])

        result = evaluator._compute_metrics(labels, predictions, fold_index=0)
        assert result.accuracy == 0.0

    def test_partial_accuracy(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        labels = np.array([0, 1, 2, 0])
        predictions = np.array([0, 1, 0, 0])

        result = evaluator._compute_metrics(labels, predictions, fold_index=0)
        assert result.accuracy == pytest.approx(0.75)

    def test_fold_index_stored(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        labels = np.array([0, 1])
        predictions = np.array([0, 1])

        result = evaluator._compute_metrics(labels, predictions, fold_index=5)
        assert result.fold_index == 5


class TestAggregate:
    def test_aggregates_correctly(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        results = [
            EvaluationResult(accuracy=0.8, precision=0.7, recall=0.6, f1_score=0.65, fold_index=0),
            EvaluationResult(accuracy=0.6, precision=0.5, recall=0.4, f1_score=0.45, fold_index=1),
        ]
        agg = evaluator.aggregate(results)

        assert agg["mean_accuracy"] == pytest.approx(0.7)
        assert agg["mean_precision"] == pytest.approx(0.6)
        assert agg["mean_recall"] == pytest.approx(0.5)
        assert agg["mean_f1_score"] == pytest.approx(0.55)
        assert agg["n_folds"] == 2

    def test_empty_results(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        agg = evaluator.aggregate([])
        assert agg["mean_accuracy"] == 0.0
        assert agg["n_folds"] == 0

    def test_single_result(self, settings):
        evaluator = WalkForwardEvaluator(settings)
        results = [
            EvaluationResult(accuracy=0.9, precision=0.85, recall=0.8, f1_score=0.82, fold_index=0),
        ]
        agg = evaluator.aggregate(results)
        assert agg["mean_accuracy"] == pytest.approx(0.9)
        assert agg["n_folds"] == 1
