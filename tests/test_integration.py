"""End-to-end integration tests for the main pipelines."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.settings import QuantumSettings, Settings, StrategySettings


def _make_df(n: int = 100) -> pd.DataFrame:
    """Create realistic OHLCV data."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": [42000.0 + i * 10 for i in range(n)],
            "high": [42050.0 + i * 10 for i in range(n)],
            "low": [41950.0 + i * 10 for i in range(n)],
            "close": [42020.0 + i * 10 for i in range(n)],
            "volume": [100.0 + i for i in range(n)],
        }
    )


def _make_settings() -> Settings:
    return Settings()


class TestBacktestPipeline:
    """Test the full backtest pipeline with mocked detector."""

    @pytest.mark.asyncio
    async def test_full_backtest_pipeline(self):
        """Run a complete backtest from data load to report."""
        from src.backtest.engine import BacktestEngine
        from src.backtest.reporter import BacktestReport
        from src.config.constants import Scenario, TrendDirection
        from src.quantum.signal import TrendSignal

        settings = _make_settings()

        # Mock data loader
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=_make_df(100))

        # Mock detector that alternates LONG/SHORT
        signals = []
        for i in range(100):
            direction = TrendDirection.LONG if i % 20 < 10 else TrendDirection.SHORT
            signals.append(
                TrendSignal(
                    direction=direction,
                    confidence=0.8,
                    scenario=Scenario.HOLD,
                    timestamp=datetime.now(timezone.utc),
                )
            )

        call_count = [0]

        async def mock_predict(features):
            idx = min(call_count[0], len(signals) - 1)
            call_count[0] += 1
            return signals[idx]

        mock_detector = MagicMock()
        mock_detector.predict = AsyncMock(side_effect=mock_predict)

        engine = BacktestEngine(settings, mock_loader, mock_detector)
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")

        assert isinstance(report, BacktestReport)
        assert report.symbol == "BTC/USDT:USDT"
        assert report.initial_balance == settings.backtest.initial_balance
        # The engine should have processed some trades
        assert mock_detector.predict.call_count > 0

    @pytest.mark.asyncio
    async def test_backtest_with_no_data(self):
        """Backtest should return empty report when no data is available."""
        from src.backtest.engine import BacktestEngine

        settings = _make_settings()
        mock_loader = MagicMock()
        mock_loader.load = AsyncMock(return_value=pd.DataFrame())
        mock_detector = MagicMock()

        engine = BacktestEngine(settings, mock_loader, mock_detector)
        report = await engine.run("BTC/USDT:USDT", "2024-01-01", "2024-12-31")

        assert report.total_trades == 0
        mock_detector.predict.assert_not_called()


class TestTrainPipeline:
    """Test the training pipeline with mocked circuit."""

    @pytest.mark.asyncio
    async def test_feature_engineer_to_trainer(self):
        """Test that FeatureEngineer output is compatible with QuantumTrainer."""
        from src.ml.feature_engineering import FeatureEngineer

        settings = StrategySettings()
        engineer = FeatureEngineer(settings, forward_period=5, threshold=0.005)
        df = _make_df(100)

        features, labels = engineer.create_dataset(df)
        assert len(features) > 0
        assert features.shape[1] == 4  # matches n_qubits=4

        # Verify features are in valid range for quantum circuit
        assert np.all(features >= 0)
        assert np.all(features <= 2 * np.pi + 0.01)
        assert set(labels).issubset({0, 1, 2})

    @pytest.mark.asyncio
    async def test_trainer_with_engineered_features(self):
        """Full pipeline: FeatureEngineer → QuantumTrainer."""
        from src.ml.feature_engineering import FeatureEngineer
        from src.ml.trainer import QuantumTrainer

        settings = StrategySettings()
        quantum_settings = QuantumSettings(
            n_qubits=4, reps=2, optimizer="COBYLA", max_iterations=3
        )

        # Create dataset
        engineer = FeatureEngineer(settings, forward_period=5, threshold=0.005)
        features, labels = engineer.create_dataset(_make_df(100))

        # Mock circuit for training
        mock_circuit = MagicMock()
        weights = np.random.uniform(-np.pi, np.pi, size=8)
        mock_circuit.get_weights.return_value = weights.copy()
        mock_circuit.num_parameters = 8
        mock_circuit.build.return_value = MagicMock()

        mock_qnn = MagicMock()
        mock_qnn.num_inputs = 4
        mock_qnn.num_weights = 8
        mock_qnn.output_shape = (3,)

        def forward_fn(feats, w):
            n = feats.shape[0]
            rng = np.random.RandomState(42)
            return rng.randn(n, 3)

        mock_qnn.forward = MagicMock(side_effect=forward_fn)
        mock_circuit.build_qnn.return_value = mock_qnn
        mock_circuit.get_qnn.return_value = mock_qnn

        def set_weights_fn(w):
            mock_circuit.get_weights.return_value = w.copy()

        mock_circuit.set_weights = MagicMock(side_effect=set_weights_fn)

        with patch("src.ml.trainer.TrendCircuit") as MockCircuit:
            MockCircuit.return_value = mock_circuit
            trainer = QuantumTrainer(quantum_settings)
            model_version = await trainer.train(features, labels)

        assert model_version.version.startswith("v")
        assert 0.0 <= model_version.accuracy <= 1.0
        assert len(model_version.parameters) > 0


class TestEvaluationPipeline:
    """Test the evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_feature_engineer_to_evaluator(self):
        """Test FeatureEngineer → WalkForwardEvaluator pipeline."""
        from src.data.models import ModelVersion
        from src.ml.evaluator import WalkForwardEvaluator
        from src.ml.feature_engineering import FeatureEngineer

        settings = StrategySettings()
        quantum_settings = QuantumSettings(
            n_qubits=4, reps=2, optimizer="COBYLA", max_iterations=3
        )

        # Create dataset
        engineer = FeatureEngineer(settings, forward_period=5, threshold=0.005)
        features, labels = engineer.create_dataset(_make_df(100))

        # Mock trainer used inside evaluator
        def mock_trainer_factory():
            trainer = MagicMock()
            trainer.train = AsyncMock(
                return_value=ModelVersion(
                    version="v_test",
                    created_at=datetime.now(timezone.utc),
                    accuracy=0.6,
                )
            )
            mock_circuit = MagicMock()
            mock_circuit.get_weights.return_value = np.random.uniform(
                -np.pi, np.pi, size=8
            )
            mock_qnn = MagicMock()

            def fwd(f, w):
                return np.random.RandomState(42).randn(f.shape[0], 3)

            mock_qnn.forward = MagicMock(side_effect=fwd)
            mock_circuit.get_qnn.return_value = mock_qnn
            trainer._circuit = mock_circuit
            return trainer

        evaluator = WalkForwardEvaluator(quantum_settings, n_folds=2)

        with patch("src.ml.evaluator.QuantumTrainer") as MockTrainer:
            MockTrainer.return_value = mock_trainer_factory()
            results = await evaluator.evaluate(features, labels)

        assert len(results) == 2
        agg = evaluator.aggregate(results)
        assert 0.0 <= agg["mean_accuracy"] <= 1.0


class TestStatusPipeline:
    """Test the status display pipeline."""

    @pytest.mark.asyncio
    async def test_status_with_mocked_deps(self, tmp_path):
        """Test _run_status with all external deps mocked."""
        from src.main import _run_status

        settings = _make_settings()
        settings.database.path = str(tmp_path / "test.db")
        settings.exchange.api_key = ""

        with (
            patch("src.data.migrations.run_migrations", new_callable=AsyncMock),
            patch("src.data.database.Database") as MockDB,
            patch("src.data.repository.Repository") as MockRepo,
        ):
            mock_db = MagicMock()
            mock_db.connect = AsyncMock()
            mock_db.disconnect = AsyncMock()
            MockDB.return_value = mock_db

            mock_repo = MagicMock()
            mock_repo.get_latest_model_version = AsyncMock(return_value=None)
            MockRepo.return_value = mock_repo

            await _run_status(settings)

            mock_db.connect.assert_called_once()
            mock_db.disconnect.assert_called_once()


class TestTrainCliPipeline:
    """Test the train async pipeline."""

    @pytest.mark.asyncio
    async def test_train_insufficient_data(self, tmp_path):
        """Train should handle insufficient data gracefully."""
        from src.main import _run_train

        settings = _make_settings()
        settings.database.path = str(tmp_path / "test.db")
        settings.exchange.api_key = ""

        with (
            patch("src.data.migrations.run_migrations", new_callable=AsyncMock),
            patch("src.data.database.Database") as MockDB,
            patch("src.data.repository.Repository") as MockRepo,
            patch("src.backtest.data_loader.DataLoader") as MockLoader,
        ):
            mock_db = MagicMock()
            mock_db.connect = AsyncMock()
            mock_db.disconnect = AsyncMock()
            MockDB.return_value = mock_db
            MockRepo.return_value = MagicMock()

            mock_loader = MagicMock()
            mock_loader.load = AsyncMock(return_value=pd.DataFrame())
            MockLoader.return_value = mock_loader

            # Should not raise, just print error
            await _run_train(settings, "BTC/USDT:USDT", "2024-01-01", "2024-12-31", None)
            mock_db.disconnect.assert_called_once()


class TestBacktestCliPipeline:
    """Test the backtest async pipeline."""

    @pytest.mark.asyncio
    async def test_backtest_pipeline_with_mocked_deps(self, tmp_path):
        """Test _run_backtest with all external deps mocked."""
        from src.backtest.reporter import BacktestReport
        from src.main import _run_backtest

        settings = _make_settings()
        settings.database.path = str(tmp_path / "test.db")
        settings.exchange.api_key = ""

        mock_report = BacktestReport(
            symbol="BTC/USDT:USDT",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_balance=10000.0,
            final_balance=10500.0,
            total_trades=5,
            total_pnl=500.0,
        )

        with (
            patch("src.data.migrations.run_migrations", new_callable=AsyncMock),
            patch("src.data.database.Database") as MockDB,
            patch("src.data.repository.Repository"),
            patch("src.backtest.data_loader.DataLoader"),
            patch("src.quantum.trend_detector.TrendDetector") as MockDetector,
            patch("src.backtest.engine.BacktestEngine") as MockEngine,
            patch("src.backtest.reporter.BacktestReporter") as MockReporter,
        ):
            mock_db = MagicMock()
            mock_db.connect = AsyncMock()
            mock_db.disconnect = AsyncMock()
            MockDB.return_value = mock_db

            mock_detector = MagicMock()
            mock_detector.initialize = AsyncMock()
            MockDetector.return_value = mock_detector

            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_report)
            MockEngine.return_value = mock_engine

            mock_reporter = MagicMock()
            MockReporter.return_value = mock_reporter

            await _run_backtest(
                settings, "BTC/USDT:USDT", "2024-01-01", "2024-12-31", None
            )

            mock_engine.run.assert_called_once_with(
                "BTC/USDT:USDT", "2024-01-01", "2024-12-31"
            )
            mock_reporter.print_summary.assert_called_once_with(mock_report)
            mock_db.disconnect.assert_called_once()
