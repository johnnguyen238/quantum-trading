"""Integration tests for the CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from src.main import app

runner = CliRunner()


class TestMigrateCommand:
    def test_migrate_runs(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with patch("src.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.database.path = db_path
            mock_load.return_value = mock_settings

            with patch(
                "src.data.migrations.run_migrations", new_callable=AsyncMock
            ) as mock_mig:
                result = runner.invoke(app, ["migrate", "--profile", "default"])
                assert result.exit_code == 0
                assert "Migrations complete" in result.output
                mock_mig.assert_called_once_with(db_path)


class TestRunCommand:
    def test_run_starts_engine(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.trading.symbols = ["BTC/USDT:USDT"]
            mock_settings.exchange.testnet = True
            mock_settings.trading.timeframe = "15m"
            mock_load.return_value = mock_settings

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(
                    app, ["run", "--profile", "testnet"]
                )
                assert result.exit_code == 0
                assert "Starting trading bot" in result.output
                mock_run.assert_called_once()

    def test_run_overrides_symbols(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.trading.symbols = ["BTC/USDT:USDT"]
            mock_settings.exchange.testnet = True
            mock_settings.trading.timeframe = "15m"
            mock_load.return_value = mock_settings

            with patch("asyncio.run"):
                result = runner.invoke(
                    app,
                    ["run", "--symbols", "ETH/USDT:USDT"],
                )
                assert result.exit_code == 0
                assert mock_settings.trading.symbols == ["ETH/USDT:USDT"]


class TestBacktestCommand:
    def test_backtest_invokes_async(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_load.return_value = MagicMock()

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(
                    app,
                    [
                        "backtest",
                        "--symbol", "BTC/USDT:USDT",
                        "--start", "2024-01-01",
                        "--end", "2024-06-30",
                    ],
                )
                assert result.exit_code == 0
                assert "Running backtest" in result.output
                mock_run.assert_called_once()

    def test_backtest_with_export(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_load.return_value = MagicMock()

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(
                    app,
                    [
                        "backtest",
                        "--export", "output/trades.csv",
                    ],
                )
                assert result.exit_code == 0
                mock_run.assert_called_once()


class TestTrainCommand:
    def test_train_invokes_async(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.quantum.n_qubits = 4
            mock_settings.quantum.reps = 2
            mock_settings.quantum.optimizer = "COBYLA"
            mock_load.return_value = mock_settings

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(
                    app,
                    [
                        "train",
                        "--symbol", "BTC/USDT:USDT",
                        "--start", "2024-01-01",
                        "--end", "2024-06-30",
                    ],
                )
                assert result.exit_code == 0
                assert "Training quantum model" in result.output
                mock_run.assert_called_once()

    def test_train_with_output(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_settings = MagicMock()
            mock_settings.quantum.n_qubits = 4
            mock_settings.quantum.reps = 2
            mock_settings.quantum.optimizer = "COBYLA"
            mock_load.return_value = mock_settings

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(
                    app,
                    ["train", "--output", "data/models/custom"],
                )
                assert result.exit_code == 0
                mock_run.assert_called_once()


class TestEvaluateCommand:
    def test_evaluate_invokes_async(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_load.return_value = MagicMock()

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(
                    app,
                    [
                        "evaluate",
                        "--symbol", "BTC/USDT:USDT",
                        "--start", "2024-01-01",
                        "--end", "2024-06-30",
                    ],
                )
                assert result.exit_code == 0
                assert "Evaluating model" in result.output
                mock_run.assert_called_once()


class TestStatusCommand:
    def test_status_invokes_async(self):
        with patch("src.config.settings.load_settings") as mock_load:
            mock_load.return_value = MagicMock()

            with patch("asyncio.run") as mock_run:
                result = runner.invoke(app, ["status"])
                assert result.exit_code == 0
                mock_run.assert_called_once()
