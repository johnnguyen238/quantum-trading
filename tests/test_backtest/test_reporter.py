"""Tests for the backtest reporter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.backtest.reporter import BacktestReport, BacktestReporter
from src.config.constants import OrderSide, OrderStatus, OrderType
from src.data.models import Trade


def _make_trade(
    side: str = "buy",
    filled_price: float = 42000.0,
    amount: float = 0.1,
    fee: float = 2.52,
    position_id: int | None = None,
    is_dca: bool = False,
) -> Trade:
    return Trade(
        symbol="BTC/USDT:USDT",
        side=OrderSide(side),
        order_type=OrderType.MARKET,
        amount=amount,
        price=filled_price,
        timestamp=datetime.now(timezone.utc),
        status=OrderStatus.FILLED,
        filled_amount=amount,
        filled_price=filled_price,
        fee=fee,
        position_id=position_id,
        is_dca=is_dca,
    )


def _make_round_trip(
    open_price: float,
    close_price: float,
    side: str = "buy",
    amount: float = 0.1,
    position_id: int = 1,
) -> list[Trade]:
    """Create a pair of trades representing open and close."""
    close_side = "sell" if side == "buy" else "buy"
    open_fee = open_price * amount * 0.001
    close_fee = close_price * amount * 0.001
    return [
        _make_trade(
            side=side,
            filled_price=open_price,
            amount=amount,
            fee=open_fee,
            position_id=position_id,
        ),
        _make_trade(
            side=close_side,
            filled_price=close_price,
            amount=amount,
            fee=close_fee,
            position_id=position_id,
        ),
    ]


class TestComputeMetrics:
    @pytest.fixture
    def reporter(self) -> BacktestReporter:
        return BacktestReporter()

    def test_empty_trades(self, reporter):
        report = reporter.compute_metrics([], 10000.0)
        assert report.total_trades == 0
        assert report.final_balance == 10000.0
        assert report.total_pnl == 0.0

    def test_single_winning_trade(self, reporter):
        # Buy at 42000, sell at 43000
        trades = _make_round_trip(42000.0, 43000.0, position_id=1)
        report = reporter.compute_metrics(trades, 10000.0)
        assert report.total_trades == 1
        assert report.winning_trades == 1
        assert report.losing_trades == 0
        assert report.win_rate == 1.0
        assert report.total_pnl > 0

    def test_single_losing_trade(self, reporter):
        # Buy at 42000, sell at 41000
        trades = _make_round_trip(42000.0, 41000.0, position_id=1)
        report = reporter.compute_metrics(trades, 10000.0)
        assert report.total_trades == 1
        assert report.winning_trades == 0
        assert report.losing_trades == 1
        assert report.win_rate == 0.0
        assert report.total_pnl < 0

    def test_mixed_trades(self, reporter):
        trades = (
            _make_round_trip(42000.0, 43000.0, position_id=1)
            + _make_round_trip(42000.0, 41000.0, position_id=2)
        )
        report = reporter.compute_metrics(trades, 10000.0)
        assert report.total_trades == 2
        assert report.winning_trades == 1
        assert report.losing_trades == 1
        assert report.win_rate == pytest.approx(0.5)

    def test_profit_factor(self, reporter):
        # Win: +100 (before fees), Loss: -100 (before fees)
        trades = (
            _make_round_trip(42000.0, 43000.0, position_id=1)
            + _make_round_trip(42000.0, 41000.0, position_id=2)
        )
        report = reporter.compute_metrics(trades, 10000.0)
        # Profit factor should be close to 1 for symmetric wins/losses
        # (slight deviation due to different fee amounts)
        assert report.profit_factor > 0

    def test_final_balance(self, reporter):
        trades = _make_round_trip(42000.0, 43000.0, position_id=1)
        report = reporter.compute_metrics(trades, 10000.0)
        assert report.final_balance == pytest.approx(
            10000.0 + report.total_pnl
        )

    def test_all_winning(self, reporter):
        trades = (
            _make_round_trip(42000.0, 43000.0, position_id=1)
            + _make_round_trip(41000.0, 42000.0, position_id=2)
        )
        report = reporter.compute_metrics(trades, 10000.0)
        assert report.profit_factor == float("inf")

    def test_short_position_pnl(self, reporter):
        # Short: sell at 43000, buy at 42000 → profit
        trades = _make_round_trip(43000.0, 42000.0, side="sell", position_id=1)
        report = reporter.compute_metrics(trades, 10000.0)
        assert report.total_pnl > 0
        assert report.winning_trades == 1


class TestMaxDrawdown:
    def test_no_drawdown(self):
        pnls = [100.0, 100.0, 100.0]
        dd = BacktestReporter._calculate_max_drawdown(pnls, 10000.0)
        assert dd == 0.0

    def test_simple_drawdown(self):
        pnls = [100.0, -200.0, 50.0]
        dd = BacktestReporter._calculate_max_drawdown(pnls, 10000.0)
        # Peak = 10100, trough after loss = 9900, dd = 200/10100
        assert dd == pytest.approx(200.0 / 10100.0, rel=1e-4)

    def test_empty_pnls(self):
        dd = BacktestReporter._calculate_max_drawdown([], 10000.0)
        assert dd == 0.0

    def test_all_losses(self):
        pnls = [-100.0, -100.0, -100.0]
        dd = BacktestReporter._calculate_max_drawdown(pnls, 10000.0)
        # Peak stays at 10000, final = 9700, dd = 300/10000
        assert dd == pytest.approx(300.0 / 10000.0)


class TestSharpeRatio:
    def test_zero_variance(self):
        pnls = [100.0, 100.0, 100.0]
        sharpe = BacktestReporter._calculate_sharpe(pnls, 10000.0)
        assert sharpe == 0.0  # zero std → 0

    def test_positive_sharpe(self):
        pnls = [100.0, 200.0, 150.0, 300.0]
        sharpe = BacktestReporter._calculate_sharpe(pnls, 10000.0)
        assert sharpe > 0

    def test_single_trade(self):
        sharpe = BacktestReporter._calculate_sharpe([100.0], 10000.0)
        assert sharpe == 0.0  # need at least 2

    def test_empty(self):
        sharpe = BacktestReporter._calculate_sharpe([], 10000.0)
        assert sharpe == 0.0


class TestExportCsv:
    def test_export_creates_file(self, tmp_path):
        reporter = BacktestReporter()
        trades = _make_round_trip(42000.0, 43000.0, position_id=1)
        report = BacktestReport(trades=trades)
        path = str(tmp_path / "trades.csv")

        reporter.export_csv(report, path)
        assert Path(path).exists()

    def test_export_has_headers(self, tmp_path):
        reporter = BacktestReporter()
        trades = _make_round_trip(42000.0, 43000.0, position_id=1)
        report = BacktestReport(trades=trades)
        path = str(tmp_path / "trades.csv")

        reporter.export_csv(report, path)
        with open(path) as f:
            header = f.readline()
        assert "symbol" in header
        assert "filled_price" in header

    def test_export_correct_row_count(self, tmp_path):
        reporter = BacktestReporter()
        trades = _make_round_trip(42000.0, 43000.0, position_id=1)
        report = BacktestReport(trades=trades)
        path = str(tmp_path / "trades.csv")

        reporter.export_csv(report, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 trades

    def test_export_empty_trades(self, tmp_path):
        reporter = BacktestReporter()
        report = BacktestReport(trades=[])
        path = str(tmp_path / "trades.csv")

        reporter.export_csv(report, path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1  # header only


class TestPrintSummary:
    def test_print_does_not_raise(self, capsys):
        reporter = BacktestReporter()
        report = BacktestReport(
            symbol="BTC/USDT:USDT",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_balance=10000.0,
            final_balance=11000.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            total_pnl=1000.0,
        )
        reporter.print_summary(report)  # Should not raise
