"""Performance metrics and backtest reporting."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from src.data.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    """Summary of a backtest run."""

    symbol: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    trades: list["Trade"] = field(default_factory=list)


class BacktestReporter:
    """Compute and format backtest performance metrics."""

    def compute_metrics(
        self,
        trades: list["Trade"],
        initial_balance: float,
    ) -> BacktestReport:
        """Calculate performance metrics from a list of completed trades.

        Expects trades to be paired: each position open has a corresponding
        close. We compute PnL from ``filled_price`` differences grouped by
        ``position_id``, or simply from the ``fee`` field for simulated trades.

        Metrics:
        - **Win rate**: fraction of trades with positive realized PnL
        - **Profit factor**: gross profit / gross loss
        - **Sharpe ratio**: mean(returns) / std(returns) * sqrt(252)
        - **Max drawdown**: largest peak-to-trough balance decline
        """
        report = BacktestReport(
            initial_balance=initial_balance,
            trades=trades,
        )

        if not trades:
            report.final_balance = initial_balance
            return report

        # Compute per-trade PnL from each trade's realized data
        pnl_list = self._compute_trade_pnls(trades)

        report.total_trades = len(pnl_list)

        if not pnl_list:
            report.final_balance = initial_balance
            return report

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        report.winning_trades = len(wins)
        report.losing_trades = len(losses)
        report.total_pnl = sum(pnl_list)
        report.final_balance = initial_balance + report.total_pnl

        # Win rate
        report.win_rate = (
            report.winning_trades / report.total_trades
            if report.total_trades > 0
            else 0.0
        )

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        report.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Sharpe ratio (annualized, assuming daily-scale returns)
        report.sharpe_ratio = self._calculate_sharpe(pnl_list, initial_balance)

        # Max drawdown
        report.max_drawdown = self._calculate_max_drawdown(
            pnl_list, initial_balance
        )

        return report

    def print_summary(self, report: BacktestReport) -> None:
        """Print a formatted summary table to the console (Rich)."""
        console = Console()

        table = Table(title="Backtest Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Symbol", report.symbol)
        table.add_row("Period", f"{report.start_date} → {report.end_date}")
        table.add_row("Initial Balance", f"${report.initial_balance:,.2f}")
        table.add_row("Final Balance", f"${report.final_balance:,.2f}")
        table.add_row("Total PnL", f"${report.total_pnl:,.2f}")
        table.add_row(
            "Return",
            f"{(report.total_pnl / report.initial_balance * 100):.2f}%"
            if report.initial_balance > 0
            else "N/A",
        )
        table.add_row("Total Trades", str(report.total_trades))
        table.add_row(
            "Win / Loss",
            f"{report.winning_trades} / {report.losing_trades}",
        )
        table.add_row("Win Rate", f"{report.win_rate:.1%}")
        table.add_row(
            "Profit Factor",
            f"{report.profit_factor:.2f}"
            if report.profit_factor != float("inf")
            else "∞",
        )
        table.add_row("Sharpe Ratio", f"{report.sharpe_ratio:.2f}")
        table.add_row("Max Drawdown", f"{report.max_drawdown:.2%}")

        console.print(table)

    def export_csv(self, report: BacktestReport, path: str) -> None:
        """Export the trade log to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "symbol",
            "side",
            "order_type",
            "amount",
            "price",
            "filled_amount",
            "filled_price",
            "fee",
            "status",
            "is_dca",
            "dca_layer",
            "position_id",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in report.trades:
                writer.writerow(
                    {
                        "timestamp": trade.timestamp.isoformat()
                        if trade.timestamp
                        else "",
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "order_type": trade.order_type.value,
                        "amount": trade.amount,
                        "price": trade.price,
                        "filled_amount": trade.filled_amount,
                        "filled_price": trade.filled_price,
                        "fee": trade.fee,
                        "status": trade.status.value,
                        "is_dca": trade.is_dca,
                        "dca_layer": trade.dca_layer,
                        "position_id": trade.position_id,
                    }
                )

        logger.info("Exported %d trades to %s", len(report.trades), path)

    @staticmethod
    def _compute_trade_pnls(trades: list["Trade"]) -> list[float]:
        """Extract per-round-trip PnL from the trade list.

        Groups trades by position_id. For each position, sums fees from
        all trades and computes PnL from the difference between opening
        and closing fill prices.

        If position_id grouping is not available, falls back to pairing
        sequential buy/sell trades.
        """
        # Group by position_id
        positions: dict[int | None, list["Trade"]] = {}
        for trade in trades:
            positions.setdefault(trade.position_id, []).append(trade)

        pnl_list: list[float] = []

        for pos_id, pos_trades in positions.items():
            if pos_id is None and len(pos_trades) == 1:
                # Single unlinked trade — use fee as cost
                continue

            total_fees = sum(t.fee for t in pos_trades)

            # Find closing trades (anything after the opener that isn't DCA)
            closes = [
                t
                for t in pos_trades
                if t != pos_trades[0] and not t.is_dca
            ]

            if not closes:
                continue

            # Simple PnL: use first open and last close
            open_trade = pos_trades[0]
            close_trade = pos_trades[-1]

            if open_trade.side.value == "buy":
                raw_pnl = (
                    (close_trade.filled_price - open_trade.filled_price)
                    * open_trade.filled_amount
                )
            else:
                raw_pnl = (
                    (open_trade.filled_price - close_trade.filled_price)
                    * open_trade.filled_amount
                )

            pnl_list.append(raw_pnl - total_fees)

        return pnl_list

    @staticmethod
    def _calculate_sharpe(
        pnl_list: list[float],
        initial_balance: float,
    ) -> float:
        """Compute annualized Sharpe ratio from trade PnLs.

        Uses ``sqrt(252)`` annualization factor.
        """
        if len(pnl_list) < 2 or initial_balance <= 0:
            return 0.0

        # Convert PnLs to returns
        returns = [p / initial_balance for p in pnl_list]
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0

        if std_ret < 1e-10:
            return 0.0

        return (mean_ret / std_ret) * math.sqrt(252)

    @staticmethod
    def _calculate_max_drawdown(
        pnl_list: list[float],
        initial_balance: float,
    ) -> float:
        """Compute maximum drawdown as a fraction of peak balance."""
        if not pnl_list:
            return 0.0

        balance = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for pnl in pnl_list:
            balance += pnl
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd
