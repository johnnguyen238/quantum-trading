"""Tests for configuration loading."""

from src.config.constants import Scenario, TrendDirection
from src.config.settings import load_settings


class TestSettings:
    def test_load_default_settings(self):
        settings = load_settings("default")
        assert settings.exchange.name == "bybit"
        assert settings.trading.max_leverage == 10
        assert settings.quantum.n_qubits == 4

    def test_load_testnet_profile(self):
        settings = load_settings("testnet")
        assert settings.exchange.testnet is True
        assert settings.trading.max_leverage == 5

    def test_load_backtest_profile(self):
        settings = load_settings("backtest")
        assert settings.backtest.initial_balance == 10000.0


class TestConstants:
    def test_trend_direction_values(self):
        assert TrendDirection.LONG == "LONG"
        assert TrendDirection.SHORT == "SHORT"
        assert TrendDirection.NEUTRAL == "NEUTRAL"

    def test_scenario_values(self):
        assert Scenario.HOLD == 1
        assert Scenario.DCA == 2
        assert Scenario.REVERSAL == 3
