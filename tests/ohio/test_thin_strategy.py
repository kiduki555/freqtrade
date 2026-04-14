"""Tests for OhioThinStrategy (FT-003 skeleton + FT-017 wiring).

Verifies class attributes, pipeline wiring, helper methods,
and all IStrategy callbacks.
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.thin_strategy import OhioThinStrategy
from freqtrade.ohio.core.domain.models import (
    DataMode,
    ExecutionPolicy,
    StateMeta,
    StrategyFitness,
    StrategyMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeDP:
    """Minimal DataProvider stub for tests."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_analyzed_dataframe(self, pair, timeframe):
        return self._df, None


def _make_strategy(**config_overrides) -> OhioThinStrategy:
    """Create a strategy instance with mocked pipeline stages."""
    config = {
        "stake_currency": "USDT",
        "stake_amount": 100,
        "dry_run": True,
        "exchange": {"name": "binance"},
        **config_overrides,
    }
    # Use lambda factories so each MagicMock() call returns a plain MagicMock instance
    with patch.multiple(
        "freqtrade.ohio.adapters.freqtrade.thin_strategy",
        FeatureBuilder=lambda: MagicMock(),
        Normalizer=lambda **kw: MagicMock(),
        FactorCalculator=lambda: MagicMock(),
        StateStabilizer=lambda: MagicMock(),
        MetaCalculator=lambda: MagicMock(),
        FitnessEstimator=lambda **kw: MagicMock(),
        PolicyGenerator=lambda: MagicMock(),
        DrawdownController=lambda: MagicMock(),
        KillSwitch=lambda: MagicMock(),
        RiskGateAdapter=lambda dd, ks: MagicMock(),
        load_default_profiles=lambda: {},
    ):
        return OhioThinStrategy(config=config)


def _ohio_row_series(**overrides) -> pd.Series:
    """Build a pandas Series with typical ohio_* columns."""
    defaults = {
        "ohio_active_mode": "trend_following",
        "ohio_policy_enabled": True,
        "ohio_policy_size_multiplier": 1.2,
        "ohio_policy_entry_threshold_adj": 0.05,
        "ohio_policy_max_positions": 3,
        "ohio_policy_stoploss_width_adj": -0.005,
        "ohio_meta_transition_risk": 0.15,
        "ohio_meta_confidence": 0.85,
        "ohio_meta_stability": 0.9,
        "ohio_meta_data_mode": "full",
        "ohio_fitness_trend_following": 0.82,
        "ohio_fitness_mean_reversion": 0.45,
        "ohio_fitness_breakout": 0.31,
        "ohio_fitness_defensive": 0.55,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _ohio_dataframe(n: int = 3, **overrides) -> pd.DataFrame:
    """Build a DataFrame with OHLCV + ohio columns."""
    base = {
        "open": [100.0] * n,
        "high": [105.0] * n,
        "low": [99.0] * n,
        "close": [103.0] * n,
        "volume": [1000.0] * n,
        "ohio_active_mode": ["trend_following"] * n,
        "ohio_policy_enabled": [True] * n,
        "ohio_policy_size_multiplier": [1.2] * n,
        "ohio_policy_entry_threshold_adj": [0.05] * n,
        "ohio_policy_max_positions": [3] * n,
        "ohio_policy_stoploss_width_adj": [-0.005] * n,
        "ohio_meta_transition_risk": [0.15] * n,
        "ohio_meta_confidence": [0.85] * n,
        "ohio_meta_stability": [0.9] * n,
        "ohio_meta_data_mode": ["full"] * n,
        "ohio_fitness_trend_following": [0.82] * n,
        "ohio_fitness_mean_reversion": [0.45] * n,
        "ohio_fitness_breakout": [0.31] * n,
        "ohio_fitness_defensive": [0.55] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def strategy() -> OhioThinStrategy:
    """Return OhioThinStrategy with mocked pipeline stages."""
    return _make_strategy()


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """A minimal OHLCV DataFrame sufficient for callback tests."""
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


@pytest.fixture()
def mock_trade() -> MagicMock:
    trade = MagicMock()
    trade.id = 1
    trade.enter_tag = "ohio_trend_following"
    trade.entry_side = "buy"
    trade.open_date_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return trade


@pytest.fixture()
def mock_order() -> MagicMock:
    order = MagicMock()
    order.ft_order_side = "buy"
    return order


@pytest.fixture()
def now() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Class-level attribute tests
# ---------------------------------------------------------------------------

class TestClassAttributes:
    def test_timeframe(self):
        assert OhioThinStrategy.timeframe == "1h"

    def test_stoploss(self):
        assert OhioThinStrategy.stoploss == -0.20

    def test_startup_candle_count(self):
        assert OhioThinStrategy.startup_candle_count == 4320

    def test_interface_version(self):
        assert OhioThinStrategy.INTERFACE_VERSION == 3

    def test_can_short(self):
        # V1 is long-only; short entries not yet implemented
        assert OhioThinStrategy.can_short is False

    def test_minimal_roi(self):
        assert OhioThinStrategy.minimal_roi == {"0": 100}

    def test_use_exit_signal(self):
        assert OhioThinStrategy.use_exit_signal is True

    def test_process_only_new_candles(self):
        assert OhioThinStrategy.process_only_new_candles is True


# ---------------------------------------------------------------------------
# __init__ tests (FT-017)
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_feature_builder(self, strategy):
        assert strategy._feature_builder is not None

    def test_creates_normalizer(self, strategy):
        assert strategy._normalizer is not None

    def test_creates_factor_calculator(self, strategy):
        assert strategy._factor_calculator is not None

    def test_creates_stabilizer(self, strategy):
        assert strategy._stabilizer is not None

    def test_creates_meta_calculator(self, strategy):
        assert strategy._meta_calculator is not None

    def test_creates_fitness_estimator(self, strategy):
        assert strategy._fitness_estimator is not None

    def test_creates_policy_generator(self, strategy):
        assert strategy._policy_generator is not None

    def test_creates_dd_controller(self, strategy):
        assert strategy._dd_controller is not None

    def test_creates_kill_switch(self, strategy):
        assert strategy._kill_switch is not None

    def test_creates_risk_gate(self, strategy):
        assert strategy._risk_gate is not None

    def test_creates_profiles(self, strategy):
        assert strategy._profiles is not None


# ---------------------------------------------------------------------------
# populate_indicators tests (FT-017)
# ---------------------------------------------------------------------------

class TestPopulateIndicators:
    def test_returns_dataframe(self, strategy, simple_df):
        # Mock pipeline stages to return the df as-is
        strategy._feature_builder.compute.return_value = simple_df
        strategy._normalizer.normalize.return_value = simple_df
        strategy._factor_calculator.compute.return_value = simple_df
        strategy._stabilizer.stabilize.return_value = simple_df
        strategy._meta_calculator.compute.return_value = simple_df
        strategy._fitness_estimator.compute_dataframe.return_value = simple_df
        strategy._policy_generator.generate_dataframe.return_value = simple_df

        result = strategy.populate_indicators(simple_df, {"pair": "BTC/USDT"})
        assert isinstance(result, pd.DataFrame)

    def test_calls_all_pipeline_stages(self, strategy, simple_df):
        strategy._feature_builder.compute.return_value = simple_df
        strategy._normalizer.normalize.return_value = simple_df
        strategy._factor_calculator.compute.return_value = simple_df
        strategy._stabilizer.stabilize.return_value = simple_df
        strategy._meta_calculator.compute.return_value = simple_df
        strategy._fitness_estimator.compute_dataframe.return_value = simple_df
        strategy._policy_generator.generate_dataframe.return_value = simple_df

        strategy.populate_indicators(simple_df, {"pair": "BTC/USDT"})

        strategy._feature_builder.compute.assert_called_once()
        strategy._normalizer.normalize.assert_called_once()
        strategy._factor_calculator.compute.assert_called_once()
        strategy._stabilizer.stabilize.assert_called_once()
        strategy._meta_calculator.compute.assert_called_once()
        strategy._fitness_estimator.compute_dataframe.assert_called_once()
        strategy._policy_generator.generate_dataframe.assert_called_once()

    def test_does_not_drop_rows(self, strategy, simple_df):
        strategy._feature_builder.compute.return_value = simple_df
        strategy._normalizer.normalize.return_value = simple_df
        strategy._factor_calculator.compute.return_value = simple_df
        strategy._stabilizer.stabilize.return_value = simple_df
        strategy._meta_calculator.compute.return_value = simple_df
        strategy._fitness_estimator.compute_dataframe.return_value = simple_df
        strategy._policy_generator.generate_dataframe.return_value = simple_df

        result = strategy.populate_indicators(simple_df, {"pair": "BTC/USDT"})
        assert len(result) == len(simple_df)


# ---------------------------------------------------------------------------
# populate_entry_trend tests (FT-017)
# ---------------------------------------------------------------------------

class TestPopulateEntryTrend:
    def test_sets_enter_long_for_enabled_rows(self, strategy):
        df = _ohio_dataframe(3, ohio_policy_enabled=[True, False, True])
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert result["enter_long"].iloc[0] == 1
        assert result["enter_long"].iloc[2] == 1

    def test_does_not_set_enter_long_for_disabled_rows(self, strategy):
        df = _ohio_dataframe(3, ohio_policy_enabled=[False, False, False])
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert (result["enter_long"] == 0).all()

    def test_sets_enter_tag_with_ohio_prefix(self, strategy):
        df = _ohio_dataframe(2, ohio_policy_enabled=[True, True])
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert result["enter_tag"].iloc[0] == "ohio_trend_following"
        assert result["enter_tag"].iloc[1] == "ohio_trend_following"

    def test_enter_tag_empty_for_disabled(self, strategy):
        df = _ohio_dataframe(2, ohio_policy_enabled=[False, False])
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert (result["enter_tag"] == "").all()

    def test_adds_enter_long_column(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert "enter_long" in result.columns

    def test_adds_enter_short_column(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert "enter_short" in result.columns

    def test_adds_enter_tag_column(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert "enter_tag" in result.columns

    def test_enter_short_always_zero(self, strategy):
        df = _ohio_dataframe(3, ohio_policy_enabled=[True, True, True])
        result = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
        assert (result["enter_short"] == 0).all()


# ---------------------------------------------------------------------------
# populate_exit_trend tests
# ---------------------------------------------------------------------------

class TestPopulateExitTrend:
    def test_adds_exit_long_column(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_exit_trend(df, {"pair": "BTC/USDT"})
        assert "exit_long" in result.columns

    def test_adds_exit_short_column(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_exit_trend(df, {"pair": "BTC/USDT"})
        assert "exit_short" in result.columns

    def test_exit_long_all_zeros(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_exit_trend(df, {"pair": "BTC/USDT"})
        assert (result["exit_long"] == 0).all()

    def test_exit_short_all_zeros(self, strategy):
        df = _ohio_dataframe(2)
        result = strategy.populate_exit_trend(df, {"pair": "BTC/USDT"})
        assert (result["exit_short"] == 0).all()


# ---------------------------------------------------------------------------
# Helper method tests (FT-017)
# ---------------------------------------------------------------------------

class TestBuildPolicyFromRow:
    def test_returns_execution_policy(self, strategy):
        row = _ohio_row_series()
        policy = strategy._build_policy_from_row(row)
        assert isinstance(policy, ExecutionPolicy)

    def test_correct_mode(self, strategy):
        row = _ohio_row_series(ohio_active_mode="mean_reversion")
        policy = strategy._build_policy_from_row(row)
        assert policy.strategy_mode == StrategyMode.MEAN_REVERSION

    def test_correct_enabled(self, strategy):
        row = _ohio_row_series(ohio_policy_enabled=True)
        policy = strategy._build_policy_from_row(row)
        assert policy.enabled is True

    def test_correct_size_multiplier(self, strategy):
        row = _ohio_row_series(ohio_policy_size_multiplier=1.2)
        policy = strategy._build_policy_from_row(row)
        assert policy.size_multiplier == pytest.approx(1.2)

    def test_correct_max_positions(self, strategy):
        row = _ohio_row_series(ohio_policy_max_positions=3)
        policy = strategy._build_policy_from_row(row)
        assert policy.max_positions == 3

    def test_defaults_for_missing(self, strategy):
        row = pd.Series({})
        policy = strategy._build_policy_from_row(row)
        assert policy.strategy_mode == StrategyMode.DEFENSIVE
        assert policy.enabled is False
        assert policy.size_multiplier == 1.0


class TestBuildMetaFromRow:
    def test_returns_state_meta(self, strategy):
        row = _ohio_row_series()
        meta = strategy._build_meta_from_row(row)
        assert isinstance(meta, StateMeta)

    def test_correct_transition_risk(self, strategy):
        row = _ohio_row_series(ohio_meta_transition_risk=0.15)
        meta = strategy._build_meta_from_row(row)
        assert meta.transition_risk == pytest.approx(0.15)

    def test_correct_confidence(self, strategy):
        row = _ohio_row_series(ohio_meta_confidence=0.85)
        meta = strategy._build_meta_from_row(row)
        assert meta.confidence == pytest.approx(0.85)

    def test_correct_data_mode(self, strategy):
        row = _ohio_row_series(ohio_meta_data_mode="fallback")
        meta = strategy._build_meta_from_row(row)
        assert meta.data_mode == DataMode.FALLBACK

    def test_defaults_for_missing(self, strategy):
        row = pd.Series({})
        meta = strategy._build_meta_from_row(row)
        assert meta.transition_risk == 0.0
        assert meta.data_mode == DataMode.FULL


class TestBuildFitnessFromRow:
    def test_returns_strategy_fitness(self, strategy):
        row = _ohio_row_series()
        fitness = strategy._build_fitness_from_row(row)
        assert isinstance(fitness, StrategyFitness)

    def test_correct_values(self, strategy):
        row = _ohio_row_series(
            ohio_fitness_trend_following=0.82,
            ohio_fitness_mean_reversion=0.45,
        )
        fitness = strategy._build_fitness_from_row(row)
        assert fitness.trend_following == pytest.approx(0.82)
        assert fitness.mean_reversion == pytest.approx(0.45)


class TestGetBestFitness:
    def test_returns_fitness_for_active_mode(self, strategy):
        row = _ohio_row_series(
            ohio_active_mode="trend_following",
            ohio_fitness_trend_following=0.82,
        )
        assert strategy._get_best_fitness(row) == pytest.approx(0.82)

    def test_returns_defensive_by_default(self, strategy):
        row = pd.Series({"ohio_fitness_defensive": 0.55})
        assert strategy._get_best_fitness(row) == pytest.approx(0.55)

    def test_returns_0_5_when_column_missing(self, strategy):
        row = pd.Series({"ohio_active_mode": "breakout"})
        assert strategy._get_best_fitness(row) == pytest.approx(0.5)


class TestExtractMode:
    def test_strips_ohio_prefix(self):
        assert OhioThinStrategy._extract_mode("ohio_trend_following") == "trend_following"

    def test_strips_ohio_prefix_mean_reversion(self):
        assert OhioThinStrategy._extract_mode("ohio_mean_reversion") == "mean_reversion"

    def test_handles_none(self):
        assert OhioThinStrategy._extract_mode(None) is None

    def test_passes_through_non_ohio(self):
        assert OhioThinStrategy._extract_mode("other_tag") == "other_tag"

    def test_handles_empty_string(self):
        assert OhioThinStrategy._extract_mode("") == ""


class TestBarsSinceEntry:
    def test_calculates_correct_hours(self, strategy):
        trade = MagicMock()
        trade.open_date_utc = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        current = datetime(2024, 1, 1, 5, 30, tzinfo=timezone.utc)
        assert strategy._bars_since_entry(trade, current) == 5  # truncated

    def test_returns_zero_for_none_open_date(self, strategy):
        trade = MagicMock()
        trade.open_date_utc = None
        current = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert strategy._bars_since_entry(trade, current) == 0

    def test_zero_for_same_time(self, strategy):
        trade = MagicMock()
        trade.open_date_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)
        current = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert strategy._bars_since_entry(trade, current) == 0


class TestBuildFitnessDict:
    def test_returns_all_four_modes(self, strategy):
        row = _ohio_row_series()
        result = strategy._build_fitness_dict(row)
        assert "trend_following" in result
        assert "mean_reversion" in result
        assert "breakout" in result
        assert "defensive" in result

    def test_correct_values(self, strategy):
        row = _ohio_row_series(
            ohio_fitness_trend_following=0.82,
            ohio_fitness_defensive=0.55,
        )
        result = strategy._build_fitness_dict(row)
        assert result["trend_following"] == pytest.approx(0.82)
        assert result["defensive"] == pytest.approx(0.55)

    def test_defaults_to_zero(self, strategy):
        row = pd.Series({})
        result = strategy._build_fitness_dict(row)
        for mode in StrategyMode:
            assert result[mode.value] == 0.0


# ---------------------------------------------------------------------------
# custom_stake_amount tests (FT-017)
# ---------------------------------------------------------------------------

class TestCustomStakeAmount:
    def test_returns_proposed_when_no_dataframe(self, strategy, now):
        strategy.dp = FakeDP(pd.DataFrame())
        result = strategy.custom_stake_amount(
            pair="BTC/USDT", current_time=now, current_rate=50000.0,
            proposed_stake=100.0, min_stake=10.0, max_stake=1000.0,
            leverage=1.0, entry_tag="ohio_trend_following", side="long",
        )
        assert result == 100.0

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.compute_stake")
    def test_calls_compute_stake(self, mock_compute, strategy, now):
        mock_compute.return_value = 75.0
        strategy.dp = FakeDP(_ohio_dataframe(1))
        strategy._dd_controller.size_scale = 0.8

        result = strategy.custom_stake_amount(
            pair="BTC/USDT", current_time=now, current_rate=50000.0,
            proposed_stake=100.0, min_stake=10.0, max_stake=1000.0,
            leverage=1.0, entry_tag="ohio_trend_following", side="long",
        )
        assert result == 75.0
        mock_compute.assert_called_once()

    def test_returns_float(self, strategy, now):
        strategy.dp = FakeDP(pd.DataFrame())
        result = strategy.custom_stake_amount(
            pair="ETH/USDT", current_time=now, current_rate=3000.0,
            proposed_stake=250.0, min_stake=5.0, max_stake=500.0,
            leverage=1.0, entry_tag=None, side="short",
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# leverage tests (FT-017)
# ---------------------------------------------------------------------------

class TestLeverage:
    def test_returns_1_when_no_profile(self, strategy, now):
        strategy._profiles = {}
        strategy.dp = FakeDP(_ohio_dataframe(1))
        result = strategy.leverage(
            pair="BTC/USDT", current_time=now, current_rate=50000.0,
            proposed_leverage=3.0, max_leverage=10.0,
            entry_tag="ohio_unknown_mode", side="long",
        )
        assert result == 1.0

    def test_returns_1_when_no_entry_tag(self, strategy, now):
        strategy.dp = FakeDP(_ohio_dataframe(1))
        result = strategy.leverage(
            pair="BTC/USDT", current_time=now, current_rate=50000.0,
            proposed_leverage=3.0, max_leverage=10.0,
            entry_tag=None, side="long",
        )
        assert result == 1.0

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.compute_leverage")
    def test_calls_compute_leverage(self, mock_lev, strategy, now):
        mock_lev.return_value = 2.5
        mock_profile = MagicMock()
        strategy._profiles = {StrategyMode.TREND_FOLLOWING: mock_profile}
        strategy.dp = FakeDP(_ohio_dataframe(1))

        result = strategy.leverage(
            pair="BTC/USDT", current_time=now, current_rate=50000.0,
            proposed_leverage=3.0, max_leverage=10.0,
            entry_tag="ohio_trend_following", side="long",
        )
        assert result == 2.5
        mock_lev.assert_called_once()

    def test_returns_float(self, strategy, now):
        strategy.dp = FakeDP(_ohio_dataframe(1))
        result = strategy.leverage(
            pair="ETH/USDT", current_time=now, current_rate=3000.0,
            proposed_leverage=5.0, max_leverage=20.0,
            entry_tag=None, side="short",
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# confirm_trade_entry tests (FT-017)
# ---------------------------------------------------------------------------

class TestConfirmTradeEntry:
    def test_returns_false_when_no_dataframe(self, strategy, now):
        strategy.dp = FakeDP(pd.DataFrame())
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT", order_type="limit", amount=0.002,
            rate=50000.0, time_in_force="GTC", current_time=now,
            entry_tag="ohio_trend_following", side="long",
        )
        assert result is False

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.Trade")
    def test_calls_risk_gate(self, mock_trade_cls, strategy, now):
        mock_trade_cls.get_trades_proxy.return_value = []
        strategy.dp = FakeDP(_ohio_dataframe(1))
        strategy._risk_gate.confirm_entry.return_value = (True, [])

        result = strategy.confirm_trade_entry(
            pair="BTC/USDT", order_type="limit", amount=0.002,
            rate=50000.0, time_in_force="GTC", current_time=now,
            entry_tag="ohio_trend_following", side="long",
        )
        assert result is True
        strategy._risk_gate.confirm_entry.assert_called_once()

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.Trade")
    def test_blocks_when_risk_gate_denies(self, mock_trade_cls, strategy, now):
        mock_trade_cls.get_trades_proxy.return_value = []
        strategy.dp = FakeDP(_ohio_dataframe(1))
        strategy._risk_gate.confirm_entry.return_value = (False, ["kill_switch"])

        result = strategy.confirm_trade_entry(
            pair="BTC/USDT", order_type="limit", amount=0.002,
            rate=50000.0, time_in_force="GTC", current_time=now,
            entry_tag="ohio_trend_following", side="long",
        )
        assert result is False


# ---------------------------------------------------------------------------
# custom_stoploss tests (FT-017)
# ---------------------------------------------------------------------------

class TestCustomStoploss:
    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.get_trade_mode")
    def test_returns_none_when_no_mode(self, mock_get_mode, strategy, mock_trade, now):
        mock_get_mode.return_value = None
        result = strategy.custom_stoploss(
            pair="BTC/USDT", trade=mock_trade, current_time=now,
            current_rate=50000.0, current_profit=0.05, after_fill=False,
        )
        assert result is None

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.compute_stoploss")
    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.get_trade_mode")
    def test_calls_compute_stoploss(self, mock_get_mode, mock_sl, strategy, mock_trade, now):
        mock_get_mode.return_value = "trend_following"
        mock_sl.return_value = -0.08
        mock_profile = MagicMock()
        strategy._profiles = {StrategyMode.TREND_FOLLOWING: mock_profile}
        strategy.dp = FakeDP(_ohio_dataframe(1))

        result = strategy.custom_stoploss(
            pair="BTC/USDT", trade=mock_trade, current_time=now,
            current_rate=50000.0, current_profit=0.05, after_fill=False,
        )
        assert result == -0.08
        mock_sl.assert_called_once()


# ---------------------------------------------------------------------------
# custom_exit tests (FT-017)
# ---------------------------------------------------------------------------

class TestCustomExit:
    def test_returns_none_when_no_dataframe(self, strategy, mock_trade, now):
        strategy.dp = FakeDP(pd.DataFrame())
        result = strategy.custom_exit(
            pair="BTC/USDT", trade=mock_trade, current_time=now,
            current_rate=50000.0, current_profit=0.05,
        )
        assert result is None

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.compute_exit")
    def test_calls_compute_exit(self, mock_exit, strategy, mock_trade, now):
        mock_exit.return_value = "fitness_drop"
        strategy.dp = FakeDP(_ohio_dataframe(1))
        strategy._kill_switch.active = False

        result = strategy.custom_exit(
            pair="BTC/USDT", trade=mock_trade, current_time=now,
            current_rate=50000.0, current_profit=0.05,
        )
        assert result == "fitness_drop"
        mock_exit.assert_called_once()


# ---------------------------------------------------------------------------
# order_filled tests (FT-017)
# ---------------------------------------------------------------------------

class TestOrderFilled:
    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.save_entry_metadata")
    def test_saves_metadata_on_entry_fill(self, mock_save, strategy, mock_trade, mock_order, now):
        mock_order.ft_order_side = "buy"
        mock_trade.entry_side = "buy"
        strategy.dp = FakeDP(_ohio_dataframe(1))

        strategy.order_filled(
            pair="BTC/USDT", trade=mock_trade, order=mock_order, current_time=now,
        )
        mock_save.assert_called_once()

    @patch("freqtrade.ohio.adapters.freqtrade.thin_strategy.save_entry_metadata")
    def test_does_not_save_on_exit_fill(self, mock_save, strategy, mock_trade, mock_order, now):
        mock_order.ft_order_side = "sell"
        mock_trade.entry_side = "buy"
        strategy.dp = FakeDP(_ohio_dataframe(1))

        strategy.order_filled(
            pair="BTC/USDT", trade=mock_trade, order=mock_order, current_time=now,
        )
        mock_save.assert_not_called()

    def test_returns_none(self, strategy, mock_trade, mock_order, now):
        mock_order.ft_order_side = "sell"
        mock_trade.entry_side = "buy"
        strategy.dp = FakeDP(_ohio_dataframe(1))

        result = strategy.order_filled(
            pair="BTC/USDT", trade=mock_trade, order=mock_order, current_time=now,
        )
        assert result is None
