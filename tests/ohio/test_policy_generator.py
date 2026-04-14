"""Tests for FT-012: Policy Generator.

Covers:
1.  fitness=0 → enabled=False
2.  fitness=1 → enabled=True, size_multiplier=1.5, max_positions=12
3.  fitness=0.5 → intermediate values
4.  fitness exactly at threshold → enabled=True
5.  fitness just below threshold → enabled=False
6.  degraded data_mode → enabled=False regardless of fitness
7.  high transition_risk reduces size_multiplier
8.  low confidence reduces size_multiplier
9.  fallback data_mode caps max_positions at 3
10. generate_best selects highest fitness mode
11. generate_best returns correct mode in policy
12. All lerp boundaries are correct (0 and 1 endpoints)
13. size_multiplier always in [0.3, 1.5] even with meta adjustments (clip)
14. entry_threshold_adj always in [-0.05, 0.15]
15. stoploss_width_adj always in [-0.01, 0.01]
16. generate_dataframe adds 5 ohio_policy_* columns
17. generate_dataframe preserves row count
18. No Freqtrade imports in module (AST check)
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.domain.models import (
    DataMode,
    ExecutionPolicy,
    StateMeta,
    StrategyFitness,
    StrategyMode,
)
from freqtrade.ohio.core.strategy_router.policy_generator import PolicyGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THRESHOLD = 0.15


def _meta(
    transition_risk: float = 0.0,
    confidence: float = 1.0,
    stability: float = 1.0,
    data_mode: DataMode = DataMode.FULL,
) -> StateMeta:
    return StateMeta(
        transition_risk=transition_risk,
        confidence=confidence,
        stability=stability,
        data_mode=data_mode,
    )


def _clean_meta() -> StateMeta:
    """Meta with no quality adjustments triggered."""
    return _meta(transition_risk=0.0, confidence=1.0, data_mode=DataMode.FULL)


_gen = PolicyGenerator(disabled_threshold=_THRESHOLD)


# ---------------------------------------------------------------------------
# 1. fitness=0 → enabled=False
# ---------------------------------------------------------------------------

class TestFitnessZero:
    def test_enabled_false_when_fitness_zero(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, _clean_meta())
        assert policy.enabled is False

    def test_size_multiplier_at_min_when_fitness_zero(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, _clean_meta())
        assert pytest.approx(policy.size_multiplier) == 0.3

    def test_entry_threshold_adj_at_max_when_fitness_zero(self) -> None:
        # low fitness → harder entry → 0.15
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, _clean_meta())
        assert pytest.approx(policy.entry_threshold_adj) == 0.15

    def test_max_positions_at_one_when_fitness_zero(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, _clean_meta())
        assert policy.max_positions == 1

    def test_stoploss_adj_at_min_when_fitness_zero(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, _clean_meta())
        assert pytest.approx(policy.stoploss_width_adj) == -0.01


# ---------------------------------------------------------------------------
# 2. fitness=1 → enabled=True, size_multiplier=1.5, max_positions=6
# ---------------------------------------------------------------------------

class TestFitnessOne:
    def test_enabled_true_when_fitness_one(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, _clean_meta())
        assert policy.enabled is True

    def test_size_multiplier_at_max_when_fitness_one(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, _clean_meta())
        assert pytest.approx(policy.size_multiplier) == 1.5

    def test_entry_threshold_adj_at_min_when_fitness_one(self) -> None:
        # high fitness → easier entry → -0.05
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, _clean_meta())
        assert pytest.approx(policy.entry_threshold_adj) == -0.05

    def test_max_positions_at_twelve_when_fitness_one(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, _clean_meta())
        assert policy.max_positions == 12

    def test_stoploss_adj_at_max_when_fitness_one(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, _clean_meta())
        assert pytest.approx(policy.stoploss_width_adj) == 0.01


# ---------------------------------------------------------------------------
# 3. fitness=0.5 → intermediate values
# ---------------------------------------------------------------------------

class TestFitnessHalf:
    def test_enabled_true_when_fitness_half(self) -> None:
        policy = _gen.generate(StrategyMode.MEAN_REVERSION, 0.5, _clean_meta())
        assert policy.enabled is True

    def test_size_multiplier_midpoint(self) -> None:
        # lerp(0.3, 1.5, 0.5) = 0.9
        policy = _gen.generate(StrategyMode.MEAN_REVERSION, 0.5, _clean_meta())
        assert pytest.approx(policy.size_multiplier) == 0.9

    def test_entry_threshold_adj_midpoint(self) -> None:
        # lerp(0.15, -0.05, 0.5) = 0.05
        policy = _gen.generate(StrategyMode.MEAN_REVERSION, 0.5, _clean_meta())
        assert pytest.approx(policy.entry_threshold_adj) == 0.05

    def test_max_positions_midpoint(self) -> None:
        # lerp(1, 12, 0.5) = 6.5 → round → 6
        policy = _gen.generate(StrategyMode.MEAN_REVERSION, 0.5, _clean_meta())
        assert policy.max_positions == 6

    def test_stoploss_adj_midpoint(self) -> None:
        # lerp(-0.01, 0.01, 0.5) = 0.0
        policy = _gen.generate(StrategyMode.MEAN_REVERSION, 0.5, _clean_meta())
        assert pytest.approx(policy.stoploss_width_adj, abs=1e-9) == 0.0


# ---------------------------------------------------------------------------
# 4. fitness exactly at threshold → enabled=True
# ---------------------------------------------------------------------------

class TestFitnessAtThreshold:
    def test_enabled_true_at_exact_threshold(self) -> None:
        policy = _gen.generate(StrategyMode.BREAKOUT, _THRESHOLD, _clean_meta())
        assert policy.enabled is True

    def test_returns_execution_policy_instance(self) -> None:
        policy = _gen.generate(StrategyMode.BREAKOUT, _THRESHOLD, _clean_meta())
        assert isinstance(policy, ExecutionPolicy)


# ---------------------------------------------------------------------------
# 5. fitness just below threshold → enabled=False
# ---------------------------------------------------------------------------

class TestFitnessBelowThreshold:
    def test_enabled_false_just_below_threshold(self) -> None:
        just_below = _THRESHOLD - 1e-9
        policy = _gen.generate(StrategyMode.BREAKOUT, just_below, _clean_meta())
        assert policy.enabled is False

    def test_numeric_outputs_still_computed_below_threshold(self) -> None:
        # enabled=False doesn't suppress numeric computation
        just_below = _THRESHOLD - 0.01
        policy = _gen.generate(StrategyMode.BREAKOUT, just_below, _clean_meta())
        assert 0.3 <= policy.size_multiplier <= 1.5


# ---------------------------------------------------------------------------
# 6. degraded data_mode → enabled=False regardless of fitness
# ---------------------------------------------------------------------------

class TestDegradedDataMode:
    def test_enabled_false_when_degraded_high_fitness(self) -> None:
        degraded = _meta(data_mode=DataMode.DEGRADED)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, degraded)
        assert policy.enabled is False

    def test_enabled_false_when_degraded_fitness_above_threshold(self) -> None:
        degraded = _meta(data_mode=DataMode.DEGRADED)
        policy = _gen.generate(StrategyMode.DEFENSIVE, 0.5, degraded)
        assert policy.enabled is False

    def test_enabled_false_when_degraded_at_threshold(self) -> None:
        degraded = _meta(data_mode=DataMode.DEGRADED)
        policy = _gen.generate(StrategyMode.BREAKOUT, _THRESHOLD, degraded)
        assert policy.enabled is False


# ---------------------------------------------------------------------------
# 7. high transition_risk reduces size_multiplier
# ---------------------------------------------------------------------------

class TestHighTransitionRisk:
    def test_size_multiplier_reduced_when_high_risk(self) -> None:
        normal = _clean_meta()
        high_risk = _meta(transition_risk=0.75, confidence=1.0)

        policy_normal = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.8, normal)
        policy_risky = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.8, high_risk)

        assert policy_risky.size_multiplier < policy_normal.size_multiplier

    def test_size_reduced_by_30_percent_when_risk_above_07(self) -> None:
        # fitness=1 → raw size 1.5, after 30% reduction → 1.05
        high_risk = _meta(transition_risk=0.71, confidence=1.0)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, high_risk)
        assert pytest.approx(policy.size_multiplier) == 1.05

    def test_size_not_reduced_at_exact_07_threshold(self) -> None:
        # transition_risk == 0.70 should NOT trigger reduction (strict >)
        at_threshold = _meta(transition_risk=0.70, confidence=1.0)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, at_threshold)
        assert pytest.approx(policy.size_multiplier) == 1.5


# ---------------------------------------------------------------------------
# 8. low confidence reduces size_multiplier
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_size_multiplier_reduced_when_low_confidence(self) -> None:
        normal = _clean_meta()
        low_conf = _meta(confidence=0.3)

        policy_normal = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.8, normal)
        policy_low = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.8, low_conf)

        assert policy_low.size_multiplier < policy_normal.size_multiplier

    def test_size_reduced_by_20_percent_when_confidence_below_05(self) -> None:
        # fitness=1 → raw size 1.5, after 20% reduction → 1.2
        low_conf = _meta(transition_risk=0.0, confidence=0.4)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, low_conf)
        assert pytest.approx(policy.size_multiplier) == 1.2

    def test_size_not_reduced_at_exact_05_confidence(self) -> None:
        # confidence == 0.5 should NOT trigger reduction (strict <)
        at_threshold = _meta(confidence=0.50)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, at_threshold)
        assert pytest.approx(policy.size_multiplier) == 1.5


# ---------------------------------------------------------------------------
# 9. fallback data_mode caps max_positions at 3
# ---------------------------------------------------------------------------

class TestFallbackDataMode:
    def test_max_positions_capped_at_3_when_fallback(self) -> None:
        fallback = _meta(data_mode=DataMode.FALLBACK)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, fallback)
        assert policy.max_positions == 3

    def test_max_positions_still_1_when_fallback_and_low_fitness(self) -> None:
        fallback = _meta(data_mode=DataMode.FALLBACK)
        policy = _gen.generate(StrategyMode.DEFENSIVE, 0.0, fallback)
        assert policy.max_positions == 1  # already at min, cap doesn't lower it

    def test_enabled_still_possible_with_fallback(self) -> None:
        fallback = _meta(data_mode=DataMode.FALLBACK)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.5, fallback)
        assert policy.enabled is True  # FALLBACK ≠ DEGRADED


# ---------------------------------------------------------------------------
# 10. generate_best selects highest fitness mode
# ---------------------------------------------------------------------------

class TestGenerateBest:
    def test_best_mode_is_highest_fitness(self) -> None:
        fitness = StrategyFitness(
            trend_following=0.9,
            mean_reversion=0.4,
            breakout=0.6,
            defensive=0.2,
        )
        policy = _gen.generate_best(fitness, _clean_meta())
        assert policy.strategy_mode == StrategyMode.TREND_FOLLOWING

    def test_best_mode_defensive_when_highest(self) -> None:
        fitness = StrategyFitness(
            trend_following=0.1,
            mean_reversion=0.2,
            breakout=0.3,
            defensive=0.95,
        )
        policy = _gen.generate_best(fitness, _clean_meta())
        assert policy.strategy_mode == StrategyMode.DEFENSIVE


# ---------------------------------------------------------------------------
# 11. generate_best returns correct mode in policy
# ---------------------------------------------------------------------------

class TestGenerateBestPolicyMode:
    def test_policy_mode_matches_best(self) -> None:
        fitness = StrategyFitness(
            trend_following=0.3,
            mean_reversion=0.8,
            breakout=0.5,
            defensive=0.1,
        )
        policy = _gen.generate_best(fitness, _clean_meta())
        assert policy.strategy_mode == StrategyMode.MEAN_REVERSION

    def test_policy_size_matches_expected_fitness(self) -> None:
        # best = breakout @ 0.7 → lerp(0.3, 1.5, 0.7) = 0.3 + 1.2 * 0.7 = 1.14
        fitness = StrategyFitness(
            trend_following=0.3,
            mean_reversion=0.5,
            breakout=0.7,
            defensive=0.1,
        )
        policy = _gen.generate_best(fitness, _clean_meta())
        assert pytest.approx(policy.size_multiplier) == pytest.approx(1.14)


# ---------------------------------------------------------------------------
# 12. All lerp boundaries are correct (0 and 1 endpoints)
# ---------------------------------------------------------------------------

class TestLerpBoundaries:
    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_size_at_zero_fitness(self, mode: StrategyMode) -> None:
        policy = _gen.generate(mode, 0.0, _clean_meta())
        assert pytest.approx(policy.size_multiplier) == 0.3

    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_size_at_full_fitness(self, mode: StrategyMode) -> None:
        policy = _gen.generate(mode, 1.0, _clean_meta())
        assert pytest.approx(policy.size_multiplier) == 1.5

    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_entry_at_zero_fitness(self, mode: StrategyMode) -> None:
        policy = _gen.generate(mode, 0.0, _clean_meta())
        assert pytest.approx(policy.entry_threshold_adj) == 0.15

    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_entry_at_full_fitness(self, mode: StrategyMode) -> None:
        policy = _gen.generate(mode, 1.0, _clean_meta())
        assert pytest.approx(policy.entry_threshold_adj) == -0.05

    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_sl_adj_at_zero_fitness(self, mode: StrategyMode) -> None:
        policy = _gen.generate(mode, 0.0, _clean_meta())
        assert pytest.approx(policy.stoploss_width_adj) == -0.01

    @pytest.mark.parametrize("mode", list(StrategyMode))
    def test_sl_adj_at_full_fitness(self, mode: StrategyMode) -> None:
        policy = _gen.generate(mode, 1.0, _clean_meta())
        assert pytest.approx(policy.stoploss_width_adj) == 0.01


# ---------------------------------------------------------------------------
# 13. size_multiplier always in [0.3, 1.5] even with meta adjustments
# ---------------------------------------------------------------------------

class TestSizeMultiplierClipping:
    def test_size_never_below_03_with_combined_reductions(self) -> None:
        # both adjustments active → 0.3 * 0.7 * 0.8 = 0.168 → clips to 0.3
        bad_meta = _meta(transition_risk=0.9, confidence=0.1)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, bad_meta)
        assert policy.size_multiplier >= 0.3

    def test_size_never_exceeds_15(self) -> None:
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, _clean_meta())
        assert policy.size_multiplier <= 1.5

    @pytest.mark.parametrize("fitness", np.linspace(0.0, 1.0, 11).tolist())
    def test_size_always_in_range_clean_meta(self, fitness: float) -> None:
        policy = _gen.generate(StrategyMode.BREAKOUT, fitness, _clean_meta())
        assert 0.3 <= policy.size_multiplier <= 1.5

    @pytest.mark.parametrize("fitness", np.linspace(0.0, 1.0, 11).tolist())
    def test_size_always_in_range_worst_meta(self, fitness: float) -> None:
        worst = _meta(transition_risk=1.0, confidence=0.0)
        policy = _gen.generate(StrategyMode.BREAKOUT, fitness, worst)
        assert 0.3 <= policy.size_multiplier <= 1.5


# ---------------------------------------------------------------------------
# 14. entry_threshold_adj always in [-0.05, 0.15]
# ---------------------------------------------------------------------------

class TestEntryThresholdRange:
    @pytest.mark.parametrize("fitness", np.linspace(0.0, 1.0, 21).tolist())
    def test_entry_adj_always_in_range(self, fitness: float) -> None:
        policy = _gen.generate(StrategyMode.DEFENSIVE, fitness, _clean_meta())
        assert -0.05 <= policy.entry_threshold_adj <= 0.15


# ---------------------------------------------------------------------------
# 15. stoploss_width_adj always in [-0.01, 0.01]
# ---------------------------------------------------------------------------

class TestStoplossAdjRange:
    @pytest.mark.parametrize("fitness", np.linspace(0.0, 1.0, 21).tolist())
    def test_stoploss_adj_always_in_range(self, fitness: float) -> None:
        policy = _gen.generate(StrategyMode.MEAN_REVERSION, fitness, _clean_meta())
        assert -0.01 <= policy.stoploss_width_adj <= 0.01


# ---------------------------------------------------------------------------
# 16. generate_dataframe adds 5 ohio_policy_* columns
# ---------------------------------------------------------------------------

_POLICY_COLUMNS = {
    "ohio_policy_enabled",
    "ohio_policy_size_multiplier",
    "ohio_policy_entry_threshold_adj",
    "ohio_policy_max_positions",
    "ohio_policy_stoploss_width_adj",
}


def _make_df(n: int = 10, mode: str = "trend_following") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ohio_fitness_trend_following": np.linspace(0.0, 1.0, n),
            "ohio_fitness_mean_reversion": np.linspace(0.1, 0.9, n),
            "ohio_fitness_breakout": np.linspace(0.2, 0.8, n),
            "ohio_fitness_defensive": np.linspace(0.05, 0.6, n),
            "ohio_active_mode": [mode] * n,
            "ohio_meta_transition_risk": np.linspace(0.0, 0.5, n),
            "ohio_meta_confidence": np.linspace(0.6, 1.0, n),
            "ohio_meta_data_mode": ["full"] * n,
        }
    )


class TestGenerateDataframe:
    def test_adds_five_policy_columns(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        assert _POLICY_COLUMNS.issubset(set(result.columns))

    def test_correct_column_count_added(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        new_cols = set(result.columns) - set(df.columns)
        assert new_cols == _POLICY_COLUMNS

    def test_enabled_is_bool_dtype(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        assert result["ohio_policy_enabled"].dtype == bool

    def test_max_positions_is_integer_dtype(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        assert np.issubdtype(result["ohio_policy_max_positions"].dtype, np.integer)

    def test_size_multiplier_in_range(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        assert result["ohio_policy_size_multiplier"].between(0.3, 1.5).all()

    def test_entry_adj_in_range(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        assert result["ohio_policy_entry_threshold_adj"].between(-0.05, 0.15).all()

    def test_stoploss_adj_in_range(self) -> None:
        df = _make_df()
        result = _gen.generate_dataframe(df)
        assert result["ohio_policy_stoploss_width_adj"].between(-0.01, 0.01).all()

    def test_degraded_rows_disabled(self) -> None:
        df = _make_df(n=5)
        df["ohio_meta_data_mode"] = "degraded"
        df["ohio_fitness_trend_following"] = 1.0
        result = _gen.generate_dataframe(df)
        assert not result["ohio_policy_enabled"].any()

    def test_fallback_caps_max_positions(self) -> None:
        df = _make_df(n=5)
        df["ohio_meta_data_mode"] = "fallback"
        df["ohio_fitness_trend_following"] = 1.0
        result = _gen.generate_dataframe(df)
        assert (result["ohio_policy_max_positions"] <= 3).all()

    def test_high_fitness_row_enabled(self) -> None:
        df = _make_df(n=5)
        df["ohio_fitness_trend_following"] = 0.9
        result = _gen.generate_dataframe(df)
        assert result["ohio_policy_enabled"].all()

    def test_low_fitness_row_disabled(self) -> None:
        df = _make_df(n=5)
        df["ohio_fitness_trend_following"] = 0.0
        result = _gen.generate_dataframe(df)
        assert not result["ohio_policy_enabled"].any()

    def test_nan_transition_risk_treated_as_worst(self) -> None:
        df = _make_df(n=3)
        df["ohio_meta_transition_risk"] = np.nan
        df["ohio_fitness_trend_following"] = 1.0
        result_nan = _gen.generate_dataframe(df)

        df2 = _make_df(n=3)
        df2["ohio_meta_transition_risk"] = 1.0
        df2["ohio_fitness_trend_following"] = 1.0
        result_worst = _gen.generate_dataframe(df2)

        pd.testing.assert_series_equal(
            result_nan["ohio_policy_size_multiplier"].reset_index(drop=True),
            result_worst["ohio_policy_size_multiplier"].reset_index(drop=True),
        )

    def test_nan_confidence_treated_as_worst(self) -> None:
        df = _make_df(n=3)
        df["ohio_meta_confidence"] = np.nan
        df["ohio_fitness_trend_following"] = 1.0
        result_nan = _gen.generate_dataframe(df)

        df2 = _make_df(n=3)
        df2["ohio_meta_confidence"] = 0.0
        df2["ohio_fitness_trend_following"] = 1.0
        result_worst = _gen.generate_dataframe(df2)

        pd.testing.assert_series_equal(
            result_nan["ohio_policy_size_multiplier"].reset_index(drop=True),
            result_worst["ohio_policy_size_multiplier"].reset_index(drop=True),
        )

    def test_nan_data_mode_treated_as_degraded(self) -> None:
        df = _make_df(n=3)
        df["ohio_meta_data_mode"] = np.nan
        df["ohio_fitness_trend_following"] = 1.0
        result = _gen.generate_dataframe(df)
        assert not result["ohio_policy_enabled"].any()

    def test_mode_column_selects_correct_fitness(self) -> None:
        n = 4
        df = pd.DataFrame(
            {
                "ohio_fitness_trend_following": [0.9, 0.1, 0.1, 0.1],
                "ohio_fitness_mean_reversion": [0.1, 0.9, 0.1, 0.1],
                "ohio_fitness_breakout": [0.1, 0.1, 0.9, 0.1],
                "ohio_fitness_defensive": [0.1, 0.1, 0.1, 0.9],
                "ohio_active_mode": [
                    "trend_following",
                    "mean_reversion",
                    "breakout",
                    "defensive",
                ],
                "ohio_meta_transition_risk": [0.0] * n,
                "ohio_meta_confidence": [1.0] * n,
                "ohio_meta_data_mode": ["full"] * n,
            }
        )
        result = _gen.generate_dataframe(df)
        # All rows have high fitness for their active mode → all enabled
        assert result["ohio_policy_enabled"].all()
        # All sizes should be near max (fitness=0.9)
        expected_size = pytest.approx(0.3 + 1.2 * 0.9, abs=0.01)
        assert (result["ohio_policy_size_multiplier"] >= 1.3).all()


# ---------------------------------------------------------------------------
# 17. generate_dataframe preserves row count
# ---------------------------------------------------------------------------

class TestGenerateDataframeRowCount:
    @pytest.mark.parametrize("n", [1, 5, 50, 1000])
    def test_row_count_preserved(self, n: int) -> None:
        df = _make_df(n=n)
        result = _gen.generate_dataframe(df)
        assert len(result) == n

    def test_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame(columns=list(_make_df().columns))
        result = _gen.generate_dataframe(df)
        assert len(result) == 0

    def test_original_dataframe_unchanged(self) -> None:
        df = _make_df(n=10)
        original_cols = set(df.columns)
        _ = _gen.generate_dataframe(df)
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# 18. No Freqtrade imports in module (AST check)
# ---------------------------------------------------------------------------

class TestNoFreqtradeImports:
    _MODULE_PATH = (
        Path(__file__).parent.parent.parent
        / "freqtrade"
        / "ohio"
        / "core"
        / "strategy_router"
        / "policy_generator.py"
    )

    def test_module_file_exists(self) -> None:
        assert self._MODULE_PATH.exists(), (
            f"policy_generator.py not found at {self._MODULE_PATH}"
        )

    def test_no_freqtrade_framework_imports(self) -> None:
        source = self._MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden: list[str] = []
        allowed_prefix = "freqtrade.ohio.core.domain"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("freqtrade") and not alias.name.startswith(
                        allowed_prefix
                    ):
                        forbidden.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("freqtrade") and not module.startswith(allowed_prefix):
                    forbidden.append(module)

        assert not forbidden, (
            f"policy_generator.py must not import Freqtrade framework modules. "
            f"Found: {forbidden}"
        )


# ---------------------------------------------------------------------------
# Additional: custom threshold constructor
# ---------------------------------------------------------------------------

class TestCustomThreshold:
    def test_custom_threshold_respected(self) -> None:
        gen_strict = PolicyGenerator(disabled_threshold=0.5)
        # fitness=0.49 → disabled with strict threshold
        policy = gen_strict.generate(StrategyMode.BREAKOUT, 0.49, _clean_meta())
        assert policy.enabled is False

    def test_custom_threshold_above_value_enables(self) -> None:
        gen_lenient = PolicyGenerator(disabled_threshold=0.05)
        # fitness=0.10 → enabled with lenient threshold
        policy = gen_lenient.generate(StrategyMode.BREAKOUT, 0.10, _clean_meta())
        assert policy.enabled is True


# ---------------------------------------------------------------------------
# Additional: both meta adjustments combined
# ---------------------------------------------------------------------------

class TestCombinedMetaAdjustments:
    def test_both_reductions_applied_sequentially(self) -> None:
        # fitness=1 → raw size 1.5
        # after transition_risk reduction: 1.5 * 0.7 = 1.05
        # after confidence reduction: 1.05 * 0.8 = 0.84
        both_bad = _meta(transition_risk=0.9, confidence=0.3)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 1.0, both_bad)
        assert pytest.approx(policy.size_multiplier) == pytest.approx(0.84)

    def test_both_adjustments_with_clip_from_below(self) -> None:
        # fitness=0 → raw size 0.3; after reductions: 0.3 * 0.7 * 0.8 = 0.168 → clips to 0.3
        both_bad = _meta(transition_risk=0.9, confidence=0.3)
        policy = _gen.generate(StrategyMode.TREND_FOLLOWING, 0.0, both_bad)
        assert policy.size_multiplier == pytest.approx(0.3)
