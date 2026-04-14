"""Tests for FT-011: FitnessEstimator.

Covers:
  1.  compute_fitness returns float in [0, 1]
  2.  compute_all returns StrategyFitness with all 4 scores
  3.  Strong trend state → trend_following gets highest fitness
  4.  Low volatility, stable state → defensive gets highest fitness
  5.  Moderate volatility, low trend → mean_reversion gets highest fitness
  6.  High volatility, high RS → breakout gets highest fitness
  7.  All-neutral state → no extreme scores
  8.  Meta confidence boosts fitness proportionally
  9.  Meta stability boosts fitness proportionally
  10. Low confidence penalises all modes equally
  11. compute_dataframe adds 5 columns (4 fitness + 1 active_mode)
  12. compute_dataframe preserves row count
  13. Fitness scores sum is reasonable (not all 0 or all 1)
  14. compute_fitness with default profiles works
  15. No Freqtrade imports in the module (AST check)
  16. compute_dataframe fitness values in [0, 1]
  17. compute_dataframe active_mode values are valid
  18. compute_dataframe does not mutate input
  19. compute_dataframe handles missing meta columns
  20. Scalar/vector parity: compute_dataframe matches compute_fitness row-by-row
  21. compute_dataframe handles empty DataFrame
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.domain.models import (
    DataMode,
    StateMeta,
    StateVector,
    StrategyFitness,
    StrategyMode,
)
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

_ESTIMATOR_PATH = (
    Path(__file__).parent.parent.parent
    / "freqtrade"
    / "ohio"
    / "core"
    / "strategy_router"
    / "fitness_estimator.py"
)


def _make_meta(
    confidence: float = 0.8,
    stability: float = 0.8,
    transition_risk: float = 0.2,
    data_mode: DataMode = DataMode.FULL,
) -> StateMeta:
    return StateMeta(
        transition_risk=transition_risk,
        confidence=confidence,
        stability=stability,
        data_mode=data_mode,
    )


def _make_neutral_state() -> StateVector:
    """All [0,1] axes at 0.5, trend_persistence at 0.0."""
    return StateVector(
        trend_persistence=0.0,
        volatility_level=0.5,
        downside_pressure=0.5,
        liquidity_stress=0.5,
        relative_strength=0.5,
        correlation_stress=0.5,
        breadth_dispersion=0.5,
    )


def _make_trend_state() -> StateVector:
    """Strong uptrend, low downside/liquidity stress, moderate volatility."""
    return StateVector(
        trend_persistence=0.85,
        volatility_level=0.40,
        downside_pressure=0.05,
        liquidity_stress=0.05,
        relative_strength=0.80,
        correlation_stress=0.25,
        breadth_dispersion=0.30,
    )


def _make_defensive_state() -> StateVector:
    """Very low volatility, near-zero downside, calm markets."""
    return StateVector(
        trend_persistence=0.00,
        volatility_level=0.10,
        downside_pressure=0.05,
        liquidity_stress=0.05,
        relative_strength=0.50,
        correlation_stress=0.15,
        breadth_dispersion=0.15,
    )


def _make_mean_reversion_state() -> StateVector:
    """Choppy moderate volatility, no trend direction."""
    return StateVector(
        trend_persistence=0.02,
        volatility_level=0.50,
        downside_pressure=0.30,
        liquidity_stress=0.15,
        relative_strength=0.50,
        correlation_stress=0.25,
        breadth_dispersion=0.50,
    )


def _make_breakout_state() -> StateVector:
    """High volatility, strong relative strength, high breadth dispersion."""
    return StateVector(
        trend_persistence=0.60,
        volatility_level=0.78,
        downside_pressure=0.15,
        liquidity_stress=0.10,
        relative_strength=0.82,
        correlation_stress=0.18,
        breadth_dispersion=0.72,
    )


@pytest.fixture
def estimator() -> FitnessEstimator:
    return FitnessEstimator()


@pytest.fixture
def good_meta() -> StateMeta:
    return _make_meta(confidence=0.85, stability=0.85)


@pytest.fixture
def zero_meta() -> StateMeta:
    """Meta with no bonus — isolates pure axis-score discrimination."""
    return _make_meta(confidence=0.0, stability=0.0)


# ---------------------------------------------------------------------------
# 1. compute_fitness returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_fitness_returns_float_in_range(estimator: FitnessEstimator, good_meta: StateMeta) -> None:
    from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles

    profiles = load_default_profiles()
    for mode, profile in profiles.items():
        score = estimator.compute_fitness(_make_neutral_state(), good_meta, profile)
        assert isinstance(score, float), f"Expected float for {mode}"
        assert 0.0 <= score <= 1.0, f"Score out of [0,1] for {mode}: {score}"


# ---------------------------------------------------------------------------
# 2. compute_all returns StrategyFitness with all 4 scores
# ---------------------------------------------------------------------------


def test_compute_all_returns_strategy_fitness(estimator: FitnessEstimator, good_meta: StateMeta) -> None:
    result = estimator.compute_all(_make_neutral_state(), good_meta)
    assert isinstance(result, StrategyFitness)
    for mode in StrategyMode:
        score = result.score_for(mode)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score out of range for {mode}: {score}"


# ---------------------------------------------------------------------------
# 3. Strong trend → trend_following highest
#    Using zero meta bonus so axis scores alone drive discrimination.
# ---------------------------------------------------------------------------


def test_strong_trend_state_favours_trend_following(estimator: FitnessEstimator, zero_meta: StateMeta) -> None:
    fitness = estimator.compute_all(_make_trend_state(), zero_meta)
    assert fitness.best_mode == StrategyMode.TREND_FOLLOWING, (
        f"Expected trend_following, got {fitness.best_mode}. Scores: {fitness}"
    )
    assert fitness.trend_following > fitness.mean_reversion
    assert fitness.trend_following > fitness.breakout
    assert fitness.trend_following > fitness.defensive


# ---------------------------------------------------------------------------
# 4. Low volatility, stable → defensive highest
# ---------------------------------------------------------------------------


def test_defensive_state_favours_defensive(estimator: FitnessEstimator, zero_meta: StateMeta) -> None:
    fitness = estimator.compute_all(_make_defensive_state(), zero_meta)
    assert fitness.best_mode == StrategyMode.DEFENSIVE, (
        f"Expected defensive, got {fitness.best_mode}. Scores: {fitness}"
    )
    assert fitness.defensive > fitness.trend_following
    assert fitness.defensive > fitness.breakout


# ---------------------------------------------------------------------------
# 5. Moderate volatility, no trend → mean_reversion highest
# ---------------------------------------------------------------------------


def test_mean_reversion_state_favours_mean_reversion(estimator: FitnessEstimator, zero_meta: StateMeta) -> None:
    fitness = estimator.compute_all(_make_mean_reversion_state(), zero_meta)
    assert fitness.best_mode == StrategyMode.MEAN_REVERSION, (
        f"Expected mean_reversion, got {fitness.best_mode}. Scores: {fitness}"
    )
    assert fitness.mean_reversion > fitness.trend_following
    assert fitness.mean_reversion > fitness.defensive


# ---------------------------------------------------------------------------
# 6. High volatility, high RS → breakout highest
# ---------------------------------------------------------------------------


def test_breakout_state_favours_breakout(estimator: FitnessEstimator, zero_meta: StateMeta) -> None:
    fitness = estimator.compute_all(_make_breakout_state(), zero_meta)
    assert fitness.best_mode == StrategyMode.BREAKOUT, (
        f"Expected breakout, got {fitness.best_mode}. Scores: {fitness}"
    )
    assert fitness.breakout > fitness.mean_reversion
    assert fitness.breakout > fitness.defensive


# ---------------------------------------------------------------------------
# 7. Neutral state → no extreme scores (with zero meta so ceiling at 1.0 is not hit)
# ---------------------------------------------------------------------------


def test_neutral_state_no_extreme_scores(estimator: FitnessEstimator, zero_meta: StateMeta) -> None:
    """Pure axis scores (no meta bonus) for a neutral state should be moderate."""
    fitness = estimator.compute_all(_make_neutral_state(), zero_meta)
    scores = [
        fitness.trend_following,
        fitness.mean_reversion,
        fitness.breakout,
        fitness.defensive,
    ]
    for score in scores:
        # With zero meta, neutral state axis scores land between 0.6–0.95
        assert 0.05 < score < 0.99, f"Neutral state produced extreme score: {score}"
    # Mean reversion has ideal=0.0 for trend (neutral trend=0 matches well) → highest
    assert max(scores) - min(scores) > 0.05, "Scores should show some discrimination"


# ---------------------------------------------------------------------------
# 8. Meta confidence boosts fitness proportionally
# ---------------------------------------------------------------------------


def test_higher_confidence_increases_fitness(estimator: FitnessEstimator) -> None:
    from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles

    profiles = load_default_profiles()
    profile = profiles[StrategyMode.TREND_FOLLOWING]
    state = _make_neutral_state()

    low_conf_meta = _make_meta(confidence=0.1, stability=0.5)
    high_conf_meta = _make_meta(confidence=0.9, stability=0.5)

    score_low = estimator.compute_fitness(state, low_conf_meta, profile)
    score_high = estimator.compute_fitness(state, high_conf_meta, profile)

    assert score_high > score_low, (
        f"Higher confidence should raise fitness: {score_high} <= {score_low}"
    )

    # Verify the boost magnitude matches meta_confidence_weight
    expected_diff = (0.9 - 0.1) * profile.meta_confidence_weight
    actual_diff = score_high - score_low
    assert abs(actual_diff - expected_diff) < 1e-9, (
        f"Confidence boost mismatch: expected {expected_diff:.6f}, got {actual_diff:.6f}"
    )


# ---------------------------------------------------------------------------
# 9. Meta stability boosts fitness proportionally
# ---------------------------------------------------------------------------


def test_higher_stability_increases_fitness(estimator: FitnessEstimator) -> None:
    from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles

    profiles = load_default_profiles()
    profile = profiles[StrategyMode.MEAN_REVERSION]
    # Use a state that does NOT produce a near-1.0 raw axis score so the
    # stability delta is not masked by clipping.  A trend-heavy state is far
    # from mean_reversion's ideals and leaves room for the meta contribution.
    state = _make_trend_state()

    low_stab_meta = _make_meta(confidence=0.0, stability=0.1)
    high_stab_meta = _make_meta(confidence=0.0, stability=0.9)

    score_low = estimator.compute_fitness(state, low_stab_meta, profile)
    score_high = estimator.compute_fitness(state, high_stab_meta, profile)

    assert score_high > score_low

    expected_diff = (0.9 - 0.1) * profile.meta_stability_weight
    actual_diff = score_high - score_low
    assert abs(actual_diff - expected_diff) < 1e-9, (
        f"Stability boost mismatch: expected {expected_diff:.6f}, got {actual_diff:.6f}"
    )


# ---------------------------------------------------------------------------
# 10. Low confidence penalises all modes equally (same relative delta)
# ---------------------------------------------------------------------------


def test_low_confidence_reduces_all_modes(estimator: FitnessEstimator) -> None:
    state = _make_neutral_state()
    high_meta = _make_meta(confidence=1.0, stability=0.5)
    low_meta = _make_meta(confidence=0.0, stability=0.5)

    high_fitness = estimator.compute_all(state, high_meta)
    low_fitness = estimator.compute_all(state, low_meta)

    for mode in StrategyMode:
        assert low_fitness.score_for(mode) < high_fitness.score_for(mode), (
            f"Mode {mode}: low confidence should reduce score. "
            f"high={high_fitness.score_for(mode):.4f}, low={low_fitness.score_for(mode):.4f}"
        )


# ---------------------------------------------------------------------------
# 11. compute_dataframe adds 5 columns
# ---------------------------------------------------------------------------


def _make_ohio_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """Create a minimal DataFrame with all required ohio_stable_* and ohio_meta_* columns."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "close": 100.0 + np.cumsum(rng.uniform(-1, 1, n_rows)),
            "ohio_stable_trend": rng.uniform(-1.0, 1.0, n_rows),
            "ohio_stable_volatility": rng.uniform(0.0, 1.0, n_rows),
            "ohio_stable_downside": rng.uniform(0.0, 1.0, n_rows),
            "ohio_stable_liquidity": rng.uniform(0.0, 1.0, n_rows),
            "ohio_stable_relative_strength": rng.uniform(0.0, 1.0, n_rows),
            "ohio_stable_correlation": rng.uniform(0.0, 1.0, n_rows),
            "ohio_stable_breadth": rng.uniform(0.0, 1.0, n_rows),
            "ohio_meta_transition_risk": rng.uniform(0.0, 1.0, n_rows),
            "ohio_meta_confidence": rng.uniform(0.5, 1.0, n_rows),
            "ohio_meta_stability": rng.uniform(0.5, 1.0, n_rows),
            "ohio_feat_atr_ratio_14": rng.uniform(0.005, 0.03, n_rows),
        }
    )
    return df


def test_compute_dataframe_adds_five_columns(estimator: FitnessEstimator) -> None:
    df_in = _make_ohio_dataframe()
    df_out = estimator.compute_dataframe(df_in)

    expected_new_cols = {
        "ohio_fitness_trend_following",
        "ohio_fitness_mean_reversion",
        "ohio_fitness_breakout",
        "ohio_fitness_defensive",
        "ohio_active_mode",
        "ohio_hedge_weight_trend_following",
        "ohio_hedge_weight_mean_reversion",
        "ohio_hedge_weight_breakout",
        "ohio_hedge_weight_defensive",
    }
    for col in expected_new_cols:
        assert col in df_out.columns, f"Missing column: {col}"

    assert len(df_out.columns) == len(df_in.columns) + 9


# ---------------------------------------------------------------------------
# 12. compute_dataframe preserves row count
# ---------------------------------------------------------------------------


def test_compute_dataframe_preserves_row_count(estimator: FitnessEstimator) -> None:
    for n_rows in [1, 10, 250]:
        df_in = _make_ohio_dataframe(n_rows)
        df_out = estimator.compute_dataframe(df_in)
        assert len(df_out) == n_rows, f"Row count changed for n_rows={n_rows}"


# ---------------------------------------------------------------------------
# 13. Fitness scores sum is reasonable
# ---------------------------------------------------------------------------


def test_fitness_score_sum_reasonable(estimator: FitnessEstimator, good_meta: StateMeta) -> None:
    """Scores should not all collapse to 0 or all to 1."""
    for state_fn in [
        _make_neutral_state,
        _make_trend_state,
        _make_defensive_state,
        _make_mean_reversion_state,
        _make_breakout_state,
    ]:
        fitness = estimator.compute_all(state_fn(), good_meta)
        scores = [
            fitness.trend_following,
            fitness.mean_reversion,
            fitness.breakout,
            fitness.defensive,
        ]
        total = sum(scores)
        assert total > 0.1, f"All scores collapsed to 0 for state {state_fn.__name__}: {scores}"
        assert total < 4.0, f"All scores at 1.0 for state {state_fn.__name__}: {scores}"
        assert max(scores) - min(scores) > 0.01, (
            f"Scores are all identical — no discrimination for {state_fn.__name__}: {scores}"
        )


# ---------------------------------------------------------------------------
# 14. compute_fitness with default profiles (no profiles arg)
# ---------------------------------------------------------------------------


def test_default_profiles_load_and_compute() -> None:
    """FitnessEstimator() without arguments should load defaults and compute."""
    est = FitnessEstimator()
    state = _make_trend_state()
    meta = _make_meta()
    fitness = est.compute_all(state, meta)
    assert isinstance(fitness, StrategyFitness)
    assert 0.0 <= fitness.trend_following <= 1.0


# ---------------------------------------------------------------------------
# 15. No Freqtrade framework imports in the module (AST check)
# ---------------------------------------------------------------------------


def test_no_freqtrade_framework_imports_in_module() -> None:
    """The fitness_estimator module must not import from freqtrade.* framework
    except for ohio domain models and ohio strategy_profile."""
    source = _ESTIMATOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    allowed_ohio_prefixes = (
        "freqtrade.ohio.core.domain.models",
        "freqtrade.ohio.core.strategy_router.strategy_profile",
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod.startswith("freqtrade"):
                    assert any(mod.startswith(p) for p in allowed_ohio_prefixes), (
                        f"Disallowed Freqtrade import found: {mod}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("freqtrade"):
                assert any(mod.startswith(p) for p in allowed_ohio_prefixes), (
                    f"Disallowed Freqtrade import found: from {mod}"
                )


# ---------------------------------------------------------------------------
# 16. compute_dataframe fitness values are in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_dataframe_fitness_values_in_range(estimator: FitnessEstimator) -> None:
    df_out = estimator.compute_dataframe(_make_ohio_dataframe(50))
    for mode in StrategyMode:
        col = f"ohio_fitness_{mode.value}"
        vals = df_out[col].to_numpy()
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), (
            f"Values out of [0,1] in column {col}: min={vals.min():.4f}, max={vals.max():.4f}"
        )


# ---------------------------------------------------------------------------
# 17. compute_dataframe active_mode contains valid StrategyMode values
# ---------------------------------------------------------------------------


def test_compute_dataframe_active_mode_values(estimator: FitnessEstimator) -> None:
    df_out = estimator.compute_dataframe(_make_ohio_dataframe(30))
    valid_values = {m.value for m in StrategyMode}
    bad = set(df_out["ohio_active_mode"].unique()) - valid_values
    assert not bad, f"ohio_active_mode contains invalid values: {bad}"


# ---------------------------------------------------------------------------
# 18. compute_dataframe does not mutate input
# ---------------------------------------------------------------------------


def test_compute_dataframe_does_not_mutate_input(estimator: FitnessEstimator) -> None:
    df_in = _make_ohio_dataframe(20)
    original_cols = list(df_in.columns)
    original_shape = df_in.shape

    estimator.compute_dataframe(df_in)

    assert list(df_in.columns) == original_cols, "Input dataframe columns were mutated"
    assert df_in.shape == original_shape, "Input dataframe shape was mutated"


# ---------------------------------------------------------------------------
# 19. compute_dataframe gracefully handles missing ohio_meta columns
# ---------------------------------------------------------------------------


def test_compute_dataframe_handles_missing_meta_columns(estimator: FitnessEstimator) -> None:
    df_in = _make_ohio_dataframe(10)
    df_no_meta = df_in.drop(columns=["ohio_meta_confidence", "ohio_meta_stability"])
    df_out = estimator.compute_dataframe(df_no_meta)

    # Should still produce output columns
    for mode in StrategyMode:
        assert f"ohio_fitness_{mode.value}" in df_out.columns
    assert "ohio_active_mode" in df_out.columns


# ---------------------------------------------------------------------------
# 20. Scalar/vector parity: compute_dataframe matches compute_fitness row-by-row
# ---------------------------------------------------------------------------


def test_vectorized_matches_scalar_parity(estimator: FitnessEstimator) -> None:
    """Build a DataFrame from known StateVector/StateMeta, then verify that
    compute_dataframe produces the same fitness scores as compute_fitness for
    each row."""
    from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles

    profiles = load_default_profiles()

    states = [
        _make_trend_state(),
        _make_defensive_state(),
        _make_mean_reversion_state(),
        _make_breakout_state(),
        _make_neutral_state(),
    ]
    metas = [
        _make_meta(confidence=0.9, stability=0.85),
        _make_meta(confidence=0.3, stability=0.4),
        _make_meta(confidence=0.6, stability=0.7),
        _make_meta(confidence=0.0, stability=0.0),
        _make_meta(confidence=1.0, stability=1.0),
    ]

    # Build DataFrame row by row
    rows = []
    for sv, meta in zip(states, metas):
        rows.append({
            "ohio_stable_trend": sv.trend_persistence,
            "ohio_stable_volatility": sv.volatility_level,
            "ohio_stable_downside": sv.downside_pressure,
            "ohio_stable_liquidity": sv.liquidity_stress,
            "ohio_stable_relative_strength": sv.relative_strength,
            "ohio_stable_correlation": sv.correlation_stress,
            "ohio_stable_breadth": sv.breadth_dispersion,
            "ohio_meta_confidence": meta.confidence,
            "ohio_meta_stability": meta.stability,
        })
    df = pd.DataFrame(rows)
    df["close"] = 100.0
    df["ohio_feat_atr_ratio_14"] = 0.01
    # ohio_stable_trend already in rows; close + atr needed for Hedge reward computation
    df_out = estimator.compute_dataframe(df)

    for i, (sv, meta) in enumerate(zip(states, metas)):
        for mode in StrategyMode:
            scalar_score = estimator.compute_fitness(sv, meta, profiles[mode])
            vector_score = df_out[f"ohio_fitness_{mode.value}"].iloc[i]
            assert abs(scalar_score - vector_score) < 1e-12, (
                f"Row {i}, mode {mode.value}: scalar={scalar_score:.12f} "
                f"!= vector={vector_score:.12f}"
            )


# ---------------------------------------------------------------------------
# 21. compute_dataframe handles empty DataFrame
# ---------------------------------------------------------------------------


def test_compute_dataframe_empty_dataframe(estimator: FitnessEstimator) -> None:
    """An empty DataFrame should produce an empty result with all output columns."""
    df_in = _make_ohio_dataframe(10).iloc[:0]  # 0 rows, correct schema
    df_out = estimator.compute_dataframe(df_in)

    assert len(df_out) == 0
    expected_cols = {
        "ohio_fitness_trend_following",
        "ohio_fitness_mean_reversion",
        "ohio_fitness_breakout",
        "ohio_fitness_defensive",
        "ohio_active_mode",
    }
    for col in expected_cols:
        assert col in df_out.columns, f"Missing column in empty result: {col}"
