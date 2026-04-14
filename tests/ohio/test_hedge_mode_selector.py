"""Tests for Hedge-based mode selection in FitnessEstimator.

Tests reward computation, Hedge weight updates, temperature flattening,
and integration with compute_dataframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.strategy_router.fitness_estimator import (
    _compute_rewards,
    FitnessEstimator,
)


def _make_arrays(n: int = 10, seed: int = 42):
    """Create test arrays for reward computation."""
    rng = np.random.default_rng(seed)
    ret = rng.uniform(-0.03, 0.03, n)
    trend = rng.uniform(-1.0, 1.0, n)
    atr = rng.uniform(0.005, 0.03, n)
    return ret, trend, atr


class TestRewardComputation:
    """Tests for _compute_rewards helper."""

    def test_reward_shapes(self):
        ret, trend, atr = _make_arrays(20)
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards.shape == (20, 4)

    def test_reward_tf_positive_when_trend_continues(self):
        ret = np.array([0.02])
        trend = np.array([0.8])
        atr = np.array([0.01])
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards[0, 0] > 0, f"TF reward should be positive: {rewards[0, 0]}"

    def test_reward_mr_positive_when_trend_reverses(self):
        ret = np.array([-0.02])
        trend = np.array([0.8])
        atr = np.array([0.01])
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards[0, 1] > 0, f"MR reward should be positive: {rewards[0, 1]}"

    def test_reward_tf_mr_mirror(self):
        ret, trend, atr = _make_arrays(50)
        rewards = _compute_rewards(ret, trend, atr)
        np.testing.assert_allclose(rewards[:, 0], -rewards[:, 1], atol=1e-12)

    def test_reward_bo_def_mirror(self):
        ret, trend, atr = _make_arrays(50)
        rewards = _compute_rewards(ret, trend, atr)
        np.testing.assert_allclose(rewards[:, 2], -rewards[:, 3], atol=1e-12)

    def test_reward_clipped_to_minus1_plus1(self):
        ret = np.array([0.50])
        trend = np.array([1.0])
        atr = np.array([0.001])
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards.min() >= -1.0
        assert rewards.max() <= 1.0

    def test_reward_nan_return_becomes_zero(self):
        # When ret=NaN → safe_ret=0: TF and MR rewards are 0 (no return signal).
        # BO/DEF depend on |ret| vs ATR: with ret=0, |ret|<ATR → BO=-1, DEF=+1.
        ret = np.array([np.nan, 0.01])
        trend = np.array([0.5, 0.5])
        atr = np.array([0.01, 0.01])
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards[0, 0] == 0.0, "TF reward should be 0 when ret=NaN"
        assert rewards[0, 1] == 0.0, "MR reward should be 0 when ret=NaN"

    def test_reward_atr_normalization(self):
        ret = np.array([0.02, 0.02])
        trend = np.array([0.5, 0.5])
        atr_low = np.array([0.01, 0.01])
        atr_high = np.array([0.04, 0.04])
        rewards_low = _compute_rewards(ret, trend, atr_low)
        rewards_high = _compute_rewards(ret, trend, atr_high)
        assert abs(rewards_low[0, 0]) > abs(rewards_high[0, 0])


def _make_ohio_dataframe(n_rows: int = 50, seed: int = 42) -> pd.DataFrame:
    """DataFrame with ohio_stable_*, ohio_meta_*, and close columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "close": 100.0 + np.cumsum(rng.uniform(-1, 1, n_rows)),
        "ohio_stable_trend": rng.uniform(-1.0, 1.0, n_rows),
        "ohio_stable_volatility": rng.uniform(0.0, 1.0, n_rows),
        "ohio_stable_downside": rng.uniform(0.0, 1.0, n_rows),
        "ohio_stable_liquidity": rng.uniform(0.0, 1.0, n_rows),
        "ohio_stable_relative_strength": rng.uniform(0.0, 1.0, n_rows),
        "ohio_stable_correlation": rng.uniform(0.0, 1.0, n_rows),
        "ohio_stable_breadth": rng.uniform(0.0, 1.0, n_rows),
        "ohio_meta_confidence": rng.uniform(0.5, 1.0, n_rows),
        "ohio_meta_stability": rng.uniform(0.5, 1.0, n_rows),
        "ohio_feat_atr_ratio_14": rng.uniform(0.005, 0.03, n_rows),
    })


class TestHedgeIntegration:

    def test_hedge_columns_exist(self):
        est = FitnessEstimator(hedge_eta=0.1)
        df_out = est.compute_dataframe(_make_ohio_dataframe())
        from freqtrade.ohio.core.domain.models import StrategyMode
        for mode in StrategyMode:
            col = f"ohio_hedge_weight_{mode.value}"
            assert col in df_out.columns, f"Missing column: {col}"

    def test_hedge_weights_sum_to_one(self):
        from freqtrade.ohio.core.domain.models import StrategyMode
        est = FitnessEstimator(hedge_eta=0.1)
        df_out = est.compute_dataframe(_make_ohio_dataframe())
        weight_cols = [f"ohio_hedge_weight_{m.value}" for m in StrategyMode]
        row_sums = df_out[weight_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_backward_compat_eta_zero_temp_one(self):
        """eta=0, temperature=1 → identical active_mode to pure argmax."""
        from freqtrade.ohio.core.domain.models import StrategyMode
        df = _make_ohio_dataframe(100)
        est = FitnessEstimator(hedge_eta=0.0, hedge_temperature=1.0)
        df_out = est.compute_dataframe(df)

        # Compute expected active_mode via pure argmax on raw fitness
        score_cols = [f"ohio_fitness_{m.value}" for m in StrategyMode]
        mode_values = [m.value for m in StrategyMode]
        expected = df_out[score_cols].values.argmax(axis=1)
        expected_modes = [mode_values[i] for i in expected]
        actual_modes = df_out["ohio_active_mode"].tolist()
        assert actual_modes == expected_modes

    def test_mr_dominance_broken(self):
        """With trending data + Hedge, MR should not be 96%+."""
        from freqtrade.ohio.core.domain.models import StrategyMode
        n = 200
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "close": 100.0 + np.cumsum(np.abs(rng.normal(0.5, 0.1, n))),
            "ohio_stable_trend": np.full(n, 0.1),         # near-neutral → MR wins raw fitness
            "ohio_stable_volatility": np.full(n, 0.5),
            "ohio_stable_downside": np.full(n, 0.3),
            "ohio_stable_liquidity": np.full(n, 0.15),
            "ohio_stable_relative_strength": np.full(n, 0.5),
            "ohio_stable_correlation": np.full(n, 0.25),
            "ohio_stable_breadth": np.full(n, 0.5),
            "ohio_meta_confidence": np.full(n, 0.8),
            "ohio_meta_stability": np.full(n, 0.8),
            "ohio_feat_atr_ratio_14": np.full(n, 0.01),
        })
        est = FitnessEstimator(hedge_eta=0.1, hedge_temperature=2.0)
        df_out = est.compute_dataframe(df)
        mode_counts = df_out["ohio_active_mode"].value_counts(normalize=True)
        mr_pct = mode_counts.get("mean_reversion", 0.0)
        assert mr_pct < 0.96, f"MR still at {mr_pct:.1%}. Distribution: {mode_counts.to_dict()}"

    def test_existing_fitness_columns_unchanged(self):
        """ohio_fitness_* columns are raw fitness, not adjusted."""
        from freqtrade.ohio.core.domain.models import StrategyMode
        df = _make_ohio_dataframe()
        df_hedge = FitnessEstimator(hedge_eta=0.1, hedge_temperature=2.0).compute_dataframe(df)
        df_raw = FitnessEstimator(hedge_eta=0.0, hedge_temperature=1.0).compute_dataframe(df)
        for mode in StrategyMode:
            col = f"ohio_fitness_{mode.value}"
            pd.testing.assert_series_equal(df_hedge[col], df_raw[col], check_names=False)

    def test_warmup_nan_handling(self):
        """NaN in first-row ret is handled without error.

        At row 0, close.pct_change() is NaN → safe_ret=0 in _compute_rewards.
        Weights at row 0 are still valid (sum=1, each >= floor).
        """
        floor = 0.05
        est = FitnessEstimator(hedge_eta=0.1, hedge_weight_floor=floor)
        df = _make_ohio_dataframe(5)
        df_out = est.compute_dataframe(df)
        from freqtrade.ohio.core.domain.models import StrategyMode
        weight_cols = [f"ohio_hedge_weight_{m.value}" for m in StrategyMode]
        row0 = df_out[weight_cols].iloc[0]
        assert not row0.isna().any(), "First-row weights must not be NaN"
        assert abs(row0.sum() - 1.0) < 1e-10, "First-row weights must sum to 1"
        assert (row0 > 0).all(), "All first-row weights must be positive"

    def test_active_mode_still_valid(self):
        from freqtrade.ohio.core.domain.models import StrategyMode
        est = FitnessEstimator(hedge_eta=0.1, hedge_temperature=2.0)
        df_out = est.compute_dataframe(_make_ohio_dataframe(100))
        valid = {m.value for m in StrategyMode}
        bad = set(df_out["ohio_active_mode"].unique()) - valid
        assert not bad, f"Invalid: {bad}"

    def test_does_not_mutate_input(self):
        est = FitnessEstimator(hedge_eta=0.1)
        df = _make_ohio_dataframe()
        original_cols = list(df.columns)
        est.compute_dataframe(df)
        assert list(df.columns) == original_cols
