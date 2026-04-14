"""Tests for Hedge-based mode selection in FitnessEstimator.

Tests reward computation, Hedge weight updates, temperature flattening,
and integration with compute_dataframe.
"""
from __future__ import annotations

import numpy as np
import pytest

from freqtrade.ohio.core.strategy_router.fitness_estimator import (
    _compute_rewards,
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
