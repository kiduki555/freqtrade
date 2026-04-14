"""Tests for _compute_hedge_weights helper."""
from __future__ import annotations

import numpy as np
import pytest

from freqtrade.ohio.core.strategy_router.fitness_estimator import (
    _compute_hedge_weights,
)


class TestHedgeWeights:

    def test_uniform_init_with_zero_eta(self):
        rewards = np.random.default_rng(42).uniform(-1, 1, (20, 4))
        weights = _compute_hedge_weights(rewards, eta=0.0, weight_floor=0.05)
        np.testing.assert_allclose(weights, 0.25, atol=1e-12)

    def test_weights_sum_to_one(self):
        rewards = np.random.default_rng(42).uniform(-1, 1, (50, 4))
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_positive_reward_increases_weight(self):
        n = 30
        rewards = np.zeros((n, 4))
        rewards[:, 0] = 0.5
        rewards[:, 1] = -0.5
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        assert weights[-1, 0] > weights[-1, 1]
        assert weights[-1, 0] > 0.25

    def test_weight_floor_prevents_mode_death(self):
        """Floor on raw weights prevents any mode from reaching exactly zero.

        With large negative cumulative rewards and weight_floor=0.0, mode 2
        would collapse to ~0.  With weight_floor > 0, the raw weight is clipped
        before normalisation so the normalised weight stays positive.
        """
        n = 100
        rewards = np.zeros((n, 4))
        rewards[:, 2] = -1.0
        # No floor: mode 2 approaches machine-zero
        weights_no_floor = _compute_hedge_weights(rewards, eta=0.5, weight_floor=0.0)
        # With floor: mode 2 raw weight is lifted, so normalised weight > 0
        weights_floored = _compute_hedge_weights(rewards, eta=0.5, weight_floor=0.05)
        assert weights_floored[-1, 2] > weights_no_floor[-1, 2]
        assert weights_floored[-1, 2] > 0.0

    def test_output_shape(self):
        rewards = np.random.default_rng(42).uniform(-1, 1, (15, 4))
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        assert weights.shape == (15, 4)

    def test_all_zero_rewards(self):
        rewards = np.zeros((10, 4))
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        np.testing.assert_allclose(weights, 0.25, atol=1e-12)
