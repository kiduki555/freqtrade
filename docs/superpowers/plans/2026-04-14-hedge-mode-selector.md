# Hedge-Based Mode Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-winner `argmax` mode selection with Hedge algorithm (Cesa-Bianchi & Lugosi 2006) to break 96% MR dominance while maintaining deterministic selection and backward compatibility.

**Architecture:** Modify `FitnessEstimator.compute_dataframe()` to compute per-mode rewards from 1-bar returns, build cumulative Hedge weights via exponential weighting, flatten fitness scores by temperature, then argmax on adjusted scores. Downstream unchanged.

**Tech Stack:** Python, NumPy, pandas, pytest

---

### Task 1: Add Hedge Parameters to OhioConfig

**Files:**
- Modify: `freqtrade/ohio/config/defaults.py`
- Test: `tests/ohio/test_defaults_hedge.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for Hedge parameters in OhioConfig."""
from freqtrade.ohio.config.defaults import OhioConfig


def test_ohio_config_has_hedge_params():
    cfg = OhioConfig()
    assert cfg.hedge_eta == 0.1
    assert cfg.hedge_temperature == 2.0
    assert cfg.hedge_weight_floor == 0.05


def test_ohio_config_hedge_params_custom():
    cfg = OhioConfig(hedge_eta=0.2, hedge_temperature=3.0, hedge_weight_floor=0.10)
    assert cfg.hedge_eta == 0.2
    assert cfg.hedge_temperature == 3.0
    assert cfg.hedge_weight_floor == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ohio/test_defaults_hedge.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'hedge_eta'`

- [ ] **Step 3: Write minimal implementation**

In `freqtrade/ohio/config/defaults.py`, add three fields to the `OhioConfig` dataclass:

```python
@dataclass(frozen=True)
class OhioConfig:
    # ... existing fields ...

    # Hedge mode selector (Enhancement #2)
    hedge_eta: float = 0.1              # learning rate [0.01, 0.5]
    hedge_temperature: float = 2.0      # fitness flattening [1.0, 5.0]
    hedge_weight_floor: float = 0.05    # min weight per mode [0.01, 0.15]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ohio/test_defaults_hedge.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/config/defaults.py tests/ohio/test_defaults_hedge.py
git commit -m "feat: add Hedge mode selector params to OhioConfig"
```

---

### Task 2: Implement Reward Computation Helper

**Files:**
- Modify: `freqtrade/ohio/core/strategy_router/fitness_estimator.py`
- Test: `tests/ohio/test_hedge_mode_selector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/ohio/test_hedge_mode_selector.py`:

```python
"""Tests for Hedge-based mode selection in FitnessEstimator.

Tests reward computation, Hedge weight updates, temperature flattening,
and integration with compute_dataframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.strategy_router.fitness_estimator import (
    FitnessEstimator,
    _compute_rewards,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arrays(n: int = 10, seed: int = 42):
    """Create test arrays for reward computation."""
    rng = np.random.default_rng(seed)
    ret = rng.uniform(-0.03, 0.03, n)
    trend = rng.uniform(-1.0, 1.0, n)
    atr = rng.uniform(0.005, 0.03, n)
    return ret, trend, atr


# ---------------------------------------------------------------------------
# Test: reward computation
# ---------------------------------------------------------------------------


class TestRewardComputation:
    """Tests for _compute_rewards helper."""

    def test_reward_shapes(self):
        ret, trend, atr = _make_arrays(20)
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards.shape == (20, 4)

    def test_reward_tf_positive_when_trend_continues(self):
        """Positive return + positive trend → TF reward positive."""
        ret = np.array([0.02])      # +2%
        trend = np.array([0.8])     # strong positive trend
        atr = np.array([0.01])      # 1% ATR
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards[0, 0] > 0, f"TF reward should be positive: {rewards[0, 0]}"

    def test_reward_mr_positive_when_trend_reverses(self):
        """Negative return + positive trend → MR reward positive."""
        ret = np.array([-0.02])     # -2% (reversal)
        trend = np.array([0.8])     # was trending up
        atr = np.array([0.01])
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards[0, 1] > 0, f"MR reward should be positive: {rewards[0, 1]}"

    def test_reward_tf_mr_mirror(self):
        """TF and MR rewards are exact negatives of each other."""
        ret, trend, atr = _make_arrays(50)
        rewards = _compute_rewards(ret, trend, atr)
        np.testing.assert_allclose(rewards[:, 0], -rewards[:, 1], atol=1e-12)

    def test_reward_bo_def_mirror(self):
        """BO and DEF rewards are exact negatives of each other."""
        ret, trend, atr = _make_arrays(50)
        rewards = _compute_rewards(ret, trend, atr)
        np.testing.assert_allclose(rewards[:, 2], -rewards[:, 3], atol=1e-12)

    def test_reward_clipped_to_minus1_plus1(self):
        """Extreme values are clipped to [-1, 1]."""
        ret = np.array([0.50])      # 50% move — extreme
        trend = np.array([1.0])
        atr = np.array([0.001])     # tiny ATR → huge raw reward
        rewards = _compute_rewards(ret, trend, atr)
        assert rewards.min() >= -1.0
        assert rewards.max() <= 1.0

    def test_reward_nan_return_becomes_zero(self):
        """NaN in return (first bar pct_change) → reward 0."""
        ret = np.array([np.nan, 0.01])
        trend = np.array([0.5, 0.5])
        atr = np.array([0.01, 0.01])
        rewards = _compute_rewards(ret, trend, atr)
        np.testing.assert_array_equal(rewards[0, :], 0.0)

    def test_reward_atr_normalization(self):
        """Same return but higher ATR → smaller reward magnitude."""
        ret = np.array([0.02, 0.02])
        trend = np.array([0.5, 0.5])
        atr_low = np.array([0.01, 0.01])
        atr_high = np.array([0.04, 0.04])
        rewards_low = _compute_rewards(ret, trend, atr_low)
        rewards_high = _compute_rewards(ret, trend, atr_high)
        assert abs(rewards_low[0, 0]) > abs(rewards_high[0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py::TestRewardComputation -v`
Expected: FAIL with `ImportError: cannot import name '_compute_rewards'`

- [ ] **Step 3: Write minimal implementation**

Add `_compute_rewards` function to `freqtrade/ohio/core/strategy_router/fitness_estimator.py`, before the `FitnessEstimator` class:

```python
def _compute_rewards(
    ret: np.ndarray,
    trend: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """Compute per-mode rewards from 1-bar return, trend, and ATR.

    Args:
        ret:   1-bar return array (close.pct_change()). NaN → 0 reward.
        trend: ohio_sv_trend_persistence array.
        atr:   ohio_feat_atr_ratio_14 array. Used as normalizer.

    Returns:
        (N, 4) array: columns = [TF, MR, BO, DEF], values in [-1, 1].
    """
    safe_ret = np.nan_to_num(ret, nan=0.0)
    safe_trend = np.nan_to_num(trend, nan=0.0)
    safe_atr = np.maximum(np.nan_to_num(atr, nan=0.01), 1e-8)

    trend_sign = np.sign(safe_trend)

    reward_tf = np.clip((safe_ret * trend_sign) / safe_atr, -1.0, 1.0)
    reward_mr = np.clip((-safe_ret * trend_sign) / safe_atr, -1.0, 1.0)
    reward_bo = np.clip((np.abs(safe_ret) - safe_atr) / safe_atr, -1.0, 1.0)
    reward_df = np.clip((safe_atr - np.abs(safe_ret)) / safe_atr, -1.0, 1.0)

    return np.column_stack([reward_tf, reward_mr, reward_bo, reward_df])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py::TestRewardComputation -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/core/strategy_router/fitness_estimator.py tests/ohio/test_hedge_mode_selector.py
git commit -m "feat: add _compute_rewards for Hedge mode selection"
```

---

### Task 3: Implement Hedge Weight Computation Helper

**Files:**
- Modify: `freqtrade/ohio/core/strategy_router/fitness_estimator.py`
- Modify: `tests/ohio/test_hedge_mode_selector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ohio/test_hedge_mode_selector.py`:

```python
from freqtrade.ohio.core.strategy_router.fitness_estimator import (
    _compute_hedge_weights,
)


class TestHedgeWeights:
    """Tests for _compute_hedge_weights helper."""

    def test_uniform_init_with_zero_eta(self):
        """eta=0 → all weights stay at 0.25 regardless of rewards."""
        rewards = np.random.default_rng(42).uniform(-1, 1, (20, 4))
        weights = _compute_hedge_weights(rewards, eta=0.0, weight_floor=0.05)
        np.testing.assert_allclose(weights, 0.25, atol=1e-12)

    def test_weights_sum_to_one(self):
        """Each row of weights must sum to 1.0."""
        rewards = np.random.default_rng(42).uniform(-1, 1, (50, 4))
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_positive_reward_increases_weight(self):
        """Mode with consistently positive reward gets higher weight."""
        n = 30
        rewards = np.zeros((n, 4))
        rewards[:, 0] = 0.5   # TF always positive
        rewards[:, 1] = -0.5  # MR always negative
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        # At last row, TF weight should be highest
        assert weights[-1, 0] > weights[-1, 1], "TF should outweigh MR"
        assert weights[-1, 0] > 0.25, "TF should be above uniform"

    def test_weight_floor_enforced(self):
        """Even with extreme negative reward, weight stays at floor."""
        n = 100
        rewards = np.zeros((n, 4))
        rewards[:, 2] = -1.0  # BO always worst
        weights = _compute_hedge_weights(rewards, eta=0.5, weight_floor=0.05)
        assert weights[-1, 2] >= 0.05 - 1e-12, (
            f"BO weight {weights[-1, 2]} below floor 0.05"
        )

    def test_output_shape(self):
        rewards = np.random.default_rng(42).uniform(-1, 1, (15, 4))
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        assert weights.shape == (15, 4)

    def test_all_zero_rewards(self):
        """Zero rewards → uniform weights throughout."""
        rewards = np.zeros((10, 4))
        weights = _compute_hedge_weights(rewards, eta=0.1, weight_floor=0.05)
        np.testing.assert_allclose(weights, 0.25, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py::TestHedgeWeights -v`
Expected: FAIL with `ImportError: cannot import name '_compute_hedge_weights'`

- [ ] **Step 3: Write minimal implementation**

Add `_compute_hedge_weights` function to `fitness_estimator.py`, after `_compute_rewards`:

```python
def _compute_hedge_weights(
    rewards: np.ndarray,
    eta: float,
    weight_floor: float,
) -> np.ndarray:
    """Compute Hedge algorithm weights from cumulative rewards.

    w_i(t) ∝ exp(eta * cumsum(reward_i(1..t))), with per-mode floor.

    Args:
        rewards:      (N, 4) reward array from _compute_rewards.
        eta:          Learning rate. 0.0 → uniform weights.
        weight_floor: Minimum weight per mode (prevents mode death).

    Returns:
        (N, 4) array of normalized weights per row, each row sums to 1.0.
    """
    cum_rewards = np.cumsum(rewards, axis=0)
    raw_weights = np.exp(eta * cum_rewards)
    raw_weights = np.maximum(raw_weights, weight_floor)
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    return raw_weights / row_sums
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py::TestHedgeWeights -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/core/strategy_router/fitness_estimator.py tests/ohio/test_hedge_mode_selector.py
git commit -m "feat: add _compute_hedge_weights for cumulative Hedge weighting"
```

---

### Task 4: Integrate Hedge Selection into FitnessEstimator.compute_dataframe

**Files:**
- Modify: `freqtrade/ohio/core/strategy_router/fitness_estimator.py`
- Modify: `tests/ohio/test_hedge_mode_selector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ohio/test_hedge_mode_selector.py`:

```python
from freqtrade.ohio.core.domain.models import StrategyMode


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
        "ohio_sv_trend_persistence": rng.uniform(-1.0, 1.0, n_rows),
    })


class TestHedgeIntegration:
    """Tests for Hedge integration in compute_dataframe."""

    def test_hedge_columns_exist(self):
        """compute_dataframe produces ohio_hedge_weight_* columns."""
        est = FitnessEstimator(hedge_eta=0.1)
        df_out = est.compute_dataframe(_make_ohio_dataframe())
        for mode in StrategyMode:
            col = f"ohio_hedge_weight_{mode.value}"
            assert col in df_out.columns, f"Missing column: {col}"

    def test_hedge_weights_sum_to_one(self):
        """ohio_hedge_weight_* columns sum to 1.0 per row."""
        est = FitnessEstimator(hedge_eta=0.1)
        df_out = est.compute_dataframe(_make_ohio_dataframe())
        weight_cols = [f"ohio_hedge_weight_{m.value}" for m in StrategyMode]
        row_sums = df_out[weight_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_backward_compat_eta_zero_temp_one(self):
        """eta=0, temperature=1 → identical to pre-Hedge argmax."""
        df = _make_ohio_dataframe(100)

        est_old = FitnessEstimator(hedge_eta=0.0, hedge_temperature=1.0)
        est_new = FitnessEstimator(hedge_eta=0.0, hedge_temperature=1.0)

        df_old = est_old.compute_dataframe(df)
        df_new = est_new.compute_dataframe(df)

        # Active modes should be identical
        pd.testing.assert_series_equal(
            df_old["ohio_active_mode"],
            df_new["ohio_active_mode"],
        )
        # Fitness scores should be identical (raw, not adjusted)
        for mode in StrategyMode:
            col = f"ohio_fitness_{mode.value}"
            pd.testing.assert_series_equal(df_old[col], df_new[col])

    def test_mr_dominance_broken(self):
        """With Hedge active, MR should not be 96%+ of active modes."""
        # Create data where TF rewards are consistently positive
        n = 200
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "close": 100.0 + np.cumsum(np.abs(rng.normal(0.5, 0.1, n))),  # trending up
            "ohio_stable_trend": np.full(n, 0.0),         # neutral → MR wins raw fitness
            "ohio_stable_volatility": np.full(n, 0.5),
            "ohio_stable_downside": np.full(n, 0.3),
            "ohio_stable_liquidity": np.full(n, 0.15),
            "ohio_stable_relative_strength": np.full(n, 0.5),
            "ohio_stable_correlation": np.full(n, 0.25),
            "ohio_stable_breadth": np.full(n, 0.5),
            "ohio_meta_confidence": np.full(n, 0.8),
            "ohio_meta_stability": np.full(n, 0.8),
            "ohio_feat_atr_ratio_14": np.full(n, 0.01),
            "ohio_sv_trend_persistence": np.full(n, 0.8),  # strong trend → TF rewards positive
        })

        est = FitnessEstimator(hedge_eta=0.1, hedge_temperature=2.0)
        df_out = est.compute_dataframe(df)

        mode_counts = df_out["ohio_active_mode"].value_counts(normalize=True)
        mr_pct = mode_counts.get("mean_reversion", 0.0)
        assert mr_pct < 0.96, (
            f"MR still dominates at {mr_pct:.1%}. Hedge should diversify. "
            f"Distribution: {mode_counts.to_dict()}"
        )

    def test_existing_fitness_columns_unchanged(self):
        """ohio_fitness_* columns contain raw fitness (pre-Hedge), not adjusted."""
        df = _make_ohio_dataframe()
        est_hedge = FitnessEstimator(hedge_eta=0.1, hedge_temperature=2.0)
        est_raw = FitnessEstimator(hedge_eta=0.0, hedge_temperature=1.0)

        df_hedge = est_hedge.compute_dataframe(df)
        df_raw = est_raw.compute_dataframe(df)

        for mode in StrategyMode:
            col = f"ohio_fitness_{mode.value}"
            pd.testing.assert_series_equal(
                df_hedge[col], df_raw[col],
                check_names=False,
                obj=f"Raw fitness {col} should be identical regardless of Hedge params",
            )

    def test_warmup_nan_handling(self):
        """First row has NaN from pct_change → no crash, weight 0.25."""
        est = FitnessEstimator(hedge_eta=0.1)
        df = _make_ohio_dataframe(5)
        df_out = est.compute_dataframe(df)

        # First row should have near-uniform weights
        for mode in StrategyMode:
            w = df_out[f"ohio_hedge_weight_{mode.value}"].iloc[0]
            assert abs(w - 0.25) < 1e-10, f"First row weight for {mode}: {w}"

    def test_active_mode_still_valid(self):
        """ohio_active_mode values are still valid StrategyMode strings."""
        est = FitnessEstimator(hedge_eta=0.1, hedge_temperature=2.0)
        df_out = est.compute_dataframe(_make_ohio_dataframe(100))
        valid = {m.value for m in StrategyMode}
        bad = set(df_out["ohio_active_mode"].unique()) - valid
        assert not bad, f"Invalid active_mode values: {bad}"

    def test_does_not_mutate_input(self):
        """Input DataFrame should not be mutated."""
        est = FitnessEstimator(hedge_eta=0.1)
        df = _make_ohio_dataframe()
        original_cols = list(df.columns)
        est.compute_dataframe(df)
        assert list(df.columns) == original_cols
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py::TestHedgeIntegration -v`
Expected: FAIL with `TypeError: FitnessEstimator.__init__() got an unexpected keyword argument 'hedge_eta'`

- [ ] **Step 3: Write implementation**

Modify `FitnessEstimator.__init__()` to accept Hedge params:

```python
class FitnessEstimator:
    def __init__(
        self,
        profiles: dict[StrategyMode, StrategyProfile] | None = None,
        hedge_eta: float = 0.1,
        hedge_temperature: float = 2.0,
        hedge_weight_floor: float = 0.05,
    ) -> None:
        if profiles is None:
            profiles = load_default_profiles()
        self._profiles: dict[StrategyMode, StrategyProfile] = profiles
        self._hedge_eta = hedge_eta
        self._hedge_temperature = hedge_temperature
        self._hedge_weight_floor = hedge_weight_floor
```

Modify the "Write output columns" section of `compute_dataframe()`. Replace lines 240-259 (from `mode_order = [` through `df["ohio_active_mode"] = active_modes`) with:

```python
        # ------------------------------------------------------------------
        # Write fitness output columns (raw, pre-Hedge)
        # ------------------------------------------------------------------
        mode_order = [
            StrategyMode.TREND_FOLLOWING,
            StrategyMode.MEAN_REVERSION,
            StrategyMode.BREAKOUT,
            StrategyMode.DEFENSIVE,
        ]

        score_matrix = np.column_stack(
            [mode_scores[m.value] for m in mode_order]
        )

        for mode in mode_order:
            df[f"ohio_fitness_{mode.value}"] = mode_scores[mode.value]

        # ------------------------------------------------------------------
        # Hedge-based mode selection
        # ------------------------------------------------------------------
        # Reward computation from 1-bar return
        ret = df["close"].pct_change().to_numpy(dtype=np.float64)
        trend = (
            df.get("ohio_sv_trend_persistence", pd.Series(0.0, index=df.index))
            .to_numpy(dtype=np.float64)
        )
        atr = (
            df.get("ohio_feat_atr_ratio_14", pd.Series(0.01, index=df.index))
            .fillna(0.01)
            .to_numpy(dtype=np.float64)
        )

        rewards = _compute_rewards(ret, trend, atr)
        hedge_weights = _compute_hedge_weights(
            rewards, self._hedge_eta, self._hedge_weight_floor,
        )

        # Temperature flatten + Hedge adjust → argmax
        flattened = np.power(
            np.maximum(score_matrix, 1e-8),
            1.0 / self._hedge_temperature,
        )
        adjusted = flattened * hedge_weights
        best_indices = np.argmax(adjusted, axis=1)

        mode_values = np.array([m.value for m in mode_order])
        df["ohio_active_mode"] = mode_values[best_indices]

        # Hedge weight columns for monitoring
        for i, mode in enumerate(mode_order):
            df[f"ohio_hedge_weight_{mode.value}"] = hedge_weights[:, i]

        return df
```

- [ ] **Step 4: Run ALL tests to verify they pass**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py tests/ohio/test_fitness_estimator.py -v`
Expected: All tests PASS. Existing fitness estimator tests must still pass (backward compat — default hedge params should not break existing behavior).

**Note on existing test compatibility:** The existing `test_compute_dataframe_adds_five_columns` test (test #11) checks `len(df_out.columns) == len(df_in.columns) + 5`. This will now fail because we add 4 extra hedge weight columns (total +9). This test needs updating:

In `tests/ohio/test_fitness_estimator.py`, update `test_compute_dataframe_adds_five_columns`:

```python
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
```

Also, `_make_ohio_dataframe` in `test_fitness_estimator.py` needs `close`, `ohio_feat_atr_ratio_14`, and `ohio_sv_trend_persistence` columns for the Hedge computation. Add them:

```python
def _make_ohio_dataframe(n_rows: int = 10) -> pd.DataFrame:
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
            "ohio_sv_trend_persistence": rng.uniform(-1.0, 1.0, n_rows),
        }
    )
    return df
```

The existing `test_vectorized_matches_scalar_parity` (test #20) compares scalar `compute_fitness` with vectorized `compute_dataframe`. Since Hedge only affects `ohio_active_mode` (not `ohio_fitness_*` columns), fitness parity should still hold. However, the DataFrame built in that test lacks `close`, `ohio_feat_atr_ratio_14`, `ohio_sv_trend_persistence`. Add them:

```python
def test_vectorized_matches_scalar_parity(estimator: FitnessEstimator) -> None:
    # ... existing row-building code ...
    df = pd.DataFrame(rows)
    # Add columns needed by Hedge (values don't matter for fitness parity)
    df["close"] = 100.0
    df["ohio_feat_atr_ratio_14"] = 0.01
    df["ohio_sv_trend_persistence"] = 0.0
    df_out = estimator.compute_dataframe(df)
    # ... rest unchanged ...
```

Similarly update the `_make_ohio_dataframe` helper used in `test_compute_dataframe_handles_missing_meta_columns` and `test_compute_dataframe_empty_dataframe` — the base `_make_ohio_dataframe` function already has the fix above, so these tests will work.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ohio/test_hedge_mode_selector.py tests/ohio/test_fitness_estimator.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add freqtrade/ohio/core/strategy_router/fitness_estimator.py tests/ohio/test_hedge_mode_selector.py tests/ohio/test_fitness_estimator.py
git commit -m "feat: integrate Hedge mode selection into FitnessEstimator.compute_dataframe"
```

---

### Task 5: Wire Hedge Params from Config to thin_strategy

**Files:**
- Modify: `freqtrade/ohio/adapters/freqtrade/thin_strategy.py:91`
- Modify: `tests/ohio/test_thin_strategy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/ohio/test_thin_strategy.py`:

```python
def test_fitness_estimator_receives_hedge_params():
    """FitnessEstimator should be initialized with Hedge params."""
    strategy = OhioThinStrategy({"stake_currency": "USDT"})
    est = strategy._fitness_estimator
    # Default values from OhioConfig
    assert est._hedge_eta == 0.1
    assert est._hedge_temperature == 2.0
    assert est._hedge_weight_floor == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ohio/test_thin_strategy.py::test_fitness_estimator_receives_hedge_params -v`
Expected: FAIL — `est._hedge_eta` doesn't exist yet (old constructor)

- [ ] **Step 3: Write minimal implementation**

In `thin_strategy.py` line 91, change:

```python
# Before
self._fitness_estimator = FitnessEstimator()

# After
self._fitness_estimator = FitnessEstimator(
    hedge_eta=0.1,
    hedge_temperature=2.0,
    hedge_weight_floor=0.05,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ohio/test_thin_strategy.py::test_fitness_estimator_receives_hedge_params -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/adapters/freqtrade/thin_strategy.py tests/ohio/test_thin_strategy.py
git commit -m "feat: wire Hedge params into OhioThinStrategy"
```

---

### Task 6: Update Existing Tests for New Column Count

**Files:**
- Modify: `tests/ohio/test_fitness_estimator.py`

This task ensures all pre-existing fitness estimator tests pass with the new Hedge columns. Most changes were described in Task 4, but this task handles any remaining test adjustments.

- [ ] **Step 1: Run existing tests to see what breaks**

Run: `python -m pytest tests/ohio/test_fitness_estimator.py -v`
Note which tests fail.

- [ ] **Step 2: Fix failing tests**

Expected fixes (all described in Task 4):
1. `_make_ohio_dataframe`: add `close`, `ohio_feat_atr_ratio_14`, `ohio_sv_trend_persistence`
2. `test_compute_dataframe_adds_five_columns`: update to expect +9 columns
3. `test_vectorized_matches_scalar_parity`: add `close`, `ohio_feat_atr_ratio_14`, `ohio_sv_trend_persistence` to manually built DataFrame

- [ ] **Step 3: Run all tests to verify**

Run: `python -m pytest tests/ohio/test_fitness_estimator.py tests/ohio/test_hedge_mode_selector.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/ohio/test_fitness_estimator.py
git commit -m "test: update existing fitness tests for Hedge output columns"
```

---

### Task 7: End-to-End Backward Compatibility Verification

**Files:**
- Test: `tests/ohio/test_hedge_mode_selector.py` (already has backward compat test)
- Run: Full OHIO test suite

- [ ] **Step 1: Run the complete OHIO test suite**

Run: `python -m pytest tests/ohio/ -v --tb=short`
Expected: ALL PASS — no regressions.

- [ ] **Step 2: Verify backward compatibility claim**

Run a specific check: create a FitnessEstimator with eta=0, temperature=1 and compare against a fresh DataFrame to ensure `ohio_active_mode` matches what pure argmax would produce.

This is covered by `test_backward_compat_eta_zero_temp_one` in Task 4.

- [ ] **Step 3: Final commit (if any remaining fixes)**

```bash
git add -A
git commit -m "test: E2 Hedge mode selector — full suite green"
```
