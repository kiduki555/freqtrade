# Hedge-Based Multi-Strategy Mode Selector

**Date:** 2026-04-14
**Status:** Draft
**Enhancement:** #2 of 6 (from literature survey)

## Problem

The current mode selection uses `np.argmax(fitness_scores)` — a single-winner-take-all approach. Mean reversion's ideal state vector (trend_persistence=0, volatility_level=0.5) closely matches "normal" market conditions, causing 96% MR dominance across all market regimes. This defeats the purpose of having 4 strategy modes.

Consequences:
1. **No regime diversification** — trend_following, breakout, defensive rarely activate
2. **MR overfitting** — system performance collapses when MR is suboptimal (trending markets)
3. **Wasted infrastructure** — 4 strategy profiles exist but only 1 is effectively used

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Algorithm | Hedge (Cesa-Bianchi & Lugosi 2006) | Full-information setting — can compute reward for ALL modes from 1-bar return, not just selected. Simpler than EXP3, no exploration term |
| Reward signal | 1-bar return aligned with mode philosophy, ATR-normalized | Bar-level feedback (no trade-close dependency), cross-regime comparable |
| Score adjustment | Temperature flattening + Hedge weight multiply | Temperature compresses raw fitness gaps; Hedge weights reflect cumulative performance |
| Selection | Deterministic argmax on adjusted scores | Backtest reproducibility preserved. No stochastic sampling |
| Vectorization | Cumulative sum → exp → normalize | Hedge weight is `exp(η × cumsum(reward))` — fully vectorized, no loop |
| Scope | FitnessEstimator internal only | Downstream (PolicyGenerator, exit_adapter, thin_strategy callbacks) unchanged — only `ohio_active_mode` values shift |

## Architecture

### Data Flow

```
populate_indicators()
  └─ ... (existing pipeline: Feature → Normalize → Factor → Stabilize → Meta)
  └─ FitnessEstimator.compute_dataframe()
       ├─ per-mode fitness scores (EXISTING, unchanged)
       ├─ per-mode 1-bar rewards (NEW)
       │    ret = close.pct_change()
       │    trend = ohio_sv_trend_persistence
       │    atr = ohio_feat_atr_ratio_14
       ├─ Hedge cumulative weights (NEW)
       │    cum_reward → exp(η × cum_reward) → normalize → floor
       ├─ Temperature flatten (NEW)
       │    flattened = fitness ^ (1/temperature)
       ├─ adjusted = flattened × hedge_weights (NEW)
       └─ active_mode = argmax(adjusted) (MODIFIED from argmax(raw))
  └─ PolicyGenerator (unchanged — reads ohio_active_mode)
```

### Reward Definition

At bar t, `pct_change()[t]` = (close[t] - close[t-1]) / close[t-1] is already known. No look-ahead.

```python
ret = df["close"].pct_change()
trend = df["ohio_sv_trend_persistence"]
atr = np.maximum(df["ohio_feat_atr_ratio_14"].fillna(0.01), 1e-8)

reward_tf = np.clip((ret * np.sign(trend)) / atr, -1, 1)       # trend continuation
reward_mr = np.clip((-ret * np.sign(trend)) / atr, -1, 1)      # trend reversal
reward_bo = np.clip((np.abs(ret) - atr) / atr, -1, 1)          # volatility expansion
reward_df = np.clip((atr - np.abs(ret)) / atr, -1, 1)          # low volatility
```

- `/atr` normalizes across volatility regimes (1% return in low-vol ≠ 1% in high-vol)
- `clip(-1, 1)` prevents weight explosion from extreme bars
- NOT zero-sum: multiple modes can have positive reward simultaneously
- NaN from first bar's `pct_change()` → reward = 0 (handled by `np.nan_to_num`)

### Hedge Weight Computation (Vectorized)

```python
# For each mode i:
cum_reward_i = reward_i.cumsum()                              # cumulative reward
raw_weight_i = np.exp(eta * cum_reward_i)                     # exponential weight

# Stack all 4 modes → (N, 4) matrix
raw_weights = np.column_stack([raw_weight_tf, raw_weight_mr, raw_weight_bo, raw_weight_df])

# Floor enforcement + normalize per row
raw_weights = np.maximum(raw_weights, weight_floor)
hedge_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
```

- No loop — `cumsum()` + `exp()` is O(N) vectorized
- Initial state: all cumulative rewards = 0 → all weights = 0.25 (uniform)
- Weight floor prevents any mode from reaching 0 (irreversible death)

### Mode Selection (Modified argmax)

```python
# Step 1: Temperature flatten — compress fitness differences
flattened = np.power(np.maximum(score_matrix, 1e-8), 1.0 / temperature)

# Step 2: Apply Hedge weights
adjusted = flattened * hedge_weights

# Step 3: Deterministic selection
best_indices = np.argmax(adjusted, axis=1)
```

**Temperature effect (T=2):**
| Fitness | Raw | Flattened (√x) |
|---------|-----|----------------|
| 0.72 | 0.72 | 0.849 |
| 0.68 | 0.68 | 0.825 |
| 0.45 | 0.45 | 0.671 |
| 0.55 | 0.55 | 0.742 |

Gap between MR(0.72) and TF(0.68): 0.04 raw → 0.024 flattened. Hedge weights now dominate the selection.

### Reward Direction Per Mode

| Mode | "Bet" | Positive reward when | Negative reward when |
|------|-------|---------------------|---------------------|
| trend_following | Trend continues | Price moves WITH trend direction | Price reverses against trend |
| mean_reversion | Trend reverses | Price moves AGAINST trend direction | Price continues trending |
| breakout | Vol expands | Absolute return > ATR | Absolute return < ATR |
| defensive | Vol contracts | Absolute return < ATR | Absolute return > ATR |

Note: TF and MR are exact mirrors. BO and DEF are exact mirrors. This is mathematically correct — in any given bar, exactly one of each pair benefits.

## Parameters

| Parameter | Field in OhioConfig | Default | Range | Role |
|-----------|-------------------|---------|-------|------|
| Learning rate | `hedge_eta` | 0.1 | [0.01, 0.5] | Higher → weights respond faster to recent performance |
| Temperature | `hedge_temperature` | 2.0 | [1.0, 5.0] | Higher → fitness differences compressed → more diversification |
| Weight floor | `hedge_weight_floor` | 0.05 | [0.01, 0.15] | Minimum weight per mode — prevents irreversible mode death |

**Tuning guidance:**
- `hedge_eta`: Start at 0.1. If mode switching is too frequent, lower to 0.05. If too sticky, raise to 0.2.
- `hedge_temperature`: Start at 2.0. If still 90%+ one mode, raise to 3.0. If switching every bar, lower to 1.5.
- `hedge_weight_floor`: 0.05 is safe default. Only increase if a mode never recovers after drawdown.

## Changes

### Files Modified (3)

| File | Change |
|------|--------|
| `fitness_estimator.py` | Add `hedge_eta`, `hedge_temperature`, `hedge_weight_floor` to `__init__()`. In `compute_dataframe()`: compute rewards, Hedge weights, temperature flatten, adjusted argmax. Add `ohio_hedge_weight_{mode}` output columns |
| `defaults.py` | Add `hedge_eta`, `hedge_temperature`, `hedge_weight_floor` to OhioConfig |
| `thin_strategy.py` | Pass Hedge params from config to `FitnessEstimator()` constructor |

### Files Unchanged

| File | Why |
|------|-----|
| `policy_generator.py` | Reads `ohio_active_mode` — column name unchanged, only values shift |
| `exit_adapter.py` | Uses profile for selected mode — no change |
| `position_adapter.py` | Uses policy — no change |
| `strategy_profile.py` | Hedge params are global, not per-mode |
| `models.py` | Domain objects unchanged |
| `*.yaml` configs | No per-mode Hedge settings |

### New Output Columns

| Column | Type | Purpose |
|--------|------|---------|
| `ohio_hedge_weight_trend_following` | float [floor, ~0.85] | Monitoring / debugging |
| `ohio_hedge_weight_mean_reversion` | float | |
| `ohio_hedge_weight_breakout` | float | |
| `ohio_hedge_weight_defensive` | float | |

Existing `ohio_fitness_{mode}` columns remain as **raw fitness** (pre-Hedge). This preserves observability of both raw fitness and Hedge-adjusted selection.

### FitnessEstimator.__init__ Signature Change

```python
# Before
class FitnessEstimator:
    def __init__(self, profiles=None):

# After
class FitnessEstimator:
    def __init__(
        self,
        profiles=None,
        hedge_eta: float = 0.1,
        hedge_temperature: float = 2.0,
        hedge_weight_floor: float = 0.05,
    ):
```

All new params have defaults → existing callers (including tests) work without modification.

## Backward Compatibility

| Condition | Behavior |
|-----------|----------|
| `hedge_eta = 0.0` | `exp(0 × anything) = 1.0` → all weights uniform → pure temperature only |
| `hedge_eta = 0.0` + `hedge_temperature = 1.0` | No flattening + uniform weights = **identical to current argmax** |
| First bar (pct_change = NaN) | reward = 0 → weight = 0.25 |
| First 168 bars (warmup) | ATR/trend may be NaN → reward defaults to 0 → weights stay near uniform |
| Missing ohio_sv_trend_persistence | trend defaults to 0 → TF/MR rewards = 0 → only BO/DEF update |

## Test Plan

| Test | Validates |
|------|-----------|
| `test_hedge_weights_uniform_init` | η=0, T=1 → all weights 0.25, active_mode identical to current argmax |
| `test_hedge_weights_update` | TF reward positive for 10 bars → TF weight rises above 0.25 |
| `test_temperature_flattening` | T=1 vs T=3: T=3 produces more compressed fitness differences |
| `test_weight_floor_enforced` | Extreme negative reward → weight stays at floor (0.05) |
| `test_reward_computation_tf_positive` | ret=+1%, trend=+0.5 → TF reward positive, MR reward negative |
| `test_reward_atr_normalization` | Same return, higher ATR → smaller reward magnitude |
| `test_mr_dominance_broken` | MR raw fitness #1 but TF cumulative reward higher → TF selected |
| `test_backward_compat_eta_zero` | η=0, T=1 → output identical to pre-Hedge implementation |
| `test_warmup_nan_handling` | First bar pct_change NaN → reward 0, weight 0.25, no crash |
| `test_hedge_columns_exist` | After compute_dataframe: 4 ohio_hedge_weight_* columns present |

## Academic References

- Cesa-Bianchi & Lugosi (2006) — Hedge algorithm, multiplicative weights for expert advice
- Auer et al. (2002) — EXP3 for adversarial bandits (inspiration; we use full-info Hedge instead)
- Ang & Timmermann (2012) — Regime switching in asset allocation
