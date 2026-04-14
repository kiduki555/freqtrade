# Regime-Adaptive Dynamic Stoploss

**Date:** 2026-04-14
**Status:** Draft
**Enhancement:** #1 of 6 (highest priority from literature survey)

## Problem

The current stoploss system uses fixed percentage ranges per strategy mode (e.g., trend_following: -12% to -8%), ignoring real-time volatility. This causes:

1. **Over-tight stops in high-vol regimes** — normal volatility spikes trigger premature exits
2. **Over-loose stops in low-vol regimes** — unnecessary drawdown before stop fires
3. **Mean reversion penalty** — Kaminski & Lo (2014) showed fixed stops actively harm MR strategies by cutting positions before mean reversion completes
4. **No trailing mechanics** — trend_following profits aren't protected by volatility-aware trailing stops

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Hybrid — existing % range x ATR ratio scaling | Preserves tuned per-mode ranges, adds vol reactivity with low risk |
| ATR source | Existing `ohio_feat_atr_ratio_14` | Already in pipeline, 14-bar ATR is industry standard (Wilder) |
| MR direction | Inverse — vol up = stop widens | Overshoots grow with vol; wider stop prevents premature cut (Dai et al. 2010) |
| Chandelier trailing | Enabled for trend_following + breakout | Leung & Zhang (2017) optimal exit; ATR infra shared with stoploss |

## Architecture

### Data Flow

```
populate_indicators()
  └─ FeatureBuilder
       ├─ ohio_feat_atr_ratio_14      (existing)
       └─ ohio_feat_atr_baseline       (NEW: 168h rolling median of atr_ratio_14)

custom_stoploss(pair, trade, current_rate, current_profit, ...)
  ├─ Read last row: ohio_feat_atr_ratio_14, ohio_feat_atr_baseline
  ├─ atr_scale = clamp(atr_ratio / atr_baseline, 0.5, 2.0)
  └─ compute_stoploss(profile, policy, current_profit,
                       transition_risk, fitness, atr_scale, atr_ratio)
```

### compute_stoploss() Logic (exit_adapter.py)

```
Step 1: base_sl = lerp(sl_wide, sl_tight, fitness)           # existing

Step 2: ATR scaling (NEW)
  IF mode.atr_stop_direction == "tighten":
    base_sl *= (1 / atr_scale)      # vol up → tighter
  ELIF mode.atr_stop_direction == "widen":
    base_sl *= atr_scale             # vol up → wider (MR only)

Step 3: transition_risk tightening                            # existing
  IF transition_risk > 0.70:
    adjusted *= 0.70

Step 4: Trailing / Profit lock (EXTENDED)
  IF chandelier_enabled AND current_profit > chandelier_activation:
    chandelier_sl = -(chandelier_multiplier * atr_ratio)
    final = max(adjusted, chandelier_sl)    # tighter wins (both negative)
  ELSE:
    IF current_profit > 0.02:
      final = max(adjusted, -(current_profit * 0.50))   # existing profit lock

Step 5: clamp(final, HARD_FLOOR=-0.20, TIGHTEST=-0.01)       # existing
```

### ATR Scaling Direction Per Mode

| Mode | vol up effect | Formula | Rationale |
|------|--------------|---------|-----------|
| trend_following | tighten | `base_sl * (1/atr_scale)` | Vol spike = trend ending, protect profits |
| mean_reversion | widen | `base_sl * atr_scale` | Vol spike = bigger overshoot, avoid premature cut |
| breakout | tighten | `base_sl * (1/atr_scale)` | Same as trend — false breakout fast cut |
| defensive | tighten | `base_sl * (1/atr_scale)` | Conservative = more protective when vol rises |

### ATR Scale Clamping

```python
atr_scale = clamp(current_atr_ratio / baseline_atr_ratio, 0.5, 2.0)
```

- Floor 0.5: stop never halves even in extremely low vol
- Cap 2.0: stop never doubles even in extreme vol spikes
- Scale 1.0: mathematically identical to current behavior (backward compatible)

### ATR Baseline

```python
ohio_feat_atr_baseline = ohio_feat_atr_ratio_14.rolling(168).median()
# 168 hours = 7 days
# Median is robust to outliers (better than mean for vol spikes)
# NaN for first 168 bars → atr_scale defaults to 1.0
```

### Chandelier Trailing Stop

**Applies to:** trend_following, breakout only
**Replaces:** simple profit lock (profit > 2% → 50% lock) for these modes

```python
# Activation: only after minimum profit reached
if current_profit > chandelier_activation:
    chandelier_sl = -(chandelier_multiplier * current_atr_ratio)
    # e.g., -(2.5 * 0.012) = -0.03 → exit if price drops 3% from current
```

**Per-mode configuration:**

| Mode | chandelier_enabled | chandelier_multiplier | chandelier_activation |
|------|-------------------|----------------------|----------------------|
| trend_following | true | 2.5 | 0.02 (2%) |
| breakout | true | 2.0 | 0.015 (1.5%) |
| mean_reversion | false | — | — |
| defensive | false | — | — |

**Priority:** `max(base_sl_after_atr, chandelier_sl)` — both are negative, so `max` picks the tighter (closer to zero) value.

## Changes

### Files Modified (8)

| File | Change |
|------|--------|
| `feature_builder.py` | Add `ohio_feat_atr_baseline` column (168h rolling median) |
| `strategy_profile.py` | Add fields: `atr_stop_direction`, `atr_scale_cap`, `chandelier_enabled`, `chandelier_multiplier`, `chandelier_activation` |
| `exit_adapter.py` | Add `atr_scale` and `atr_ratio` params to `compute_stoploss()`. Implement ATR scaling + Chandelier logic |
| `thin_strategy.py` | Read ATR values from last row, pass to `compute_stoploss()` |
| `trend_following.yaml` | Add `atr_stop_direction: tighten`, `chandelier_enabled: true`, `chandelier_multiplier: 2.5`, `chandelier_activation: 0.02` |
| `mean_reversion.yaml` | Add `atr_stop_direction: widen`, `chandelier_enabled: false` |
| `breakout.yaml` | Add `atr_stop_direction: tighten`, `chandelier_enabled: true`, `chandelier_multiplier: 2.0`, `chandelier_activation: 0.015` |
| `defensive.yaml` | Add `atr_stop_direction: tighten`, `chandelier_enabled: false` |

### New Fields in StrategyProfile

```python
atr_stop_direction: Literal["tighten", "widen"] = "tighten"
atr_scale_cap: float = 2.0           # max atr_scale effect
chandelier_enabled: bool = False
chandelier_multiplier: float = 2.5   # N x ATR distance from peak
chandelier_activation: float = 0.02  # min profit before chandelier activates
```

All have defaults → existing profiles without these fields still work (backward compatible).

## Test Plan

| Test | Validates |
|------|-----------|
| `test_atr_scaling_tighten` | atr_scale=1.5 + trend_following → stop tighter than base |
| `test_atr_scaling_widen` | atr_scale=1.5 + mean_reversion → stop wider than base |
| `test_atr_scale_clamped` | atr_scale=5.0 → clamped to 2.0 |
| `test_atr_baseline_nan` | First 168 bars (NaN baseline) → atr_scale=1.0 fallback |
| `test_chandelier_activation` | profit < activation → existing logic; profit > activation → chandelier |
| `test_chandelier_tighter_wins` | When chandelier is tighter than base, chandelier is selected |
| `test_chandelier_disabled_mr` | MR mode → chandelier off, existing profit lock active |
| `test_backward_compat` | atr_scale=1.0 → output identical to current implementation |
| `test_profile_yaml_new_fields` | YAML validation accepts new fields with correct types |

## Backward Compatibility

- When `atr_scale = 1.0` (vol equals baseline): all formulas reduce to current behavior
- When `chandelier_enabled = false` (default): existing profit lock unchanged
- New YAML fields all have defaults: existing configs work without modification
- `HARD_FLOOR` (-0.20) and `TIGHTEST` (-0.01) unchanged

## Academic References

- Kaminski & Lo (2014) — stop-loss regime dependency; MR stops harmful
- Dai, Zhang & Zhu (2010) — regime-dependent exit thresholds
- Leung & Zhang (2017) — optimal trailing stop + limit sell combination
- Wilder (1978) / Chandelier Exit — ATR-based stop placement
