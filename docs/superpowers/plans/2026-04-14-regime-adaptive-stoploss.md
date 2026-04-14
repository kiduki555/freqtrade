# Regime-Adaptive Dynamic Stoploss — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-percentage stoploss with ATR-scaled, mode-dependent dynamic stoploss and Chandelier trailing for trend/breakout modes.

**Architecture:** Hybrid approach — existing per-mode % ranges are multiplied by an ATR ratio (current vol / baseline vol). Mean-reversion widens stops when vol rises; all other modes tighten. Chandelier trailing stop replaces simple profit lock for trend_following and breakout.

**Tech Stack:** Python 3.13, pandas, numpy, pydantic, pytest, YAML configs

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `freqtrade/ohio/core/market_state/feature_builder.py` | Modify | Add `ohio_feat_atr_baseline` column |
| `freqtrade/ohio/core/strategy_router/strategy_profile.py` | Modify | Add 5 new fields to `StrategyProfile` |
| `freqtrade/ohio/adapters/freqtrade/exit_adapter.py` | Modify | ATR scaling + Chandelier in `compute_stoploss()` |
| `freqtrade/ohio/adapters/freqtrade/thin_strategy.py` | Modify | Pass ATR values to `compute_stoploss()` |
| `freqtrade/ohio/config/strategies/trend_following.yaml` | Modify | Add ATR + Chandelier fields |
| `freqtrade/ohio/config/strategies/mean_reversion.yaml` | Modify | Add ATR fields (widen direction) |
| `freqtrade/ohio/config/strategies/breakout.yaml` | Modify | Add ATR + Chandelier fields |
| `freqtrade/ohio/config/strategies/defensive.yaml` | Modify | Add ATR fields |
| `tests/ohio/test_exit_adapter.py` | Modify | Add 9 new test classes |
| `tests/ohio/test_feature_builder.py` | Modify | Add ATR baseline test |

---

### Task 1: Add ATR Baseline to FeatureBuilder

**Files:**
- Modify: `freqtrade/ohio/core/market_state/feature_builder.py`
- Modify: `tests/ohio/test_feature_builder.py` (if exists, else check)

- [ ] **Step 1: Write the failing test**

Add to `tests/ohio/test_feature_builder.py`:

```python
class TestAtrBaseline:
    def test_atr_baseline_computed(self):
        """ohio_feat_atr_baseline is the 168h rolling median of atr_ratio_14."""
        builder = FeatureBuilder()
        # Need at least 168+14 rows for non-NaN baseline
        df = _make_ohlcv(200)
        df = builder.compute(df)
        assert "ohio_feat_atr_baseline" in df.columns
        # After 168 rows, baseline should be non-NaN
        assert not pd.isna(df["ohio_feat_atr_baseline"].iloc[-1])

    def test_atr_baseline_nan_during_warmup(self):
        """First 168 bars should have NaN baseline."""
        builder = FeatureBuilder()
        df = _make_ohlcv(200)
        df = builder.compute(df)
        # Row 100 (< 168) should be NaN
        assert pd.isna(df["ohio_feat_atr_baseline"].iloc[100])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ohio/test_feature_builder.py::TestAtrBaseline -v`
Expected: FAIL — `ohio_feat_atr_baseline` column not found

- [ ] **Step 3: Implement — add baseline to feature_builder.py**

In `feature_builder.py`, add to `FEATURE_COLUMNS` list:

```python
# After "ohio_feat_parkinson_vol_24"
"ohio_feat_atr_baseline",
```

At the end of `_compute_volatility()`, add:

```python
# ATR baseline: 168h (7-day) rolling median for stoploss scaling
df["ohio_feat_atr_baseline"] = df["ohio_feat_atr_ratio_14"].rolling(
    window=168, min_periods=168
).median()
```

Also update the docstring of `compute()` to mention 19 features (was 18).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ohio/test_feature_builder.py::TestAtrBaseline -v`
Expected: PASS

- [ ] **Step 5: Run full existing tests to confirm no regressions**

Run: `pytest tests/ohio/test_exit_adapter.py tests/ohio/test_feature_builder.py -v`
Expected: All pass (existing + new)

- [ ] **Step 6: Commit**

```bash
git add freqtrade/ohio/core/market_state/feature_builder.py tests/ohio/test_feature_builder.py
git commit -m "feat: add ohio_feat_atr_baseline (168h rolling median) to FeatureBuilder"
```

---

### Task 2: Add New Fields to StrategyProfile

**Files:**
- Modify: `freqtrade/ohio/core/strategy_router/strategy_profile.py`
- Modify: `freqtrade/ohio/config/strategies/trend_following.yaml`
- Modify: `freqtrade/ohio/config/strategies/mean_reversion.yaml`
- Modify: `freqtrade/ohio/config/strategies/breakout.yaml`
- Modify: `freqtrade/ohio/config/strategies/defensive.yaml`

- [ ] **Step 1: Write the failing test**

Add to `tests/ohio/test_exit_adapter.py` (profile validation):

```python
class TestProfileNewFields:
    def test_trend_following_has_atr_fields(self):
        """Trend-following profile loads with new ATR/Chandelier fields."""
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode

        profiles = load_default_profiles()
        tf = profiles[StrategyMode.TREND_FOLLOWING]
        assert tf.atr_stop_direction == "tighten"
        assert tf.atr_scale_cap == 2.0
        assert tf.chandelier_enabled is True
        assert tf.chandelier_multiplier == 2.5
        assert tf.chandelier_activation == 0.02

    def test_mean_reversion_widens(self):
        """Mean-reversion profile has widen direction and no chandelier."""
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode

        profiles = load_default_profiles()
        mr = profiles[StrategyMode.MEAN_REVERSION]
        assert mr.atr_stop_direction == "widen"
        assert mr.chandelier_enabled is False

    def test_breakout_chandelier(self):
        """Breakout has chandelier with tighter multiplier."""
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode

        profiles = load_default_profiles()
        bo = profiles[StrategyMode.BREAKOUT]
        assert bo.chandelier_enabled is True
        assert bo.chandelier_multiplier == 2.0
        assert bo.chandelier_activation == 0.015

    def test_defensive_no_chandelier(self):
        """Defensive has tighten direction and no chandelier."""
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode

        profiles = load_default_profiles()
        d = profiles[StrategyMode.DEFENSIVE]
        assert d.atr_stop_direction == "tighten"
        assert d.chandelier_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ohio/test_exit_adapter.py::TestProfileNewFields -v`
Expected: FAIL — `StrategyProfile` has no attribute `atr_stop_direction`

- [ ] **Step 3: Add fields to StrategyProfile model**

In `strategy_profile.py`, add these fields to the `StrategyProfile` class (after `leverage_range`):

```python
from typing import Literal

# ... inside StrategyProfile class:
atr_stop_direction: Literal["tighten", "widen"] = "tighten"
atr_scale_cap: float = 2.0
chandelier_enabled: bool = False
chandelier_multiplier: float = 2.5
chandelier_activation: float = 0.02
```

Add validators:

```python
@field_validator("atr_scale_cap")
@classmethod
def atr_scale_cap_positive(cls, v: float) -> float:
    if v <= 0:
        raise ValueError(f"atr_scale_cap must be positive, got {v}")
    return v

@field_validator("chandelier_multiplier")
@classmethod
def chandelier_multiplier_positive(cls, v: float) -> float:
    if v <= 0:
        raise ValueError(f"chandelier_multiplier must be positive, got {v}")
    return v

@field_validator("chandelier_activation")
@classmethod
def chandelier_activation_range(cls, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"chandelier_activation must be in [0, 1], got {v}")
    return v
```

- [ ] **Step 4: Update YAML configs**

**trend_following.yaml** — append after `leverage_range`:

```yaml
atr_stop_direction: tighten
atr_scale_cap: 2.0
chandelier_enabled: true
chandelier_multiplier: 2.5
chandelier_activation: 0.02
```

**mean_reversion.yaml** — append after `leverage_range`:

```yaml
atr_stop_direction: widen
atr_scale_cap: 2.0
chandelier_enabled: false
```

**breakout.yaml** — append after `leverage_range`:

```yaml
atr_stop_direction: tighten
atr_scale_cap: 2.0
chandelier_enabled: true
chandelier_multiplier: 2.0
chandelier_activation: 0.015
```

**defensive.yaml** — append after `leverage_range`:

```yaml
atr_stop_direction: tighten
atr_scale_cap: 2.0
chandelier_enabled: false
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/ohio/test_exit_adapter.py::TestProfileNewFields -v`
Expected: PASS

- [ ] **Step 6: Run full profile/strategy tests**

Run: `pytest tests/ohio/ -v -k "profile or strategy" --no-header -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add freqtrade/ohio/core/strategy_router/strategy_profile.py \
       freqtrade/ohio/config/strategies/*.yaml \
       tests/ohio/test_exit_adapter.py
git commit -m "feat: add ATR scaling + Chandelier fields to StrategyProfile and YAMLs"
```

---

### Task 3: Implement ATR Scaling in compute_stoploss

**Files:**
- Modify: `freqtrade/ohio/adapters/freqtrade/exit_adapter.py`
- Modify: `tests/ohio/test_exit_adapter.py`

- [ ] **Step 1: Write the failing tests for ATR scaling**

Add to `tests/ohio/test_exit_adapter.py`:

```python
@dataclass(frozen=True)
class FakeProfileV2:
    """Profile with ATR + Chandelier fields."""
    stoploss_range: tuple[float, float]
    atr_stop_direction: str = "tighten"
    atr_scale_cap: float = 2.0
    chandelier_enabled: bool = False
    chandelier_multiplier: float = 2.5
    chandelier_activation: float = 0.02


def _trend_profile_v2() -> FakeProfileV2:
    return FakeProfileV2(
        stoploss_range=(-0.12, -0.08),
        atr_stop_direction="tighten",
        chandelier_enabled=True,
        chandelier_multiplier=2.5,
        chandelier_activation=0.02,
    )


def _mr_profile_v2() -> FakeProfileV2:
    return FakeProfileV2(
        stoploss_range=(-0.08, -0.05),
        atr_stop_direction="widen",
        chandelier_enabled=False,
    )


class TestAtrScalingTighten:
    def test_high_vol_tightens_trend(self):
        """atr_scale=1.5 + trend_following (tighten) → stop closer to zero."""
        base = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.0,
        )
        tightened = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.5,
        )
        assert tightened > base  # closer to zero = tighter


class TestAtrScalingWiden:
    def test_high_vol_widens_mr(self):
        """atr_scale=1.5 + mean_reversion (widen) → stop further from zero."""
        base = compute_stoploss(
            _mr_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.0,
        )
        widened = compute_stoploss(
            _mr_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.5,
        )
        assert widened < base  # further from zero = wider


class TestAtrScaleClamped:
    def test_extreme_scale_clamped(self):
        """atr_scale=5.0 should behave same as atr_scale=2.0 (cap)."""
        at_cap = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=2.0,
        )
        extreme = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=5.0,
        )
        assert extreme == pytest.approx(at_cap, abs=1e-9)


class TestBackwardCompat:
    def test_scale_1_matches_original(self):
        """atr_scale=1.0 produces identical result to no-ATR call."""
        # Use FakeProfile (old style, no ATR fields) for original
        original = compute_stoploss(
            _trend_profile(), _default_policy(), 0.0, 0.3, fitness_score=0.5,
        )
        # Use FakeProfileV2 with atr_scale=1.0
        with_atr = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.3,
            fitness_score=0.5, atr_scale=1.0,
        )
        assert with_atr == pytest.approx(original, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ohio/test_exit_adapter.py::TestAtrScalingTighten tests/ohio/test_exit_adapter.py::TestAtrScalingWiden tests/ohio/test_exit_adapter.py::TestAtrScaleClamped tests/ohio/test_exit_adapter.py::TestBackwardCompat -v`
Expected: FAIL — `compute_stoploss()` doesn't accept `atr_scale` parameter

- [ ] **Step 3: Implement ATR scaling in exit_adapter.py**

Replace the entire `compute_stoploss` function in `exit_adapter.py`:

```python
def compute_stoploss(
    profile: object,
    policy: object,
    current_profit: float,
    transition_risk: float,
    fitness_score: float = 0.5,
    atr_scale: float = 1.0,
    atr_ratio: float = 0.0,
) -> float:
    """Compute the dynamic stoploss ratio for an open position.

    Args:
        profile: A ``StrategyProfile`` with ``stoploss_range`` and optional
                 ``atr_stop_direction``, ``atr_scale_cap``, ``chandelier_*``.
        policy: An ``ExecutionPolicy`` with ``enabled`` and
                ``stoploss_width_adj``.
        current_profit: Unrealised profit ratio of the position.
        transition_risk: Regime-transition probability in [0, 1].
        fitness_score: Current fitness [0, 1] for lerping within stoploss_range.
        atr_scale: Ratio of current ATR to baseline ATR. 1.0 = average vol.
                   Clamped internally by profile.atr_scale_cap.
        atr_ratio: Current ATR/close ratio for Chandelier calculation.

    Returns:
        Negative float representing the stoploss ratio.
    """
    if not policy.enabled:  # type: ignore[union-attr]
        return _HARD_FLOOR

    sl_wide, sl_tight = profile.stoploss_range  # type: ignore[union-attr]
    base_sl = _lerp(sl_wide, sl_tight, fitness_score)

    adjusted = base_sl + policy.stoploss_width_adj  # type: ignore[union-attr]

    # --- ATR scaling (NEW) ---
    atr_cap = getattr(profile, "atr_scale_cap", 2.0)
    clamped_scale = _clamp(atr_scale, 1.0 / atr_cap, atr_cap)
    direction = getattr(profile, "atr_stop_direction", "tighten")

    if direction == "widen":
        # MR: vol up → stop wider (more negative)
        adjusted *= clamped_scale
    else:
        # tighten: vol up → stop tighter (closer to zero)
        adjusted *= (1.0 / clamped_scale)

    # Regime-transition tightening (graduated: fires before entry block at 0.85)
    if transition_risk > _TRANSITION_TIGHTEN_THRESHOLD:
        adjusted *= _TRANSITION_TIGHTEN_FACTOR  # closer to zero = tighter

    # --- Trailing: Chandelier or profit lock ---
    chandelier_on = getattr(profile, "chandelier_enabled", False)
    chandelier_mult = getattr(profile, "chandelier_multiplier", 2.5)
    chandelier_act = getattr(profile, "chandelier_activation", 0.02)

    if chandelier_on and atr_ratio > 0 and current_profit > chandelier_act:
        chandelier_sl = -(chandelier_mult * atr_ratio)
        adjusted = max(adjusted, chandelier_sl)  # tighter wins (both negative)
    elif current_profit > _TRAILING_PROFIT_THRESHOLD:
        adjusted = max(adjusted, -(current_profit * _TRAILING_PROFIT_RATIO))

    return _clamp(adjusted, _HARD_FLOOR, _TIGHTEST)
```

- [ ] **Step 4: Run ATR scaling tests**

Run: `pytest tests/ohio/test_exit_adapter.py::TestAtrScalingTighten tests/ohio/test_exit_adapter.py::TestAtrScalingWiden tests/ohio/test_exit_adapter.py::TestAtrScaleClamped tests/ohio/test_exit_adapter.py::TestBackwardCompat -v`
Expected: PASS

- [ ] **Step 5: Run ALL existing tests to confirm no regressions**

Run: `pytest tests/ohio/test_exit_adapter.py -v`
Expected: All 26 existing + 4 new = 30 passing

- [ ] **Step 6: Commit**

```bash
git add freqtrade/ohio/adapters/freqtrade/exit_adapter.py tests/ohio/test_exit_adapter.py
git commit -m "feat: ATR-scaled dynamic stoploss with mode-dependent direction"
```

---

### Task 4: Implement Chandelier Trailing Stop Tests

**Files:**
- Modify: `tests/ohio/test_exit_adapter.py`

- [ ] **Step 1: Write Chandelier tests**

Add to `tests/ohio/test_exit_adapter.py`:

```python
class TestChandelierActivation:
    def test_below_activation_uses_profit_lock(self):
        """Profit below chandelier_activation → falls back to profit lock."""
        sl = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.01,  # below 0.02 activation
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,
        )
        # Should not use chandelier (profit < activation)
        # Should not use profit lock either (profit < 0.02)
        # Just base stoploss
        base = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.0,
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,
        )
        assert sl == pytest.approx(base, abs=1e-9)

    def test_above_activation_uses_chandelier(self):
        """Profit above activation → chandelier trailing active."""
        sl = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.05,  # above 0.02 activation
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,  # chandelier = -(2.5 * 0.012) = -0.03
        )
        # Chandelier SL = -0.03
        # Base SL (trend, fitness 0.5) ~= -0.10
        # max(-0.10, -0.03) = -0.03 (chandelier wins, tighter)
        assert sl == pytest.approx(-0.03, abs=0.005)


class TestChandelierTighterWins:
    def test_chandelier_overrides_base_when_tighter(self):
        """When chandelier SL is tighter than base, chandelier is used."""
        # Low ATR ratio → tight chandelier
        sl = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.05,
            transition_risk=0.0,
            fitness_score=0.0,   # low fitness → wide base (sl_wide = -0.12)
            atr_scale=1.0,
            atr_ratio=0.010,     # chandelier = -(2.5 * 0.01) = -0.025
        )
        # max(-0.12, -0.025) = -0.025 → chandelier wins
        assert sl == pytest.approx(-0.025, abs=0.005)


class TestChandelierDisabledMR:
    def test_mr_uses_profit_lock_not_chandelier(self):
        """MR mode (chandelier_enabled=False) uses profit lock at 2%+."""
        sl = compute_stoploss(
            _mr_profile_v2(), _default_policy(),
            current_profit=0.10,  # 10% profit
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,
        )
        # MR: chandelier disabled → profit lock: -(0.10 * 0.50) = -0.05
        # base_sl ~= -0.065
        # max(-0.065, -0.05) = -0.05 → profit lock
        assert sl == pytest.approx(-0.05, abs=0.005)
```

- [ ] **Step 2: Run Chandelier tests**

Run: `pytest tests/ohio/test_exit_adapter.py::TestChandelierActivation tests/ohio/test_exit_adapter.py::TestChandelierTighterWins tests/ohio/test_exit_adapter.py::TestChandelierDisabledMR -v`
Expected: PASS (logic already implemented in Task 3)

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ohio/test_exit_adapter.py -v`
Expected: All pass (26 existing + 4 ATR + 4 Chandelier + 4 profile = ~38)

- [ ] **Step 4: Commit**

```bash
git add tests/ohio/test_exit_adapter.py
git commit -m "test: Chandelier trailing stop tests for trend/breakout/MR modes"
```

---

### Task 5: Wire ATR Values in thin_strategy.py

**Files:**
- Modify: `freqtrade/ohio/adapters/freqtrade/thin_strategy.py`

- [ ] **Step 1: Update custom_stoploss to pass ATR values**

In `thin_strategy.py`, modify the `custom_stoploss` method. Replace the current return statement:

```python
# Before (current):
return compute_stoploss(profile, policy, current_profit, transition_risk, fitness)

# After (new):
atr_ratio = float(last.get("ohio_feat_atr_ratio_14", 0.0))
atr_baseline = float(last.get("ohio_feat_atr_baseline", 0.0))

# Compute ATR scale — default to 1.0 if baseline unavailable (warmup)
if atr_baseline > 0 and atr_ratio > 0:
    atr_scale = atr_ratio / atr_baseline
else:
    atr_scale = 1.0

return compute_stoploss(
    profile, policy, current_profit, transition_risk, fitness,
    atr_scale=atr_scale,
    atr_ratio=atr_ratio,
)
```

- [ ] **Step 2: Run the full ohio test suite**

Run: `pytest tests/ohio/ -v --no-header -q`
Expected: All pass — this is a wiring change, no new logic

- [ ] **Step 3: Commit**

```bash
git add freqtrade/ohio/adapters/freqtrade/thin_strategy.py
git commit -m "feat: wire ATR ratio + baseline from DataFrame to compute_stoploss"
```

---

### Task 6: ATR Baseline NaN Fallback Test

**Files:**
- Modify: `tests/ohio/test_exit_adapter.py`

- [ ] **Step 1: Write the NaN fallback test**

Add to `tests/ohio/test_exit_adapter.py`:

```python
class TestAtrBaselineNan:
    def test_nan_baseline_defaults_scale_to_1(self):
        """When atr_baseline is NaN/0 (warmup), atr_scale should be 1.0.

        This is tested at the thin_strategy wiring level, but we verify
        compute_stoploss with atr_scale=1.0 matches no-scaling behavior.
        """
        no_scale = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5,
        )
        explicit_1 = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.0,
        )
        assert no_scale == pytest.approx(explicit_1, abs=1e-9)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/ohio/test_exit_adapter.py::TestAtrBaselineNan -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/ohio/test_exit_adapter.py
git commit -m "test: verify ATR baseline NaN fallback produces scale=1.0"
```

---

### Task 7: Final Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Run complete ohio test suite**

Run: `pytest tests/ohio/ -v --tb=short`
Expected: All tests pass, 0 failures

- [ ] **Step 2: Verify YAML profiles load correctly**

Run: `python -c "from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles; p = load_default_profiles(); [print(f'{m.value}: atr={p[m].atr_stop_direction}, chandelier={p[m].chandelier_enabled}') for m in p]"`

Expected output:
```
trend_following: atr=tighten, chandelier=True
mean_reversion: atr=widen, chandelier=False
breakout: atr=tighten, chandelier=True
defensive: atr=tighten, chandelier=False
```

- [ ] **Step 3: Verify feature builder outputs new column**

Run: `python -c "from freqtrade.ohio.core.market_state.feature_builder import FEATURE_COLUMNS; print('ohio_feat_atr_baseline' in FEATURE_COLUMNS)"`

Expected: `True`

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git status  # verify only expected files
git commit -m "chore: regime-adaptive dynamic stoploss — integration verification"
```
