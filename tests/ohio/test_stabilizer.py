"""Tests for StateStabilizer and StreamingStateStabilizer.

Covers:
 1. EMA smoothing reduces noise (output variance < input variance)
 2. Jump gate: small oscillations below threshold are suppressed
 3. Jump gate: large jumps above threshold pass through
 4. Dwell controller: changes within min_dwell bars are blocked
 5. Dwell controller: changes after min_dwell bars are allowed
 6. All 7 ohio_stable_* columns present in output
 7. trend output in [-1, 1] (preserves input range)
 8. volatility output in [0, 1]
 9. Constant input → output equals input (no drift)
10. Step function: sudden jump is delayed by EMA then accepted
11. Batch vs streaming parity (bit-exact, atol=1e-12)
12. Streaming state persists between update() calls
13. Different ema_alpha values change smoothing intensity
14. Different jump_threshold values change sensitivity
15. Different min_dwell values change minimum holding time
16. Numba JIT function works (basic smoke test)
17. Large dataset performance (1000+ rows < 1 second)
18. Batch vs streaming parity with leading NaN (warmup)
19. No Freqtrade imports in implementation module
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.market_state.stabilizer import (
    StateStabilizer,
    StreamingStateStabilizer,
    _FACTOR_COLS,
    _STABLE_COLS,
    _AXIS_NAMES,
    _jump_dwell_pass,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 500  # enough rows for all tests


def _make_factor_df(
    n: int = _N,
    seed: int = 42,
    noisy: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame with all 7 ohio_factor_* columns."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in _FACTOR_COLS:
        if noisy:
            data[col] = rng.normal(0.5, 0.2, size=n).clip(0, 1)
        else:
            data[col] = np.full(n, 0.5)
    return pd.DataFrame(data)


def _run_streaming(stabilizer: StateStabilizer, df: pd.DataFrame) -> pd.DataFrame:
    """Re-run the same parameters through StreamingStateStabilizer bar by bar."""
    stream = StreamingStateStabilizer(
        ema_alpha=stabilizer.ema_alpha,
        jump_threshold=stabilizer.jump_threshold,
        min_dwell=stabilizer.min_dwell,
    )
    rows: list[dict[str, float]] = []
    for i in range(len(df)):
        factors = {
            axis: float(df[f"ohio_factor_{axis}"].iloc[i])
            for axis in _AXIS_NAMES
        }
        rows.append(stream.update(factors))
    return pd.DataFrame(rows, columns=_AXIS_NAMES)


# ---------------------------------------------------------------------------
# Test 1: EMA smoothing reduces noise
# ---------------------------------------------------------------------------


def test_ema_reduces_noise() -> None:
    """Smoothed output should have lower variance than raw input."""
    df = _make_factor_df(noisy=True)
    stab = StateStabilizer(ema_alpha=0.10, jump_threshold=0.0, min_dwell=0)
    result = stab.stabilize(df)

    for factor_col, stable_col in zip(_FACTOR_COLS, _STABLE_COLS):
        input_var = df[factor_col].var()
        output_var = result[stable_col].var()
        assert output_var < input_var, (
            f"{stable_col}: stable variance {output_var:.4f} "
            f"not less than input variance {input_var:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Jump gate suppresses small oscillations
# ---------------------------------------------------------------------------


def test_jump_gate_suppresses_small_oscillations() -> None:
    """Values oscillating below jump_threshold should freeze."""
    n = 100
    # Build a series around 0.5 with tiny oscillations (amplitude 0.01)
    values = np.full(n, 0.5)
    values[1::2] = 0.51  # alternating tiny nudge

    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})
    stab = StateStabilizer(ema_alpha=0.5, jump_threshold=0.20, min_dwell=0)
    result = stab.stabilize(df)

    stable = result[_STABLE_COLS[0]].to_numpy()
    # After warm-up, the stabilized value should barely move
    unique_vals = np.unique(stable[10:])
    assert len(unique_vals) == 1, (
        f"Expected frozen stable value, got {len(unique_vals)} unique values"
    )


# ---------------------------------------------------------------------------
# Test 3: Jump gate passes large jumps
# ---------------------------------------------------------------------------


def test_jump_gate_passes_large_jumps() -> None:
    """A step change larger than jump_threshold must eventually be accepted."""
    n = 50
    values = np.full(n, 0.1)
    values[20:] = 0.9  # large jump at bar 20

    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})
    # Use high alpha so EMA catches up quickly, no dwell constraint
    stab = StateStabilizer(ema_alpha=0.9, jump_threshold=0.10, min_dwell=0)
    result = stab.stabilize(df)

    stable = result[_STABLE_COLS[0]].to_numpy()
    # By bar 35 (15 bars after the jump) the stable value must have moved above 0.5
    assert stable[-1] > 0.5, (
        f"Expected stable to follow large jump, last value = {stable[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Dwell controller blocks changes within min_dwell bars
# ---------------------------------------------------------------------------


def test_dwell_blocks_rapid_changes() -> None:
    """Two large jumps in quick succession: only the first should pass."""
    n = 40
    values = np.full(n, 0.1, dtype=float)
    values[5:] = 0.9    # first big jump at bar 5
    values[8:] = 0.1    # second reversal at bar 8 (only 3 bars later)

    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})
    stab = StateStabilizer(ema_alpha=0.9, jump_threshold=0.10, min_dwell=10)
    result = stab.stabilize(df)

    stable = result[_STABLE_COLS[0]].to_numpy()
    # At bar 8 the dwell hasn't elapsed — stable should still be high
    assert stable[9] > 0.5, (
        f"Expected dwell to block reversal, stable[9] = {stable[9]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Dwell controller allows changes after min_dwell
# ---------------------------------------------------------------------------


def test_dwell_allows_change_after_min_dwell() -> None:
    """A reversal that occurs after min_dwell bars should be accepted."""
    n = 60
    values = np.full(n, 0.1, dtype=float)
    values[5:] = 0.9     # first jump
    values[25:] = 0.1    # reversal at bar 25 — 20 bars after jump (> min_dwell=10)

    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})
    stab = StateStabilizer(ema_alpha=0.9, jump_threshold=0.10, min_dwell=10)
    result = stab.stabilize(df)

    stable = result[_STABLE_COLS[0]].to_numpy()
    assert stable[-1] < 0.5, (
        f"Expected dwell to allow reversal after min_dwell, stable[-1] = {stable[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: All 7 ohio_stable_* columns present
# ---------------------------------------------------------------------------


def test_all_stable_columns_present() -> None:
    """Output DataFrame must contain all 7 ohio_stable_* columns."""
    df = _make_factor_df()
    stab = StateStabilizer()
    result = stab.stabilize(df)

    for col in _STABLE_COLS:
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test 7: trend output in [-1, 1]
# ---------------------------------------------------------------------------


def test_trend_output_range() -> None:
    """ohio_stable_trend must stay within [-1, 1] when input does."""
    n = 200
    trend_vals = np.random.default_rng(7).uniform(-1, 1, size=n)
    df = _make_factor_df(n=n)
    df["ohio_factor_trend"] = trend_vals

    stab = StateStabilizer(jump_threshold=0.0, min_dwell=0)
    result = stab.stabilize(df)
    stable_trend = result["ohio_stable_trend"].dropna()

    assert stable_trend.min() >= -1.0 - 1e-9
    assert stable_trend.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Test 8: volatility output in [0, 1]
# ---------------------------------------------------------------------------


def test_volatility_output_range() -> None:
    """ohio_stable_volatility must stay within [0, 1] when input does."""
    n = 200
    vol_vals = np.random.default_rng(8).uniform(0, 1, size=n)
    df = _make_factor_df(n=n)
    df["ohio_factor_volatility"] = vol_vals

    stab = StateStabilizer(jump_threshold=0.0, min_dwell=0)
    result = stab.stabilize(df)
    stable_vol = result["ohio_stable_volatility"].dropna()

    assert stable_vol.min() >= 0.0 - 1e-9
    assert stable_vol.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Test 9: Constant input → output equals input
# ---------------------------------------------------------------------------


def test_constant_input_no_drift() -> None:
    """Constant input must pass through unchanged (no EMA or dwell drift)."""
    df = _make_factor_df(noisy=False)  # all columns = 0.5
    stab = StateStabilizer()
    result = stab.stabilize(df)

    for stable_col in _STABLE_COLS:
        vals = result[stable_col].to_numpy()
        assert np.allclose(vals, 0.5, atol=1e-12), (
            f"{stable_col}: constant input drifted, range [{vals.min():.6f}, {vals.max():.6f}]"
        )


# ---------------------------------------------------------------------------
# Test 10: Step function — sudden jump is delayed by EMA then accepted
# ---------------------------------------------------------------------------


def test_step_function_delayed_then_accepted() -> None:
    """A step change must be delayed by EMA smoothing, then eventually accepted."""
    n = 100
    values = np.zeros(n)
    values[30:] = 1.0  # step from 0 to 1 at bar 30

    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})
    stab = StateStabilizer(ema_alpha=0.10, jump_threshold=0.05, min_dwell=1)
    result = stab.stabilize(df)

    stable = result[_STABLE_COLS[0]].to_numpy()
    # Immediately after the step the EMA hasn't caught up — stable should still be ~0
    assert stable[31] < 0.5, (
        f"EMA should delay step: stable[31] = {stable[31]:.4f}"
    )
    # Well after the step the stable value should be high
    assert stable[-1] > 0.5, (
        f"Step should eventually be accepted: stable[-1] = {stable[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11: Batch vs streaming parity (bit-exact)
# ---------------------------------------------------------------------------


def test_batch_streaming_parity() -> None:
    """StateStabilizer and StreamingStateStabilizer must produce bit-exact results."""
    df = _make_factor_df(n=200, seed=11)
    stab = StateStabilizer(ema_alpha=0.10, jump_threshold=0.15, min_dwell=6)
    batch_result = stab.stabilize(df)
    stream_result = _run_streaming(stab, df)

    for axis, stable_col in zip(_AXIS_NAMES, _STABLE_COLS):
        batch_vals = batch_result[stable_col].to_numpy()
        stream_vals = stream_result[axis].to_numpy()

        assert np.allclose(batch_vals, stream_vals, rtol=0, atol=1e-12), (
            f"Parity failure on {stable_col}: max diff = "
            f"{np.max(np.abs(batch_vals - stream_vals)):.2e}"
        )


# ---------------------------------------------------------------------------
# Test 12: Streaming state persists between update() calls
# ---------------------------------------------------------------------------


def test_streaming_state_persists() -> None:
    """StreamingStateStabilizer must remember state across update() calls."""
    stream = StreamingStateStabilizer(ema_alpha=0.10, jump_threshold=0.15, min_dwell=6)

    factors = {axis: 0.5 for axis in _AXIS_NAMES}
    # Feed many bars at the same value
    for _ in range(20):
        stream.update(factors)

    # Confirm internal state is populated
    assert len(stream._ema_state) == len(_AXIS_NAMES)
    assert len(stream._stable_state) == len(_AXIS_NAMES)
    assert len(stream._dwell_counter) == len(_AXIS_NAMES)
    assert stream._initialized is True

    # Now send a large jump and confirm result reflects previous state
    jump_factors = {axis: 0.95 for axis in _AXIS_NAMES}
    result1 = stream.update(jump_factors)
    result2 = stream.update(jump_factors)

    # After the first jump bar the dwell counter resets to 0, so result2 should
    # still equal result1 if dwell > 1 (min_dwell=6)
    for axis in _AXIS_NAMES:
        assert result2[axis] == result1[axis], (
            f"State not persisted for axis {axis}: "
            f"result1={result1[axis]:.4f}, result2={result2[axis]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 13: Different ema_alpha changes smoothing intensity
# ---------------------------------------------------------------------------


def test_ema_alpha_controls_smoothing() -> None:
    """Higher ema_alpha should track raw input more closely (less smoothing)."""
    df = _make_factor_df(noisy=True, seed=13)
    col = _FACTOR_COLS[0]
    stable_col = _STABLE_COLS[0]

    stab_slow = StateStabilizer(ema_alpha=0.05, jump_threshold=0.0, min_dwell=0)
    stab_fast = StateStabilizer(ema_alpha=0.50, jump_threshold=0.0, min_dwell=0)

    slow_var = stab_slow.stabilize(df)[stable_col].var()
    fast_var = stab_fast.stabilize(df)[stable_col].var()

    assert fast_var > slow_var, (
        f"Higher alpha should produce higher variance: fast={fast_var:.6f}, slow={slow_var:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 14: Different jump_threshold changes sensitivity
# ---------------------------------------------------------------------------


def test_jump_threshold_controls_sensitivity() -> None:
    """Higher threshold should produce fewer unique stable values (more frozen)."""
    n = 300
    rng = np.random.default_rng(14)
    values = rng.normal(0.5, 0.1, size=n).clip(0, 1)
    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})

    col = _STABLE_COLS[0]
    low_thresh = StateStabilizer(ema_alpha=0.3, jump_threshold=0.02, min_dwell=0)
    high_thresh = StateStabilizer(ema_alpha=0.3, jump_threshold=0.30, min_dwell=0)

    changes_low = (low_thresh.stabilize(df)[col].diff().abs() > 1e-12).sum()
    changes_high = (high_thresh.stabilize(df)[col].diff().abs() > 1e-12).sum()

    assert changes_high < changes_low, (
        f"Higher threshold should cause fewer changes: "
        f"high={changes_high}, low={changes_low}"
    )


# ---------------------------------------------------------------------------
# Test 15: Different min_dwell changes minimum holding time
# ---------------------------------------------------------------------------


def test_min_dwell_controls_holding_time() -> None:
    """Higher min_dwell should produce fewer state transitions."""
    n = 300
    rng = np.random.default_rng(15)
    values = rng.normal(0.5, 0.25, size=n).clip(0, 1)
    df = pd.DataFrame({col: values.copy() for col in _FACTOR_COLS})

    col = _STABLE_COLS[0]
    short_dwell = StateStabilizer(ema_alpha=0.5, jump_threshold=0.05, min_dwell=1)
    long_dwell = StateStabilizer(ema_alpha=0.5, jump_threshold=0.05, min_dwell=30)

    changes_short = (short_dwell.stabilize(df)[col].diff().abs() > 1e-12).sum()
    changes_long = (long_dwell.stabilize(df)[col].diff().abs() > 1e-12).sum()

    assert changes_long <= changes_short, (
        f"Longer dwell should produce fewer or equal changes: "
        f"long={changes_long}, short={changes_short}"
    )


# ---------------------------------------------------------------------------
# Test 16: Numba JIT function smoke test
# ---------------------------------------------------------------------------


def test_numba_jit_smoke() -> None:
    """_jump_dwell_pass must execute without error and return correct shape."""
    smoothed = np.linspace(0.0, 1.0, 50)
    result = _jump_dwell_pass(smoothed, jump_threshold=0.10, min_dwell=3)

    assert isinstance(result, np.ndarray)
    assert result.shape == smoothed.shape
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# Test 17: Large dataset performance
# ---------------------------------------------------------------------------


def test_large_dataset_performance() -> None:
    """1000+ rows should stabilize all 7 axes in under 1 second."""
    df = _make_factor_df(n=1000, seed=17)
    stab = StateStabilizer()

    start = time.perf_counter()
    result = stab.stabilize(df)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Stabilization took {elapsed:.3f}s — exceeds 1s budget"
    assert len(result) == 1000


# ---------------------------------------------------------------------------
# Test 18: No Freqtrade imports in implementation module
# ---------------------------------------------------------------------------


def test_batch_streaming_parity_with_leading_nan() -> None:
    """Parity must hold when input has leading NaN (warmup period)."""
    n = 200
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        col: np.concatenate([
            np.full(30, np.nan),  # 30 bars of NaN warmup
            rng.normal(0.5, 0.2, size=n - 30).clip(0, 1),
        ])
        for col in _FACTOR_COLS
    })

    stab = StateStabilizer(ema_alpha=0.10, jump_threshold=0.15, min_dwell=6)
    batch_result = stab.stabilize(df)
    stream_result = _run_streaming(stab, df)

    for axis, stable_col in zip(_AXIS_NAMES, _STABLE_COLS):
        batch_vals = batch_result[stable_col].to_numpy()
        stream_vals = stream_result[axis].to_numpy()

        # Leading NaN must be NaN in both
        assert np.all(np.isnan(batch_vals[:30])), (
            f"Batch should have NaN in warmup for {stable_col}"
        )
        assert np.all(np.isnan(stream_vals[:30])), (
            f"Stream should have NaN in warmup for {axis}"
        )

        # Valid region must be bit-exact
        valid_batch = batch_vals[30:]
        valid_stream = stream_vals[30:]
        assert np.allclose(valid_batch, valid_stream, rtol=0, atol=1e-12), (
            f"Parity failure (leading NaN) on {stable_col}: max diff = "
            f"{np.max(np.abs(valid_batch - valid_stream)):.2e}"
        )


# ---------------------------------------------------------------------------
# Test 19: No Freqtrade imports in implementation module
# ---------------------------------------------------------------------------


def test_no_freqtrade_imports() -> None:
    """Implementation module must not import anything from freqtrade.*."""
    module_path = Path(__file__).parents[2] / (
        "freqtrade/ohio/core/market_state/stabilizer.py"
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("freqtrade"), (
                    f"Forbidden import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("freqtrade"):
                raise AssertionError(f"Forbidden import: from {node.module}")
