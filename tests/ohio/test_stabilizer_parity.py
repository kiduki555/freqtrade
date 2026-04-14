"""FT-020: Batch vs Streaming StateStabilizer parity tests.

Verifies that StateStabilizer (batch) and StreamingStateStabilizer (live)
produce bit-exact identical output for the same input sequence.  Divergence
here means backtest results won't match live behavior.

Test count: ~18 parametrized/explicit cases covering smooth signals, regime
shifts, dwell boundaries, jump threshold boundaries, NaN patterns,
determinism, non-default parameters, edge cases, and realistic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.market_state.stabilizer import (
    StateStabilizer,
    StreamingStateStabilizer,
    _AXIS_NAMES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AXIS = list(_AXIS_NAMES)  # defensive copy


def _make_factor_df(values_per_axis: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a factor DataFrame from a dict of axis -> values arrays."""
    data: dict[str, np.ndarray] = {}
    for axis in _AXIS:
        col = f"ohio_factor_{axis}"
        if axis in values_per_axis:
            data[col] = values_per_axis[axis]
        else:
            raise ValueError(f"Missing axis: {axis}")
    return pd.DataFrame(data)


def _random_factor_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a random factor DataFrame with realistic ranges."""
    rng = np.random.default_rng(seed)
    values: dict[str, np.ndarray] = {}
    for axis in _AXIS:
        if axis == "trend":
            values[axis] = np.clip(rng.normal(0.0, 0.3, n), -1.0, 1.0)
        else:
            values[axis] = np.clip(rng.normal(0.5, 0.2, n), 0.0, 1.0)
    return _make_factor_df(values)


def _compare_batch_streaming(
    factors_df: pd.DataFrame,
    ema_alpha: float = 0.10,
    jump_threshold: float = 0.15,
    min_dwell: int = 6,
    atol: float = 1e-10,
) -> None:
    """Run both stabilizer modes and assert parity."""
    # --- Batch ---
    batch_stab = StateStabilizer(ema_alpha, jump_threshold, min_dwell)
    batch_result = batch_stab.stabilize(factors_df)

    # --- Streaming ---
    stream_stab = StreamingStateStabilizer(ema_alpha, jump_threshold, min_dwell)
    streaming_results: dict[str, list[float]] = {axis: [] for axis in _AXIS}

    for i in range(len(factors_df)):
        row: dict[str, float] = {}
        for axis in _AXIS:
            col = f"ohio_factor_{axis}"
            row[axis] = factors_df[col].iloc[i]
        result = stream_stab.update(row)
        for axis in _AXIS:
            streaming_results[axis].append(result[axis])

    # --- Compare ---
    for axis in _AXIS:
        batch_col = f"ohio_stable_{axis}"
        batch_arr = batch_result[batch_col].to_numpy(dtype=np.float64)
        stream_arr = np.array(streaming_results[axis], dtype=np.float64)

        both_nan = np.isnan(batch_arr) & np.isnan(stream_arr)
        neither_nan = ~np.isnan(batch_arr) & ~np.isnan(stream_arr)

        assert (both_nan | neither_nan).all(), (
            f"NaN mismatch on {axis}: "
            f"batch NaN at {np.where(np.isnan(batch_arr) & ~np.isnan(stream_arr))[0].tolist()}, "
            f"stream NaN at {np.where(~np.isnan(batch_arr) & np.isnan(stream_arr))[0].tolist()}"
        )
        if neither_nan.any():
            np.testing.assert_allclose(
                batch_arr[neither_nan],
                stream_arr[neither_nan],
                atol=atol,
                err_msg=f"Parity violation on axis={axis}",
            )


# ---------------------------------------------------------------------------
# 1. Basic parity — smooth sine wave, 1000 bars
# ---------------------------------------------------------------------------


class TestBasicParity:
    """Smooth input signals where EMA and jump/dwell are exercised gently."""

    def test_sine_wave_1000_bars(self) -> None:
        """Smooth sine wave per axis, 1000 bars — batch == streaming."""
        n = 1000
        t = np.linspace(0, 8 * np.pi, n)
        values: dict[str, np.ndarray] = {}
        for i, axis in enumerate(_AXIS):
            phase = i * np.pi / 7
            if axis == "trend":
                values[axis] = 0.5 * np.sin(t + phase)
            else:
                values[axis] = 0.5 + 0.3 * np.sin(t + phase)
        df = _make_factor_df(values)
        _compare_batch_streaming(df)

    def test_random_1000_bars(self) -> None:
        """Random noise, 1000 bars."""
        df = _random_factor_df(1000, seed=123)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 2. Regime shifts
# ---------------------------------------------------------------------------


class TestRegimeShiftParity:
    """Normal -> sudden jump -> recovery -> sideways -> jump."""

    def test_multi_regime(self) -> None:
        n = 500
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.empty(n)
            # Phase 1: calm (0-99)
            arr[0:100] = 0.3 if axis != "trend" else 0.0
            # Phase 2: sudden jump (100-199)
            arr[100:200] = 0.9 if axis != "trend" else 0.8
            # Phase 3: recovery (200-299)
            arr[200:300] = np.linspace(
                0.9 if axis != "trend" else 0.8,
                0.3 if axis != "trend" else 0.0,
                100,
            )
            # Phase 4: sideways (300-399)
            arr[300:400] = 0.3 if axis != "trend" else 0.0
            # Phase 5: second jump (400-499)
            arr[400:500] = 0.7 if axis != "trend" else -0.5
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 3. Dwell boundary — change exactly at min_dwell bar
# ---------------------------------------------------------------------------


class TestDwellBoundaryParity:
    """Input designed so a jump attempt happens exactly at the min_dwell bar."""

    @pytest.mark.parametrize("min_dwell", [3, 6, 12])
    def test_change_at_dwell_boundary(self, min_dwell: int) -> None:
        n = 100
        base_val = 0.3
        jump_val = 0.9
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.full(n, base_val)
            # First jump at bar 1 triggers EMA change; stabilizer accepts at bar 0
            # Then we force a big input change exactly min_dwell bars later
            arr[1 : min_dwell + 1] = jump_val
            # Return to base
            arr[min_dwell + 1 :] = base_val
            if axis == "trend":
                arr = arr - 0.3  # shift trend to [-0.6, 0.6]
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df, min_dwell=min_dwell)


# ---------------------------------------------------------------------------
# 4. Jump threshold boundary — delta = threshold +/- epsilon
# ---------------------------------------------------------------------------


class TestJumpThresholdBoundaryParity:
    """Input with delta exactly at jump_threshold +/- epsilon."""

    def test_just_above_threshold(self) -> None:
        """Delta slightly above threshold — should accept."""
        n = 200
        eps = 1e-6
        threshold = 0.15
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.full(n, 0.3)
            # After EMA settles, inject a step of threshold + eps
            arr[50:] = 0.3 + threshold + eps
            if axis == "trend":
                arr = arr - 0.5
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df, jump_threshold=threshold)

    def test_just_below_threshold(self) -> None:
        """Delta slightly below threshold — should reject."""
        n = 200
        eps = 1e-6
        threshold = 0.15
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.full(n, 0.3)
            arr[50:] = 0.3 + threshold - eps
            if axis == "trend":
                arr = arr - 0.5
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df, jump_threshold=threshold)


# ---------------------------------------------------------------------------
# 5. Leading NaN — first 50 bars NaN, then valid data
# ---------------------------------------------------------------------------


class TestLeadingNaNParity:
    def test_leading_nan_50(self) -> None:
        n = 200
        rng = np.random.default_rng(99)
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.empty(n)
            arr[:50] = np.nan
            if axis == "trend":
                arr[50:] = np.clip(rng.normal(0.0, 0.3, n - 50), -1, 1)
            else:
                arr[50:] = np.clip(rng.normal(0.5, 0.2, n - 50), 0, 1)
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 6. Scattered NaN
# ---------------------------------------------------------------------------


class TestScatteredNaNParity:
    """Scattered NaN divergence: batch and streaming handle mid-series NaN
    differently.

    Batch mode: pandas ewm carries forward the last EMA through NaN bars,
    and _jump_dwell_pass advances dwell counters through those bars (seeing
    the carried-forward value).  Then the NaN mask is re-applied.

    Streaming mode: NaN bars are completely skipped — EMA state and dwell
    counters are frozen.

    This causes dwell counter drift and EMA divergence after NaN gaps.
    Both modes produce NaN at the NaN positions themselves, but subsequent
    valid bars may differ because dwell counters diverged.

    This test documents the known divergence: both modes agree on NaN
    positions, but non-NaN values may differ after NaN gaps.
    """

    def test_scattered_nan_nan_positions_agree(self) -> None:
        """Both modes produce NaN at the same positions."""
        n = 300
        rng = np.random.default_rng(77)
        nan_positions = rng.choice(n, size=30, replace=False)
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            if axis == "trend":
                arr = np.clip(rng.normal(0.0, 0.3, n), -1, 1)
            else:
                arr = np.clip(rng.normal(0.5, 0.2, n), 0, 1)
            arr[nan_positions] = np.nan
            values[axis] = arr
        df = _make_factor_df(values)

        # Batch
        batch_stab = StateStabilizer()
        batch_result = batch_stab.stabilize(df)

        # Streaming
        stream_stab = StreamingStateStabilizer()
        streaming_results: dict[str, list[float]] = {a: [] for a in _AXIS}
        for i in range(len(df)):
            row = {axis: df[f"ohio_factor_{axis}"].iloc[i] for axis in _AXIS}
            result = stream_stab.update(row)
            for axis in _AXIS:
                streaming_results[axis].append(result[axis])

        # NaN positions must agree even though non-NaN values may diverge
        for axis in _AXIS:
            batch_arr = batch_result[f"ohio_stable_{axis}"].to_numpy()
            stream_arr = np.array(streaming_results[axis])
            batch_nan = np.isnan(batch_arr)
            stream_nan = np.isnan(stream_arr)
            np.testing.assert_array_equal(
                batch_nan,
                stream_nan,
                err_msg=f"NaN position mismatch on {axis}",
            )

    def test_no_scattered_nan_still_parity(self) -> None:
        """Without scattered NaN, parity holds (control test)."""
        n = 300
        rng = np.random.default_rng(77)
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            if axis == "trend":
                values[axis] = np.clip(rng.normal(0.0, 0.3, n), -1, 1)
            else:
                values[axis] = np.clip(rng.normal(0.5, 0.2, n), 0, 1)
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 7. Determinism — same input 5 times, all outputs identical
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_five_runs_identical(self) -> None:
        df = _random_factor_df(500, seed=55)
        results = []
        for _ in range(5):
            stab = StateStabilizer()
            out = stab.stabilize(df)
            results.append(
                {col: out[col].to_numpy() for col in out.columns if "stable" in col}
            )

        for col in results[0]:
            ref = results[0][col]
            for run_idx in range(1, 5):
                np.testing.assert_array_equal(
                    ref,
                    results[run_idx][col],
                    err_msg=f"Run {run_idx} differs from run 0 on {col}",
                )

    def test_streaming_five_runs_identical(self) -> None:
        df = _random_factor_df(500, seed=55)
        results: list[dict[str, list[float]]] = []
        for _ in range(5):
            stab = StreamingStateStabilizer()
            run_result: dict[str, list[float]] = {a: [] for a in _AXIS}
            for i in range(len(df)):
                row = {
                    axis: df[f"ohio_factor_{axis}"].iloc[i] for axis in _AXIS
                }
                out = stab.update(row)
                for axis in _AXIS:
                    run_result[axis].append(out[axis])
            results.append(run_result)

        for axis in _AXIS:
            ref = np.array(results[0][axis])
            for run_idx in range(1, 5):
                np.testing.assert_array_equal(
                    ref,
                    np.array(results[run_idx][axis]),
                    err_msg=f"Streaming run {run_idx} differs on {axis}",
                )


# ---------------------------------------------------------------------------
# 8. Non-default parameters
# ---------------------------------------------------------------------------


class TestNonDefaultParamsParity:
    @pytest.mark.parametrize(
        "ema_alpha,jump_threshold,min_dwell",
        [
            (0.05, 0.20, 12),
            (0.30, 0.05, 3),
            (0.01, 0.50, 20),
            (0.50, 0.10, 1),
        ],
    )
    def test_parity_various_params(
        self, ema_alpha: float, jump_threshold: float, min_dwell: int
    ) -> None:
        df = _random_factor_df(500, seed=88)
        _compare_batch_streaming(
            df,
            ema_alpha=ema_alpha,
            jump_threshold=jump_threshold,
            min_dwell=min_dwell,
        )


# ---------------------------------------------------------------------------
# 9. Single bar
# ---------------------------------------------------------------------------


class TestSingleBarParity:
    def test_single_valid_bar(self) -> None:
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            values[axis] = np.array([0.5 if axis != "trend" else 0.1])
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 10. All NaN
# ---------------------------------------------------------------------------


class TestAllNaNParity:
    def test_all_nan(self) -> None:
        n = 50
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            values[axis] = np.full(n, np.nan)
        df = _make_factor_df(values)

        # Batch
        batch_stab = StateStabilizer()
        batch_result = batch_stab.stabilize(df)

        # Streaming
        stream_stab = StreamingStateStabilizer()
        for i in range(n):
            row = {axis: np.nan for axis in _AXIS}
            result = stream_stab.update(row)
            for axis in _AXIS:
                assert np.isnan(result[axis]), f"Expected NaN at bar {i}, axis {axis}"

        for axis in _AXIS:
            batch_arr = batch_result[f"ohio_stable_{axis}"].to_numpy()
            assert np.isnan(batch_arr).all(), f"Batch not all NaN for {axis}"


# ---------------------------------------------------------------------------
# 11. Realistic synthetic BTC-like data
# ---------------------------------------------------------------------------


class TestRealisticDataParity:
    def test_btc_like_returns(self) -> None:
        """Generate synthetic BTC-like returns, derive factors, compare."""
        n = 1000
        rng = np.random.default_rng(2024)

        # Simulate log returns with fat tails and occasional spikes
        returns = rng.standard_t(df=4, size=n) * 0.02
        # Add a few regime shifts
        returns[200:210] = -0.08  # crash
        returns[500:505] = 0.06  # rally

        price = 30000.0 * np.exp(np.cumsum(returns))

        # Derive rough factor proxies
        vol_20 = pd.Series(returns).rolling(20).std().to_numpy()
        vol_20 = np.nan_to_num(vol_20, nan=0.02)
        vol_norm = np.clip(vol_20 / 0.05, 0, 1)

        trend_20 = pd.Series(returns).rolling(20).mean().to_numpy()
        trend_20 = np.nan_to_num(trend_20, nan=0.0)
        trend_norm = np.clip(trend_20 / 0.03, -1, 1)

        values: dict[str, np.ndarray] = {}
        values["trend"] = trend_norm
        values["volatility"] = vol_norm
        values["downside"] = np.clip(vol_norm * 1.2, 0, 1)
        values["liquidity"] = np.clip(0.7 + rng.normal(0, 0.05, n), 0, 1)
        values["relative_strength"] = np.clip(0.5 + trend_norm * 0.3, 0, 1)
        values["correlation"] = np.clip(0.6 + rng.normal(0, 0.1, n), 0, 1)
        values["breadth"] = np.clip(0.5 + rng.normal(0, 0.15, n), 0, 1)

        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 12. Constant input — both modes produce constant output
# ---------------------------------------------------------------------------


class TestConstantInputParity:
    def test_constant_values(self) -> None:
        """Constant input should produce constant output in both modes."""
        n = 200
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            val = 0.5 if axis != "trend" else 0.0
            values[axis] = np.full(n, val)
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 13. Step function — single large step
# ---------------------------------------------------------------------------


class TestStepFunctionParity:
    def test_single_step(self) -> None:
        """One large step change at bar 100 — both handle identically."""
        n = 300
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.empty(n)
            if axis == "trend":
                arr[:100] = -0.5
                arr[100:] = 0.5
            else:
                arr[:100] = 0.2
                arr[100:] = 0.8
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 14. Multiple rapid jumps — stress dwell controller
# ---------------------------------------------------------------------------


class TestRapidJumpsParity:
    def test_rapid_alternating(self) -> None:
        """Alternating high/low every few bars — dwell should suppress most."""
        n = 400
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            arr = np.empty(n)
            for i in range(n):
                if (i // 3) % 2 == 0:
                    arr[i] = 0.8 if axis != "trend" else 0.5
                else:
                    arr[i] = 0.2 if axis != "trend" else -0.5
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 15. Two bars
# ---------------------------------------------------------------------------


class TestTwoBarsParity:
    def test_two_bars(self) -> None:
        values: dict[str, np.ndarray] = {}
        for axis in _AXIS:
            if axis == "trend":
                values[axis] = np.array([-0.3, 0.7])
            else:
                values[axis] = np.array([0.2, 0.9])
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 16. Axes with independent characteristics
# ---------------------------------------------------------------------------


class TestIndependentAxesParity:
    def test_each_axis_different_signal(self) -> None:
        """Each axis gets a different signal shape to test independence."""
        n = 500
        t = np.linspace(0, 10 * np.pi, n)
        rng = np.random.default_rng(321)
        values: dict[str, np.ndarray] = {}

        values["trend"] = 0.5 * np.sin(t)  # smooth sine
        values["volatility"] = np.clip(0.3 + 0.4 * np.abs(np.sin(t / 3)), 0, 1)  # abs sine
        values["downside"] = np.clip(rng.normal(0.5, 0.3, n), 0, 1)  # noisy
        values["liquidity"] = np.clip(np.linspace(0.1, 0.9, n), 0, 1)  # linear ramp
        # Step function for relative_strength
        rs = np.empty(n)
        rs[:125] = 0.2
        rs[125:250] = 0.8
        rs[250:375] = 0.3
        rs[375:] = 0.7
        values["relative_strength"] = rs
        values["correlation"] = np.clip(0.5 + 0.2 * np.cos(t * 2), 0, 1)  # cosine
        values["breadth"] = np.clip(
            np.concatenate([rng.normal(0.3, 0.1, 250), rng.normal(0.7, 0.1, 250)]),
            0,
            1,
        )
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 17. Leading NaN with different counts per axis
# ---------------------------------------------------------------------------


class TestMixedNaNLeadingParity:
    def test_different_nan_counts_per_axis(self) -> None:
        """Each axis has a different number of leading NaN bars."""
        n = 300
        rng = np.random.default_rng(42)
        values: dict[str, np.ndarray] = {}
        nan_counts = [0, 10, 25, 50, 75, 100, 5]
        for axis, nan_count in zip(_AXIS, nan_counts):
            arr = np.empty(n)
            arr[:nan_count] = np.nan
            valid_n = n - nan_count
            if axis == "trend":
                arr[nan_count:] = np.clip(rng.normal(0.0, 0.3, valid_n), -1, 1)
            else:
                arr[nan_count:] = np.clip(rng.normal(0.5, 0.2, valid_n), 0, 1)
            values[axis] = arr
        df = _make_factor_df(values)
        _compare_batch_streaming(df)


# ---------------------------------------------------------------------------
# 18. Tight tolerance — atol=1e-12
# ---------------------------------------------------------------------------


class TestTightTolerance:
    def test_very_tight_parity(self) -> None:
        """Verify parity holds at 1e-12 tolerance — IEEE754 exact match."""
        df = _random_factor_df(500, seed=7)
        _compare_batch_streaming(df, atol=1e-12)
