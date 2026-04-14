"""Unit tests for scripts/full_backtest.py helper functions.

Tests determinism checking, signal analysis, and sanity check logic
using mock DataFrames — no real candle data required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — mirror the script's own sys.path trick
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.full_backtest import (
    analyse_signals,
    check_determinism,
    run_sanity_checks,
    _WARMUP_BARS,
    _STABLE_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_df(
    n_bars: int = 5000,
    *,
    entry_rate: float = 0.30,
    modes: list[str] | None = None,
    confidence_positive_ratio: float = 0.95,
    inject_stable_nan: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic DataFrame that looks like full-pipeline output.

    Includes ohio_stable_*, ohio_meta_*, ohio_fitness_*, ohio_policy_*,
    and ohio_active_mode columns with controllable characteristics.
    """
    rng = np.random.RandomState(seed)
    if modes is None:
        modes = ["trend_following", "mean_reversion", "breakout", "defensive"]

    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({
        "date": dates,
        "open": 100 + rng.randn(n_bars).cumsum() * 0.1,
        "high": 101 + rng.randn(n_bars).cumsum() * 0.1,
        "low": 99 + rng.randn(n_bars).cumsum() * 0.1,
        "close": 100 + rng.randn(n_bars).cumsum() * 0.1,
        "volume": rng.uniform(100, 1000, n_bars),
    })

    # ohio_stable_* columns
    for col in _STABLE_COLS:
        vals = rng.uniform(0, 1, n_bars)
        if inject_stable_nan:
            # Sprinkle NaN in the post-warmup region
            nan_idx = rng.choice(
                range(_WARMUP_BARS, n_bars), size=5, replace=False
            )
            vals[nan_idx] = np.nan
        df[col] = vals

    # ohio_meta_* columns
    df["ohio_meta_transition_risk"] = rng.uniform(0, 1, n_bars)
    conf = rng.uniform(0, 1, n_bars)
    # Force a portion to be <= 0 to test confidence check
    post_warmup_count = max(0, n_bars - _WARMUP_BARS)
    zero_count = int(post_warmup_count * (1 - confidence_positive_ratio))
    if zero_count > 0 and post_warmup_count > 0:
        idx = rng.choice(
            range(_WARMUP_BARS, n_bars),
            size=min(zero_count, post_warmup_count),
            replace=False,
        )
        conf[idx] = 0.0
    # Ensure post-warmup bars that aren't zeroed have positive confidence
    if post_warmup_count > 0:
        for i in range(_WARMUP_BARS, n_bars):
            if conf[i] == 0.0:
                continue
            conf[i] = max(conf[i], 0.01)  # ensure > 0
    df["ohio_meta_confidence"] = conf
    df["ohio_meta_stability"] = rng.uniform(0, 1, n_bars)
    df["ohio_meta_data_mode"] = "full"

    # ohio_fitness_* columns
    for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
        df[f"ohio_fitness_{mode}"] = rng.uniform(0.2, 0.9, n_bars)

    # ohio_active_mode — pick from provided modes
    df["ohio_active_mode"] = rng.choice(modes, n_bars)

    # ohio_policy_enabled — control entry rate in post-warmup region
    # Use the smaller of _WARMUP_BARS and n_bars as the warmup boundary
    warmup_end = min(_WARMUP_BARS, n_bars)
    enabled = np.zeros(n_bars, dtype=bool)
    active_count = n_bars - warmup_end
    if active_count > 0:
        entry_count = int(active_count * entry_rate)
        entry_idx = rng.choice(
            range(warmup_end, n_bars),
            size=min(entry_count, active_count),
            replace=False,
        )
        enabled[entry_idx] = True
    df["ohio_policy_enabled"] = enabled

    df["ohio_policy_size_multiplier"] = rng.uniform(0.3, 1.5, n_bars)
    df["ohio_policy_entry_threshold_adj"] = rng.uniform(-0.05, 0.15, n_bars)
    df["ohio_policy_max_positions"] = rng.randint(1, 7, n_bars)

    return df


# ---------------------------------------------------------------------------
# Test 1: determinism check passes with identical DataFrames
# ---------------------------------------------------------------------------

class TestDeterminismCheckIdentical:
    """check_determinism should pass when the pipeline is deterministic.

    We monkeypatch run_pipeline to return a known DataFrame, ensuring
    two identical runs succeed.
    """

    def test_identical_frames_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fixed_result = _make_pipeline_df(n_bars=5000, seed=99)

        call_count = 0

        def _fake_pipeline(df: pd.DataFrame) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return fixed_result.copy()

        import scripts.full_backtest as mod
        monkeypatch.setattr(mod, "run_pipeline", _fake_pipeline)

        result_df, t1, t2 = check_determinism(pd.DataFrame({"dummy": [1]}))
        assert call_count == 2, "Pipeline should be called exactly twice"
        pd.testing.assert_frame_equal(result_df, fixed_result)
        assert t1 >= 0 and t2 >= 0


# ---------------------------------------------------------------------------
# Test 2: determinism check fails with different DataFrames
# ---------------------------------------------------------------------------

class TestDeterminismCheckDifferent:
    """check_determinism should raise AssertionError when outputs differ."""

    def test_different_frames_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        df_a = _make_pipeline_df(n_bars=5000, seed=1)
        df_b = _make_pipeline_df(n_bars=5000, seed=2)

        toggle = {"call": 0}

        def _fake_pipeline(df: pd.DataFrame) -> pd.DataFrame:
            toggle["call"] += 1
            return df_a.copy() if toggle["call"] == 1 else df_b.copy()

        import scripts.full_backtest as mod
        monkeypatch.setattr(mod, "run_pipeline", _fake_pipeline)

        with pytest.raises(AssertionError, match="DETERMINISM FAILURE"):
            check_determinism(pd.DataFrame({"dummy": [1]}))


# ---------------------------------------------------------------------------
# Test 3: signal analysis metrics extraction
# ---------------------------------------------------------------------------

class TestSignalAnalysis:
    """analyse_signals should extract correct metrics from a mock DataFrame."""

    def test_metrics_basic(self) -> None:
        df = _make_pipeline_df(n_bars=5000, entry_rate=0.25, seed=10)
        metrics = analyse_signals(df, "TEST_PAIR")

        assert metrics["pair"] == "TEST_PAIR"
        assert metrics["total_bars"] == 5000
        assert metrics["warmup_bars"] == _WARMUP_BARS
        assert metrics["active_bars"] == 5000 - _WARMUP_BARS

        # Entry rate should be close to 0.25 (within tolerance due to rounding)
        assert 0.15 <= metrics["entry_rate"] <= 0.35, (
            f"Expected entry rate ~0.25, got {metrics['entry_rate']:.3f}"
        )

        # Mode distribution should have entries for all 4 modes
        assert len(metrics["mode_distribution"]) >= 2

        # Average fitness should be in [0, 1]
        assert 0.0 <= metrics["avg_fitness"] <= 1.0

        # Transition risk should be in [0, 1]
        assert 0.0 <= metrics["avg_transition_risk"] <= 1.0

    def test_single_mode_distribution(self) -> None:
        df = _make_pipeline_df(
            n_bars=5000, entry_rate=0.20, modes=["defensive"], seed=20
        )
        metrics = analyse_signals(df, "SINGLE_MODE")

        assert len(metrics["mode_distribution"]) == 1
        assert "defensive" in metrics["mode_distribution"]
        assert metrics["mode_distribution"]["defensive"] == 100.0


# ---------------------------------------------------------------------------
# Test 4: sanity check bounds
# ---------------------------------------------------------------------------

class TestSanityChecks:
    """run_sanity_checks should flag violations and pass clean data."""

    def test_clean_data_no_warnings(self) -> None:
        df = _make_pipeline_df(
            n_bars=5000,
            entry_rate=0.30,
            confidence_positive_ratio=0.95,
            inject_stable_nan=False,
        )
        metrics = analyse_signals(df, "CLEAN")
        warnings = run_sanity_checks(df, metrics)
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    def test_low_entry_rate_warning(self) -> None:
        df = _make_pipeline_df(n_bars=5000, entry_rate=0.02)
        metrics = analyse_signals(df, "LOW_ENTRY")
        warnings = run_sanity_checks(df, metrics)
        rate_warnings = [w for w in warnings if "entry rate too low" in w]
        assert len(rate_warnings) == 1

    def test_zero_entry_rate_warning(self) -> None:
        df = _make_pipeline_df(n_bars=5000, entry_rate=0.0)
        metrics = analyse_signals(df, "ZERO_ENTRY")
        warnings = run_sanity_checks(df, metrics)
        rate_warnings = [w for w in warnings if "entry rate" in w]
        assert len(rate_warnings) >= 1

    def test_single_mode_warning(self) -> None:
        df = _make_pipeline_df(
            n_bars=5000, entry_rate=0.30, modes=["breakout"]
        )
        metrics = analyse_signals(df, "SINGLE")
        warnings = run_sanity_checks(df, metrics)
        mode_warnings = [w for w in warnings if "mode(s) used" in w]
        assert len(mode_warnings) == 1

    def test_stable_nan_warning(self) -> None:
        df = _make_pipeline_df(
            n_bars=5000, entry_rate=0.30, inject_stable_nan=True
        )
        metrics = analyse_signals(df, "NAN_STABLE")
        warnings = run_sanity_checks(df, metrics)
        nan_warnings = [w for w in warnings if "NaN values after warmup" in w]
        assert len(nan_warnings) >= 1

    def test_low_confidence_warning(self) -> None:
        df = _make_pipeline_df(
            n_bars=5000,
            entry_rate=0.30,
            confidence_positive_ratio=0.50,
        )
        metrics = analyse_signals(df, "LOW_CONF")
        warnings = run_sanity_checks(df, metrics)
        conf_warnings = [w for w in warnings if "ohio_meta_confidence" in w]
        assert len(conf_warnings) == 1
