"""Tests for MetaCalculator — inference quality metadata computation.

Covers:
1.  transition_risk in [0, 1] for various inputs
2.  confidence in [0, 1] for various inputs
3.  stability in [0, 1] for various inputs
4.  data_mode = "full" when no NaN in norm features
5.  data_mode = "fallback" when some NaN (< 30%)
6.  data_mode = "degraded" when many NaN (>= 30%)
7.  High axis deltas → high transition_risk
8.  Zero axis deltas → low transition_risk
9.  All features present → higher confidence than many NaN
10. Many NaN features → low confidence
11. Stable state → high stability
12. compute_meta returns StateMeta instance
13. compute adds all 4 ohio_meta_* columns
14. High transition_risk → lower stability than low transition_risk
15. compute_meta and compute produce consistent results for equivalent input
16. No Freqtrade imports in implementation module
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.domain.models import DataMode, StateMeta
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N = 60  # enough bars for rolling windows to warm up

_AXES = [
    "trend",
    "volatility",
    "downside",
    "liquidity",
    "relative_strength",
    "correlation",
    "breadth",
]


def _make_df(
    factor_vals: float = 0.5,
    stable_vals: float = 0.5,
    norm_nan_frac: float = 0.0,
    n: int = _N,
) -> pd.DataFrame:
    """Build a minimal DataFrame with ohio_factor_*, ohio_stable_*, ohio_norm_* columns."""
    df = pd.DataFrame(index=range(n))

    for axis in _AXES:
        df[f"ohio_factor_{axis}"] = factor_vals
        df[f"ohio_stable_{axis}"] = stable_vals

    # Build norm columns with controlled NaN fraction
    n_norm = 14  # arbitrary number of norm features
    for i in range(n_norm):
        col = f"ohio_norm_feat_{i:02d}"
        values = np.ones(n) * 0.5
        if norm_nan_frac > 0.0:
            nan_indices = _RNG.choice(n, size=int(n * norm_nan_frac), replace=False)
            values[nan_indices] = np.nan
        df[col] = values

    return df


def _make_df_per_row_nan(nan_frac_per_row: float, n: int = _N) -> pd.DataFrame:
    """Build DataFrame where each row has exactly nan_frac_per_row of norm cols NaN."""
    df = pd.DataFrame(index=range(n))

    for axis in _AXES:
        df[f"ohio_factor_{axis}"] = 0.5
        df[f"ohio_stable_{axis}"] = 0.5

    n_norm = 10
    for i in range(n_norm):
        df[f"ohio_norm_feat_{i:02d}"] = 0.5

    # Set exact fraction of norm columns to NaN on every row
    norm_cols = [c for c in df.columns if c.startswith("ohio_norm_")]
    n_nan = round(len(norm_cols) * nan_frac_per_row)
    for col in norm_cols[:n_nan]:
        df[col] = np.nan

    return df


def _run(df: pd.DataFrame) -> pd.DataFrame:
    return MetaCalculator().compute(df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransitionRiskRange:
    """Test 1 — transition_risk in [0, 1]."""

    def test_range_stable_input(self) -> None:
        df = _make_df()
        result = _run(df)
        col = result["ohio_meta_transition_risk"].dropna()
        assert len(col) > 0
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_range_random_input(self) -> None:
        df = pd.DataFrame(index=range(_N))
        for axis in _AXES:
            df[f"ohio_factor_{axis}"] = _RNG.uniform(0, 1, _N)
            df[f"ohio_stable_{axis}"] = _RNG.uniform(0, 1, _N)
        for i in range(8):
            df[f"ohio_norm_feat_{i:02d}"] = _RNG.uniform(0, 1, _N)
        result = _run(df)
        col = result["ohio_meta_transition_risk"].dropna()
        assert len(col) > 0
        assert (col >= 0.0).all() and (col <= 1.0).all()


class TestConfidenceRange:
    """Test 2 — confidence in [0, 1]."""

    def test_range_stable_input(self) -> None:
        result = _run(_make_df())
        col = result["ohio_meta_confidence"].dropna()
        assert len(col) > 0
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_range_noisy_input(self) -> None:
        df = pd.DataFrame(index=range(_N))
        for axis in _AXES:
            df[f"ohio_factor_{axis}"] = _RNG.uniform(0, 1, _N)
            df[f"ohio_stable_{axis}"] = _RNG.uniform(0, 1, _N)
        result = _run(df)
        col = result["ohio_meta_confidence"].dropna()
        assert len(col) > 0
        assert (col >= 0.0).all() and (col <= 1.0).all()


class TestStabilityRange:
    """Test 3 — stability in [0, 1]."""

    def test_range_stable_input(self) -> None:
        result = _run(_make_df())
        col = result["ohio_meta_stability"].dropna()
        assert len(col) > 0
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_range_noisy_input(self) -> None:
        df = pd.DataFrame(index=range(_N))
        for axis in _AXES:
            df[f"ohio_factor_{axis}"] = _RNG.uniform(-1, 1, _N)
            df[f"ohio_stable_{axis}"] = _RNG.uniform(-1, 1, _N)
        result = _run(df)
        col = result["ohio_meta_stability"].dropna()
        assert len(col) > 0
        assert (col >= 0.0).all() and (col <= 1.0).all()


class TestDataModeFull:
    """Test 4 — data_mode = "full" when no NaN."""

    def test_full_when_no_nan(self) -> None:
        df = _make_df_per_row_nan(nan_frac_per_row=0.0)
        result = _run(df)
        assert (result["ohio_meta_data_mode"] == DataMode.FULL.value).all()


class TestDataModeFallback:
    """Test 5 — data_mode = "fallback" when NaN fraction < 30%."""

    def test_fallback_when_some_nan(self) -> None:
        # 20% of norm columns NaN on every row → fallback
        df = _make_df_per_row_nan(nan_frac_per_row=0.2)
        result = _run(df)
        assert (result["ohio_meta_data_mode"] == DataMode.FALLBACK.value).all()


class TestDataModeDegraded:
    """Test 6 — data_mode = "degraded" when NaN fraction >= 30%."""

    def test_degraded_when_many_nan(self) -> None:
        # 50% of norm columns NaN on every row → degraded
        df = _make_df_per_row_nan(nan_frac_per_row=0.5)
        result = _run(df)
        assert (result["ohio_meta_data_mode"] == DataMode.DEGRADED.value).all()

    def test_exactly_30pct_is_degraded(self) -> None:
        df = _make_df_per_row_nan(nan_frac_per_row=0.30)
        result = _run(df)
        assert (result["ohio_meta_data_mode"] == DataMode.DEGRADED.value).all()


class TestHighDeltasRaisesTransitionRisk:
    """Test 7 — high axis deltas → high transition_risk."""

    def test_large_factor_stable_gap_raises_risk(self) -> None:
        # factor=1.0, stable=0.0 → delta=1.0 everywhere → max transition_risk
        df_high = _make_df(factor_vals=1.0, stable_vals=0.0)
        df_low = _make_df(factor_vals=0.5, stable_vals=0.5)
        high_risk = _run(df_high)["ohio_meta_transition_risk"].mean()
        low_risk = _run(df_low)["ohio_meta_transition_risk"].mean()
        assert high_risk > low_risk


class TestZeroDeltasLowRisk:
    """Test 8 — zero axis deltas → low transition_risk."""

    def test_identical_factor_stable_low_risk(self) -> None:
        # Constant series → no volatility, no flips, no delta
        df = _make_df(factor_vals=0.5, stable_vals=0.5)
        result = _run(df)
        # After warm-up, transition_risk should be near 0 for constant inputs
        # (deltas=0, volatility=0 once rolling std stabilises at 0)
        warmup_skip = 15
        mean_risk = result["ohio_meta_transition_risk"].iloc[warmup_skip:].mean()
        assert mean_risk < 0.15


class TestFullDataHighConfidence:
    """Test 9 — all features present → higher confidence than many NaN."""

    def test_full_data_beats_nan_data_confidence(self) -> None:
        df_full = _make_df_per_row_nan(nan_frac_per_row=0.0)
        df_nan = _make_df_per_row_nan(nan_frac_per_row=0.6)
        conf_full = _run(df_full)["ohio_meta_confidence"].mean()
        conf_nan = _run(df_nan)["ohio_meta_confidence"].mean()
        assert conf_full > conf_nan


class TestManyNanLowConfidence:
    """Test 10 — many NaN features → low confidence."""

    def test_all_norm_nan_suppresses_confidence(self) -> None:
        # 100% NaN norm columns → nan_ratio = 1.0 → full 0.35 penalty
        df = _make_df_per_row_nan(nan_frac_per_row=1.0)
        result = _run(df)
        assert result["ohio_meta_confidence"].mean() < 0.7


class TestStableStateHighStability:
    """Test 11 — stable state → high stability."""

    def test_constant_series_yields_high_stability(self) -> None:
        df = _make_df(factor_vals=0.5, stable_vals=0.5, norm_nan_frac=0.0)
        result = _run(df)
        warmup_skip = 15
        mean_stability = result["ohio_meta_stability"].iloc[warmup_skip:].mean()
        assert mean_stability > 0.70


class TestComputeMetaReturnType:
    """Test 12 — compute_meta returns StateMeta instance."""

    def test_returns_state_meta(self) -> None:
        row = {f"ohio_factor_{a}": 0.5 for a in _AXES}
        row.update({f"ohio_stable_{a}": 0.5 for a in _AXES})
        row.update({f"ohio_norm_feat_{i:02d}": 0.5 for i in range(5)})
        meta = MetaCalculator().compute_meta(row)
        assert isinstance(meta, StateMeta)

    def test_fields_in_range(self) -> None:
        row = {f"ohio_factor_{a}": 0.3 for a in _AXES}
        row.update({f"ohio_stable_{a}": 0.5 for a in _AXES})
        row.update({f"ohio_norm_feat_{i:02d}": 0.5 for i in range(6)})
        meta = MetaCalculator().compute_meta(row)
        assert 0.0 <= meta.transition_risk <= 1.0
        assert 0.0 <= meta.confidence <= 1.0
        assert 0.0 <= meta.stability <= 1.0
        assert isinstance(meta.data_mode, DataMode)


class TestComputeOutputColumns:
    """Test 13 — compute adds all 4 ohio_meta_* columns."""

    def test_all_four_columns_present(self) -> None:
        df = _make_df()
        result = _run(df)
        expected = {
            "ohio_meta_transition_risk",
            "ohio_meta_confidence",
            "ohio_meta_stability",
            "ohio_meta_data_mode",
        }
        assert expected.issubset(set(result.columns))

    def test_length_unchanged(self) -> None:
        df = _make_df(n=50)
        result = _run(df)
        assert len(result) == 50


class TestHighRiskLowStability:
    """Test 14 — high transition_risk scenario yields lower stability."""

    def test_high_delta_scenario_lower_stability(self) -> None:
        # Large factor-stable gap → high risk → lower stability
        df_volatile = _make_df(factor_vals=1.0, stable_vals=0.0)
        df_calm = _make_df(factor_vals=0.5, stable_vals=0.5)
        stab_volatile = _run(df_volatile)["ohio_meta_stability"].mean()
        stab_calm = _run(df_calm)["ohio_meta_stability"].mean()
        assert stab_calm > stab_volatile


class TestConsistency:
    """Test 15 — compute_meta and compute produce consistent results."""

    def test_single_row_matches_dataframe_last_row(self) -> None:
        # Build a DataFrame where factors and stables are constant — no rolling
        # effects — so the single-row compute_meta and vectorized compute agree.
        n = 40
        df = pd.DataFrame(index=range(n))
        for axis in _AXES:
            df[f"ohio_factor_{axis}"] = 0.6
            df[f"ohio_stable_{axis}"] = 0.4

        n_norm = 6
        for i in range(n_norm):
            df[f"ohio_norm_feat_{i:02d}"] = 0.5

        result = _run(df)

        # Single-row path (no rolling context, so volatility/flip defaults to 0)
        row = {f"ohio_factor_{a}": 0.6 for a in _AXES}
        row.update({f"ohio_stable_{a}": 0.4 for a in _AXES})
        row.update({f"ohio_norm_feat_{i:02d}": 0.5 for i in range(n_norm)})
        meta = MetaCalculator().compute_meta(row)

        # Both should agree on NaN-ratio-driven data_mode
        last_mode = result["ohio_meta_data_mode"].iloc[-1]
        assert last_mode == meta.data_mode.value

        # transition_risk direction: both detect non-zero delta
        df_risk = result["ohio_meta_transition_risk"].iloc[-1]
        assert abs(df_risk - meta.transition_risk) < 0.40  # same ballpark


class TestNoFreqtradeImports:
    """Test 16 — implementation module has no Freqtrade framework imports."""

    def test_no_freqtrade_framework_imports(self) -> None:
        module_path = Path(__file__).parents[2] / (
            "freqtrade/ohio/core/market_state/meta_calculator.py"
        )
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Domain model imports (freqtrade.ohio.*) are explicitly allowed.
        # Any other freqtrade.* import is forbidden.
        _ALLOWED_PREFIX = "freqtrade.ohio"

        forbidden: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("freqtrade") and not alias.name.startswith(
                        _ALLOWED_PREFIX
                    ):
                        forbidden.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.startswith("freqtrade") and not mod.startswith(_ALLOWED_PREFIX):
                    forbidden.append(mod)

        assert forbidden == [], f"Forbidden freqtrade imports found: {forbidden}"


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_dataframe_returns_meta_columns(self) -> None:
        df = pd.DataFrame(columns=[f"ohio_factor_{a}" for a in _AXES])
        result = MetaCalculator().compute(df)
        assert "ohio_meta_transition_risk" in result.columns
        assert "ohio_meta_confidence" in result.columns
        assert "ohio_meta_stability" in result.columns
        assert "ohio_meta_data_mode" in result.columns

    def test_missing_factor_columns_does_not_crash(self) -> None:
        # DataFrame with no ohio_factor_* or ohio_stable_* at all
        df = pd.DataFrame({"close": np.ones(_N)})
        result = MetaCalculator().compute(df)
        assert "ohio_meta_transition_risk" in result.columns

    def test_compute_meta_nan_norm_features(self) -> None:
        row = {f"ohio_factor_{a}": 0.5 for a in _AXES}
        row.update({f"ohio_stable_{a}": 0.5 for a in _AXES})
        row.update({f"ohio_norm_feat_{i:02d}": float("nan") for i in range(5)})
        meta = MetaCalculator().compute_meta(row)
        assert meta.data_mode == DataMode.DEGRADED
        assert 0.0 <= meta.confidence <= 1.0

    def test_compute_meta_no_norm_features(self) -> None:
        # No ohio_norm_* keys → nan_ratio = 1.0 → degraded
        row = {f"ohio_factor_{a}": 0.5 for a in _AXES}
        row.update({f"ohio_stable_{a}": 0.5 for a in _AXES})
        meta = MetaCalculator().compute_meta(row)
        assert meta.data_mode == DataMode.DEGRADED
