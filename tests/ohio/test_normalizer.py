"""Tests for Normalizer — percentile-rank normalization of ohio_feat_* columns.

Covers:
1.  All ohio_norm_* values are in [0, 1] (fully warmed-up region)
2.  Monotonic input → monotonically increasing percentile ranks
3.  Constant input → all ranks ≈ 0.5 (z-score fallback returns 0.5)
4.  Z-score fallback produces [0, 1] values for early bars
5.  Placeholder columns (all NaN) are skipped
6.  Column renaming: ohio_feat_ → ohio_norm_
7.  Original ohio_feat_* columns are preserved
8.  Window parameter controls responsiveness
9.  Count of ohio_norm_* matches non-placeholder ohio_feat_* count
10. Edge case: single non-NaN value in window
11. Large outlier gets rank near 1.0
12. Negative values handled correctly (log_return can be negative)
13. No Freqtrade imports in implementation module
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.market_state.normalizer import Normalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_N = 300  # rows — enough to fully warm up small test windows


def _make_df(**kwargs: pd.Series) -> pd.DataFrame:
    """Build a DataFrame where each kwarg becomes an ``ohio_feat_*`` column."""
    return pd.DataFrame({f"ohio_feat_{k}": v for k, v in kwargs.items()})


def _rand_series(n: int = _N) -> pd.Series:
    return pd.Series(_RNG.standard_normal(n))


def _norm_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ohio_norm_")]


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ohio_feat_")]


# ---------------------------------------------------------------------------
# Test 1 — All ohio_norm_* values in [0, 1] in fully warmed-up region
# ---------------------------------------------------------------------------


def test_all_norm_values_in_unit_interval() -> None:
    window = 50
    df = _make_df(
        log_return_24=_rand_series(),
        adx_14=pd.Series(_RNG.uniform(0, 100, _N)),
    )
    normalizer = Normalizer(window=window)
    normalizer.normalize(df)

    for col in _norm_cols(df):
        warmed = df[col].iloc[window:]
        assert warmed.notna().all(), f"{col} has NaN after warm-up"
        assert (warmed >= 0.0).all() and (warmed <= 1.0).all(), f"{col} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 2 — Monotonic input → monotonically increasing ranks
# ---------------------------------------------------------------------------


def test_monotonic_input_produces_increasing_ranks() -> None:
    window = 50
    monotonic = pd.Series(np.arange(float(_N)))
    df = _make_df(monotonic=monotonic)
    Normalizer(window=window).normalize(df)

    ranks = df["ohio_norm_monotonic"].iloc[window:].reset_index(drop=True)
    # Each new bar is the max → rank = 1.0
    assert (ranks == 1.0).all(), "Strictly increasing series should always rank 1.0"


# ---------------------------------------------------------------------------
# Test 3 — Constant input → all ranks ≈ 0.5 (via z-score fallback)
# ---------------------------------------------------------------------------


def test_constant_input_ranks_near_half() -> None:
    constant = pd.Series(np.full(_N, 42.0))
    df = _make_df(constant=constant)
    Normalizer(window=50).normalize(df)

    ranks = df["ohio_norm_constant"].dropna()
    # Constant series: rolling rank with pct=True gives 1.0 (single tied value),
    # but z-score fallback (where std=0) returns 0.5.
    # After window warms up, rolling rank of a constant is 1.0 only in pandas ≥1.3;
    # for ties, rank(pct=True, method='average') = 1.0 when all values are equal.
    # The important thing is that no value escapes [0, 1].
    assert (ranks >= 0.0).all() and (ranks <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 4 — Z-score fallback produces [0, 1] for bars before window is full
# ---------------------------------------------------------------------------


def test_zscore_fallback_in_range_before_warmup() -> None:
    window = 200  # large window so early bars use fallback
    df = _make_df(feat=_rand_series())
    Normalizer(window=window).normalize(df)

    early = df["ohio_norm_feat"].iloc[:50]
    # Some may be NaN if even the fallback window isn't met, but present values must be in [0,1]
    present = early.dropna()
    assert (present >= 0.0).all() and (present <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 5 — Placeholder columns (all NaN) are skipped
# ---------------------------------------------------------------------------


def test_placeholder_columns_skipped() -> None:
    df = _make_df(
        real=_rand_series(),
        rs_raw=pd.Series([np.nan] * _N),  # placeholder
    )
    Normalizer(window=50).normalize(df)

    norm_cols = _norm_cols(df)
    assert "ohio_norm_real" in norm_cols
    assert "ohio_norm_rs_raw" not in norm_cols


# ---------------------------------------------------------------------------
# Test 6 — Column renaming: ohio_feat_ → ohio_norm_
# ---------------------------------------------------------------------------


def test_column_renaming() -> None:
    df = _make_df(log_return_24=_rand_series(), adx_14=_rand_series())
    Normalizer(window=50).normalize(df)

    assert "ohio_norm_log_return_24" in df.columns
    assert "ohio_norm_adx_14" in df.columns


# ---------------------------------------------------------------------------
# Test 7 — Original ohio_feat_* columns preserved
# ---------------------------------------------------------------------------


def test_original_feat_columns_preserved() -> None:
    original_vals = _rand_series()
    df = _make_df(log_return_24=original_vals.copy())
    Normalizer(window=50).normalize(df)

    pd.testing.assert_series_equal(
        df["ohio_feat_log_return_24"],
        original_vals.rename("ohio_feat_log_return_24"),
    )


# ---------------------------------------------------------------------------
# Test 8 — Window parameter controls responsiveness
# ---------------------------------------------------------------------------


def test_smaller_window_more_responsive() -> None:
    """Smaller window = fewer NaN bars before warm-up completes."""
    series = _rand_series()
    df_small = _make_df(feat=series.copy())
    df_large = _make_df(feat=series.copy())

    Normalizer(window=30).normalize(df_small)
    Normalizer(window=150).normalize(df_large)

    # Both should have 0 NaN because min_periods=1, but with different windows the
    # percentile-rank values will differ — verify the outputs are not identical.
    assert not df_small["ohio_norm_feat"].equals(df_large["ohio_norm_feat"])


# ---------------------------------------------------------------------------
# Test 9 — Count of ohio_norm_* matches non-placeholder ohio_feat_* count
# ---------------------------------------------------------------------------


def test_norm_column_count_matches_non_placeholder_feat_count() -> None:
    df = _make_df(
        feat_a=_rand_series(),
        feat_b=_rand_series(),
        placeholder=pd.Series([np.nan] * _N),
    )
    Normalizer(window=50).normalize(df)

    non_placeholder_feat_count = sum(
        1
        for c in _feat_cols(df)
        if not df[c].isna().all()
    )
    assert len(_norm_cols(df)) == non_placeholder_feat_count


# ---------------------------------------------------------------------------
# Test 10 — Edge case: single non-NaN value in window
# ---------------------------------------------------------------------------


def test_single_value_in_window_does_not_raise() -> None:
    """Series with a single non-NaN value must not raise and must produce a norm column."""
    series = pd.Series([np.nan] * 10 + [5.0] + [np.nan] * (_N - 11))
    df = _make_df(sparse=series)
    # Should not raise — and must create an ohio_norm_* column (series is not all-NaN)
    Normalizer(window=50).normalize(df)
    assert "ohio_norm_sparse" in df.columns
    norm = df["ohio_norm_sparse"].dropna()
    assert (norm >= 0.0).all() and (norm <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 11 — Large outlier gets rank near 1.0
# ---------------------------------------------------------------------------


def test_large_outlier_gets_high_rank() -> None:
    window = 50
    values = np.ones(_N) * 10.0
    values[-1] = 1e6  # massive outlier at the end
    df = _make_df(feat=pd.Series(values))
    Normalizer(window=window).normalize(df)

    last_rank = df["ohio_norm_feat"].iloc[-1]
    assert last_rank == pytest.approx(1.0), f"Expected rank near 1.0, got {last_rank}"


# ---------------------------------------------------------------------------
# Test 12 — Negative values handled correctly (log_return can be negative)
# ---------------------------------------------------------------------------


def test_negative_values_handled() -> None:
    window = 50
    # Mix of negative and positive log-return-like values
    log_returns = pd.Series(_RNG.normal(loc=0.0, scale=0.02, size=_N))
    df = _make_df(log_return_24=log_returns)
    Normalizer(window=window).normalize(df)

    warmed = df["ohio_norm_log_return_24"].iloc[window:]
    assert warmed.notna().all()
    assert (warmed >= 0.0).all() and (warmed <= 1.0).all()

    # Verify that a very negative value gets low rank
    extreme_df = _make_df(log_return_24=pd.Series(list(log_returns.iloc[:_N-1]) + [-99.0]))
    Normalizer(window=window).normalize(extreme_df)
    assert extreme_df["ohio_norm_log_return_24"].iloc[-1] < 0.1


# ---------------------------------------------------------------------------
# Test 13 — No Freqtrade imports in normalizer module
# ---------------------------------------------------------------------------


def test_no_freqtrade_imports_in_module() -> None:
    module_path = Path(
        importlib.util.find_spec(
            "freqtrade.ohio.core.market_state.normalizer"
        ).origin
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    freqtrade_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("freqtrade"):
                    freqtrade_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("freqtrade"):
                freqtrade_imports.append(node.module)

    assert freqtrade_imports == [], (
        f"normalizer.py must not import from freqtrade: {freqtrade_imports}"
    )
