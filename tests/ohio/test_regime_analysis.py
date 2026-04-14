"""Tests for scripts/regime_analysis.py — regime clustering on OHIO state vectors.

Covers:
  1. K-means with synthetic 7-axis data recovers 3 known clusters
  2. Silhouette-based k selection picks a reasonable k
  3. Transition matrix rows sum to ~1.0
  4. Per-regime metrics have expected columns
  5. NaN rows are properly excluded from clustering
  6. cluster_regimes with k_override respects the override
  7. compute_transition_matrix handles edge case of single regime
  8. parse_args defaults are correct
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.regime_analysis import (
    STABLE_AXES,
    cluster_regimes,
    compute_regime_metrics,
    compute_transition_matrix,
    parse_args,
    select_best_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 300, n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with 7 stable axes drawn from *n_clusters* well-separated blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, size=(n_clusters, len(STABLE_AXES))) * 3
    rows_per = n // n_clusters
    data_parts: list[np.ndarray] = []
    true_labels: list[int] = []

    for i, center in enumerate(centers):
        chunk = rng.normal(loc=center, scale=0.15, size=(rows_per, len(STABLE_AXES)))
        data_parts.append(chunk)
        true_labels.extend([i] * rows_per)

    arr = np.vstack(data_parts)
    df = pd.DataFrame(arr, columns=STABLE_AXES)

    # Add supporting columns the metrics function expects
    df["close"] = 100 + rng.normal(0, 1, size=len(df)).cumsum() * 0.01
    df["ohio_active_mode"] = rng.choice(
        ["trend_following", "mean_reversion", "breakout", "defensive"], size=len(df),
    )
    df["ohio_meta_transition_risk"] = rng.uniform(0, 1, size=len(df))
    df["ohio_stable_volatility"] = df["ohio_stable_volatility"]  # already present
    df["ohio_stable_trend"] = df["ohio_stable_trend"]  # already present
    for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
        df[f"ohio_fitness_{mode}"] = rng.uniform(0, 1, size=len(df))

    df["_true_label"] = true_labels
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKmeansRecovery:
    """Test 1: K-means with synthetic 7-axis data recovers 3 known clusters."""

    def test_recovers_known_clusters(self) -> None:
        df = _make_synthetic_df(n=300, n_clusters=3)
        df_out, best_k, _ = cluster_regimes(df, k_override=3, random_state=42)

        # All non-NaN rows should be assigned a regime in {0, 1, 2}
        assigned = df_out[df_out["regime"] >= 0]
        assert len(assigned) == len(df)
        assert set(assigned["regime"].unique()) == {0, 1, 2}

        # Cluster assignments should mostly agree with true labels
        # (label indices may differ, so check via contingency)
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(df["_true_label"].values, assigned["regime"].values)
        assert ari > 0.8, f"Adjusted Rand Index too low: {ari:.3f}"


class TestSilhouetteSelection:
    """Test 2: Silhouette-based k selection picks a reasonable k."""

    def test_selects_reasonable_k(self) -> None:
        df = _make_synthetic_df(n=300, n_clusters=3)
        features = df[STABLE_AXES].values
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(features)

        best_k, scores = select_best_k(scaled, candidates=[2, 3, 4, 5], random_state=42)

        # With 3 well-separated clusters, best k should be 3
        assert best_k == 3, f"Expected k=3, got k={best_k}. Scores: {scores}"
        assert len(scores) == 4
        assert all(isinstance(v, float) for v in scores.values())


class TestTransitionMatrix:
    """Test 3: Transition matrix rows sum to ~1.0."""

    def test_rows_sum_to_one(self) -> None:
        df = _make_synthetic_df(n=300, n_clusters=3)
        df, _, _ = cluster_regimes(df, k_override=3, random_state=42)

        tm = compute_transition_matrix(df, n_regimes=3)

        assert tm.shape == (3, 3)
        row_sums = tm.values.sum(axis=1)
        for i, s in enumerate(row_sums):
            assert abs(s - 1.0) < 1e-9, f"Row {i} sums to {s}, expected ~1.0"

    def test_single_regime_matrix(self) -> None:
        """Edge case: only 1 regime — row should sum to 1.0."""
        df = pd.DataFrame({"regime": [0, 0, 0, 0, 0]})
        tm = compute_transition_matrix(df, n_regimes=1)
        assert tm.shape == (1, 1)
        assert abs(tm.values[0, 0] - 1.0) < 1e-9


class TestRegimeMetrics:
    """Test 4: Per-regime metrics have expected columns."""

    def test_expected_columns(self) -> None:
        df = _make_synthetic_df(n=300, n_clusters=3)
        df, _, _ = cluster_regimes(df, k_override=3, random_state=42)

        metrics = compute_regime_metrics(df)

        expected_cols = {
            "bar_count",
            "pct_of_total",
            "avg_return_1h",
            "avg_volatility",
            "avg_trend",
            "dominant_mode",
            "mode_distribution",
            "avg_transition_risk",
            "avg_fitness_trend_following",
            "avg_fitness_mean_reversion",
            "avg_fitness_breakout",
            "avg_fitness_defensive",
        }
        assert expected_cols.issubset(set(metrics.columns)), (
            f"Missing columns: {expected_cols - set(metrics.columns)}"
        )

    def test_pct_of_total_sums_to_100(self) -> None:
        df = _make_synthetic_df(n=300, n_clusters=3)
        df, _, _ = cluster_regimes(df, k_override=3, random_state=42)
        metrics = compute_regime_metrics(df)
        total_pct = metrics["pct_of_total"].sum()
        assert abs(total_pct - 100.0) < 0.1, f"pct_of_total sums to {total_pct}"


class TestNaNExclusion:
    """Test 5: NaN rows are properly excluded from clustering."""

    def test_nan_rows_get_regime_minus_one(self) -> None:
        df = _make_synthetic_df(n=100, n_clusters=3)

        # Inject NaN into 10 rows
        nan_indices = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        for idx in nan_indices:
            if idx < len(df):
                df.loc[idx, STABLE_AXES[0]] = np.nan

        df_out, _, _ = cluster_regimes(df, k_override=3, random_state=42)

        # NaN rows should have regime = -1
        for idx in nan_indices:
            if idx < len(df_out):
                assert df_out.loc[idx, "regime"] == -1, (
                    f"Row {idx} should have regime=-1 but got {df_out.loc[idx, 'regime']}"
                )

        # Non-NaN rows should have regime >= 0
        valid = df_out[~df_out.index.isin(nan_indices)]
        assert (valid["regime"] >= 0).all()

    def test_all_nan_produces_all_minus_one(self) -> None:
        df = pd.DataFrame({col: [np.nan] * 10 for col in STABLE_AXES})
        df["close"] = 100.0
        df_out, best_k, _ = cluster_regimes(df, k_override=3, random_state=42)
        # When no valid rows exist, KMeans cannot run; all should be -1
        # The function should handle this gracefully
        assert (df_out["regime"] == -1).all()


class TestKOverride:
    """Test 6: cluster_regimes with k_override respects the override."""

    def test_override_k(self) -> None:
        df = _make_synthetic_df(n=300, n_clusters=3)
        _, k_used, scores = cluster_regimes(df, k_override=5, random_state=42)
        assert k_used == 5
        assert scores == {}  # no silhouette search when overridden


class TestParseArgs:
    """Test 8: parse_args defaults are correct."""

    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.pair == "BTC_USDT"
        assert args.k is None
        assert args.output_dir is None

    def test_custom_args(self) -> None:
        args = parse_args(["--pair", "ETH_USDT", "--k", "4", "--output-dir", "/tmp/out"])
        assert args.pair == "ETH_USDT"
        assert args.k == 4
        assert args.output_dir == "/tmp/out"
