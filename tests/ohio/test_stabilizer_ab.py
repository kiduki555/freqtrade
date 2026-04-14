"""Tests for scripts/stabilizer_ab_test.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure freqtrade root is importable
_FREQTRADE_ROOT = str(Path(__file__).resolve().parents[2])
if _FREQTRADE_ROOT not in sys.path:
    sys.path.insert(0, _FREQTRADE_ROOT)

from scripts.stabilizer_ab_test import (
    EMA_ALPHAS,
    JUMP_THRESHOLDS,
    MIN_DWELLS,
    STRATEGY_MODES,
    build_param_grid,
    extract_metrics,
    parse_args,
)


# ---------------------------------------------------------------------------
# 1. Grid generates exactly 125 combinations
# ---------------------------------------------------------------------------


class TestParamGrid:
    def test_grid_size(self) -> None:
        grid = build_param_grid()
        assert len(grid) == 125

    def test_grid_size_matches_product(self) -> None:
        expected = len(EMA_ALPHAS) * len(JUMP_THRESHOLDS) * len(MIN_DWELLS)
        assert len(build_param_grid()) == expected

    def test_grid_keys(self) -> None:
        grid = build_param_grid()
        for params in grid:
            assert set(params.keys()) == {"ema_alpha", "jump_threshold", "min_dwell"}

    def test_grid_unique_combinations(self) -> None:
        grid = build_param_grid()
        tuples = [(p["ema_alpha"], p["jump_threshold"], p["min_dwell"]) for p in grid]
        assert len(set(tuples)) == 125


# ---------------------------------------------------------------------------
# 2. Metric extraction from a mock DataFrame
# ---------------------------------------------------------------------------


def _make_mock_df(n_rows: int = 100) -> pd.DataFrame:
    """Build a DataFrame that mimics full-pipeline output columns."""
    rng = np.random.RandomState(42)
    modes = rng.choice(STRATEGY_MODES, size=n_rows)
    return pd.DataFrame(
        {
            "ohio_active_mode": modes,
            "ohio_meta_transition_risk": rng.uniform(0, 1, n_rows),
            "ohio_meta_confidence": rng.uniform(0, 1, n_rows),
            "ohio_meta_stability": rng.uniform(0, 1, n_rows),
        }
    )


class TestMetricExtraction:
    def test_all_keys_present(self) -> None:
        df = _make_mock_df()
        metrics = extract_metrics(df)
        expected_keys = {
            "regime_change_count",
            "avg_dwell_bars",
            "avg_transition_risk",
            "avg_confidence",
            "avg_stability",
        }
        for mode in STRATEGY_MODES:
            expected_keys.add(f"pct_{mode}")
        assert expected_keys.issubset(set(metrics.keys()))

    def test_regime_change_count_nonnegative(self) -> None:
        df = _make_mock_df()
        metrics = extract_metrics(df)
        assert metrics["regime_change_count"] >= 0

    def test_pct_sums_to_100(self) -> None:
        df = _make_mock_df()
        metrics = extract_metrics(df)
        total_pct = sum(metrics[f"pct_{m}"] for m in STRATEGY_MODES)
        assert abs(total_pct - 100.0) < 0.1

    def test_averages_in_range(self) -> None:
        df = _make_mock_df()
        metrics = extract_metrics(df)
        for key in ("avg_transition_risk", "avg_confidence", "avg_stability"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_constant_mode_zero_changes(self) -> None:
        """If ohio_active_mode never changes, regime_change_count == 0."""
        df = pd.DataFrame(
            {
                "ohio_active_mode": ["defensive"] * 50,
                "ohio_meta_transition_risk": [0.1] * 50,
                "ohio_meta_confidence": [0.8] * 50,
                "ohio_meta_stability": [0.9] * 50,
            }
        )
        metrics = extract_metrics(df)
        assert metrics["regime_change_count"] == 0
        assert metrics["pct_defensive"] == 100.0

    def test_missing_columns_handled(self) -> None:
        """extract_metrics should not crash on a bare DataFrame."""
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        metrics = extract_metrics(df)
        assert metrics["regime_change_count"] == 0
        assert np.isnan(metrics["avg_stability"])


# ---------------------------------------------------------------------------
# 3. CSV output has expected columns
# ---------------------------------------------------------------------------


class TestCSVOutput:
    def test_results_df_columns(self, tmp_path: Path) -> None:
        """Simulate building a results DataFrame and verify column set."""
        grid = build_param_grid()[:3]  # just 3 combos
        rows = []
        for params in grid:
            mock_metrics = extract_metrics(_make_mock_df(20))
            rows.append({**params, **mock_metrics})

        results_df = pd.DataFrame(rows)
        csv_path = tmp_path / "test_results.csv"
        results_df.to_csv(csv_path, index=False)

        loaded = pd.read_csv(csv_path)
        expected_cols = {
            "ema_alpha",
            "jump_threshold",
            "min_dwell",
            "regime_change_count",
            "avg_dwell_bars",
            "avg_transition_risk",
            "avg_confidence",
            "avg_stability",
        }
        for mode in STRATEGY_MODES:
            expected_cols.add(f"pct_{mode}")

        assert expected_cols.issubset(set(loaded.columns))
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# 4. Tiny synthetic DataFrame (5 rows) does not crash the pipeline
# ---------------------------------------------------------------------------


class TestTinySyntheticPipeline:
    def test_stabilizer_pipeline_no_crash(self) -> None:
        """Run StateStabilizer -> MetaCalculator on a tiny synthetic DF."""
        from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
        from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator

        n = 5
        df = pd.DataFrame(
            {
                "ohio_factor_trend": np.linspace(-0.5, 0.5, n),
                "ohio_factor_volatility": np.linspace(0.2, 0.8, n),
                "ohio_factor_downside": np.linspace(0.1, 0.6, n),
                "ohio_factor_liquidity": np.linspace(0.3, 0.7, n),
                "ohio_factor_relative_strength": np.linspace(0.4, 0.6, n),
                "ohio_factor_correlation": [0.5] * n,
                "ohio_factor_breadth": [0.5] * n,
            }
        )

        stabilized = StateStabilizer(
            ema_alpha=0.10,
            jump_threshold=0.15,
            min_dwell=3,
        ).stabilize(df)

        # Verify ohio_stable_* columns exist
        stable_cols = [c for c in stabilized.columns if c.startswith("ohio_stable_")]
        assert len(stable_cols) == 7

        # MetaCalculator should not crash
        meta_df = MetaCalculator().compute(stabilized)
        assert "ohio_meta_stability" in meta_df.columns
        assert len(meta_df) == n

    def test_extract_metrics_on_tiny_df(self) -> None:
        """extract_metrics should work on a 5-row mock DataFrame."""
        df = _make_mock_df(5)
        metrics = extract_metrics(df)
        assert isinstance(metrics["regime_change_count"], int)


# ---------------------------------------------------------------------------
# 5. Argparse defaults
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_default_pair(self) -> None:
        args = parse_args([])
        assert args.pair == "BTC_USDT"

    def test_default_output_path(self) -> None:
        args = parse_args([])
        assert "stabilizer_ab_results.csv" in args.output

    def test_custom_pair(self) -> None:
        args = parse_args(["--pair", "ETH_USDT"])
        assert args.pair == "ETH_USDT"

    def test_custom_output(self) -> None:
        args = parse_args(["--output", "/tmp/custom.csv"])
        assert args.output == "/tmp/custom.csv"

    def test_output_in_backtest_results_dir(self) -> None:
        args = parse_args([])
        assert "backtest_results" in args.output
