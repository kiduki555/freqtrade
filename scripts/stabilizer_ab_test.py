"""A/B test for StateStabilizer parameter tuning.

Runs the full OHIO pipeline (FeatureBuilder -> Normalizer -> FactorCalculator
-> StateStabilizer -> MetaCalculator -> FitnessEstimator -> PolicyGenerator)
across a grid of stabilizer parameters and collects regime-stability metrics.

Usage::

    python scripts/stabilizer_ab_test.py
    python scripts/stabilizer_ab_test.py --pair ETH_USDT
    python scripts/stabilizer_ab_test.py --output path/to/results.csv
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure freqtrade root is importable
# ---------------------------------------------------------------------------

_FREQTRADE_ROOT = str(Path(__file__).resolve().parents[1])
if _FREQTRADE_ROOT not in sys.path:
    sys.path.insert(0, _FREQTRADE_ROOT)

from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder
from freqtrade.ohio.core.market_state.normalizer import Normalizer
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator
from freqtrade.ohio.core.strategy_router.policy_generator import PolicyGenerator

# Optional tqdm import
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

EMA_ALPHAS: list[float] = [0.03, 0.05, 0.10, 0.15, 0.20]
JUMP_THRESHOLDS: list[float] = [0.05, 0.10, 0.15, 0.20, 0.30]
MIN_DWELLS: list[int] = [3, 6, 9, 12, 18]

STRATEGY_MODES = ["trend_following", "mean_reversion", "breakout", "defensive"]


def build_param_grid() -> list[dict]:
    """Return list of parameter dicts for all grid combinations."""
    return [
        {"ema_alpha": a, "jump_threshold": j, "min_dwell": d}
        for a, j, d in itertools.product(EMA_ALPHAS, JUMP_THRESHOLDS, MIN_DWELLS)
    ]


# ---------------------------------------------------------------------------
# Pipeline: steps 1-3 are shared across all combos (computed once)
# ---------------------------------------------------------------------------


def run_shared_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run FeatureBuilder -> Normalizer -> FactorCalculator (steps 1-3).

    Returns a new DataFrame with ohio_factor_* columns ready for stabilization.
    """
    result = df.copy()
    result = FeatureBuilder().compute(result)
    result = Normalizer(window=4320).normalize(result)
    result = FactorCalculator().compute(result)
    return result


def run_stabilizer_pipeline(
    base_df: pd.DataFrame,
    ema_alpha: float,
    jump_threshold: float,
    min_dwell: int,
) -> pd.DataFrame:
    """Run steps 4-7: Stabilizer -> Meta -> Fitness -> Policy.

    Args:
        base_df: DataFrame already containing ohio_factor_* columns.
        ema_alpha: EMA smoothing factor for the stabilizer.
        jump_threshold: Minimum delta to accept a state change.
        min_dwell: Minimum bars between accepted changes.

    Returns:
        Final DataFrame with all ohio_* columns.
    """
    result = base_df.copy()
    result = StateStabilizer(ema_alpha, jump_threshold, min_dwell).stabilize(result)
    result = MetaCalculator().compute(result)
    result = FitnessEstimator().compute_dataframe(result)
    result = PolicyGenerator().generate_dataframe(result)
    return result


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_metrics(df: pd.DataFrame) -> dict:
    """Extract stability metrics from a fully-processed DataFrame.

    Returns a dict with:
        regime_change_count, avg_dwell_bars, avg_transition_risk,
        avg_confidence, avg_stability, mode_distribution (per-mode %).
    """
    metrics: dict = {}

    # --- regime change count and avg dwell ---
    mode_col = df.get("ohio_active_mode")
    if mode_col is not None and not mode_col.isna().all():
        changes = mode_col != mode_col.shift(1)
        # Exclude the first row (always counts as a "change")
        regime_change_count = int(changes.iloc[1:].sum())
        metrics["regime_change_count"] = regime_change_count

        if regime_change_count > 0:
            # Average bars between consecutive mode changes
            change_indices = changes[changes].index.tolist()
            if len(change_indices) > 1:
                diffs = [
                    change_indices[i + 1] - change_indices[i]
                    for i in range(len(change_indices) - 1)
                ]
                metrics["avg_dwell_bars"] = round(sum(diffs) / len(diffs), 2)
            else:
                metrics["avg_dwell_bars"] = float(len(df))
        else:
            metrics["avg_dwell_bars"] = float(len(df))
    else:
        metrics["regime_change_count"] = 0
        metrics["avg_dwell_bars"] = float(len(df)) if len(df) > 0 else 0.0

    # --- meta averages ---
    for col_suffix, key in [
        ("ohio_meta_transition_risk", "avg_transition_risk"),
        ("ohio_meta_confidence", "avg_confidence"),
        ("ohio_meta_stability", "avg_stability"),
    ]:
        if col_suffix in df.columns:
            metrics[key] = round(float(df[col_suffix].mean(skipna=True)), 6)
        else:
            metrics[key] = float("nan")

    # --- mode distribution ---
    if mode_col is not None and not mode_col.isna().all():
        total = mode_col.notna().sum()
        for mode in STRATEGY_MODES:
            count = (mode_col == mode).sum()
            pct = round(100.0 * count / total, 2) if total > 0 else 0.0
            metrics[f"pct_{mode}"] = pct
    else:
        for mode in STRATEGY_MODES:
            metrics[f"pct_{mode}"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A/B test for OHIO StateStabilizer parameters",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="BTC_USDT",
        help="Trading pair (default: BTC_USDT)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path(_FREQTRADE_ROOT)
            / "user_data"
            / "backtest_results"
            / "stabilizer_ab_results.csv"
        ),
        help="Output CSV path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the stabilizer A/B test."""
    args = parse_args(argv)
    pair = args.pair
    output_path = Path(args.output)

    # --- Load candle data ---
    feather_path = (
        Path(_FREQTRADE_ROOT)
        / "user_data"
        / "data"
        / "binance"
        / f"{pair}-1h.feather"
    )
    if not feather_path.exists():
        print(f"ERROR: Candle data not found at {feather_path}")
        sys.exit(1)

    print(f"Loading candle data from {feather_path} ...")
    raw_df = pd.read_feather(feather_path)
    print(f"  Loaded {len(raw_df)} candles")

    # --- Run shared pipeline (steps 1-3) once ---
    print("Running shared pipeline (FeatureBuilder -> Normalizer -> FactorCalculator) ...")
    t0 = time.time()
    base_df = run_shared_pipeline(raw_df)
    print(f"  Shared pipeline completed in {time.time() - t0:.1f}s")

    # --- Grid search ---
    grid = build_param_grid()
    print(f"Running {len(grid)} parameter combinations ...")

    results: list[dict] = []

    if tqdm is not None:
        iterator = tqdm(grid, desc="Stabilizer A/B", unit="combo")
    else:
        iterator = grid

    for i, params in enumerate(iterator):
        if tqdm is None and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(grid)}")

        try:
            final_df = run_stabilizer_pipeline(
                base_df,
                ema_alpha=params["ema_alpha"],
                jump_threshold=params["jump_threshold"],
                min_dwell=params["min_dwell"],
            )
            metrics = extract_metrics(final_df)
        except Exception as exc:
            logger.warning(
                "Failed for params %s: %s", params, exc,
            )
            metrics = {
                "regime_change_count": -1,
                "avg_dwell_bars": float("nan"),
                "avg_transition_risk": float("nan"),
                "avg_confidence": float("nan"),
                "avg_stability": float("nan"),
            }
            for mode in STRATEGY_MODES:
                metrics[f"pct_{mode}"] = float("nan")

        row = {**params, **metrics}
        results.append(row)

    # --- Save results ---
    results_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # --- Print top-10 by avg_stability ---
    print("\n" + "=" * 80)
    print("TOP 10 COMBINATIONS BY avg_stability")
    print("=" * 80)

    if "avg_stability" in results_df.columns:
        top10 = (
            results_df.dropna(subset=["avg_stability"])
            .sort_values("avg_stability", ascending=False)
            .head(10)
        )
        display_cols = [
            "ema_alpha",
            "jump_threshold",
            "min_dwell",
            "avg_stability",
            "avg_confidence",
            "avg_transition_risk",
            "regime_change_count",
            "avg_dwell_bars",
        ]
        existing_cols = [c for c in display_cols if c in top10.columns]
        print(top10[existing_cols].to_string(index=False))
    else:
        print("No avg_stability data available.")

    print()


if __name__ == "__main__":
    main()
