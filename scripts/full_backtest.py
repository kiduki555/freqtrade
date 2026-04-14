"""Full pipeline backtest: standalone determinism + signal analysis for OHIO Market State Engine.

Loads OHLCV feather data, runs the 7-phase pipeline twice to verify determinism,
then produces a human-readable report and CSV of entry signals.

Usage::

    python scripts/full_backtest.py
    python scripts/full_backtest.py --pairs BTC_USDT,ETH_USDT --output-dir user_data/backtest_results
"""
from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder
from freqtrade.ohio.core.market_state.normalizer import Normalizer
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator
from freqtrade.ohio.core.strategy_router.policy_generator import PolicyGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WARMUP_BARS = 4320
_DATA_DIR = Path(__file__).resolve().parents[1] / "user_data" / "data" / "binance"

_STABLE_COLS = [
    "ohio_stable_trend",
    "ohio_stable_volatility",
    "ohio_stable_downside",
    "ohio_stable_liquidity",
    "ohio_stable_relative_strength",
    "ohio_stable_correlation",
    "ohio_stable_breadth",
]

_SIGNAL_EXPORT_COLS = [
    "date",
    "close",
    "ohio_active_mode",
    "ohio_policy_enabled",
    "ohio_policy_size_multiplier",
    "ohio_policy_entry_threshold_adj",
    "ohio_policy_max_positions",
    "ohio_meta_transition_risk",
    "ohio_meta_confidence",
    "ohio_meta_stability",
    "ohio_fitness_trend_following",
    "ohio_fitness_mean_reversion",
    "ohio_fitness_breakout",
    "ohio_fitness_defensive",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candles(pair: str, data_dir: Path) -> pd.DataFrame:
    """Load feather file for a given pair and validate OHLCV columns."""
    feather_path = data_dir / f"{pair}-1h.feather"
    if not feather_path.exists():
        raise FileNotFoundError(
            f"Candle data not found: {feather_path}\n"
            f"Run 'scripts/download_ohio_data.sh' first."
        )
    df = pd.read_feather(feather_path)
    expected = {"date", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns in {pair}: {missing}")
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full 7-phase OHIO pipeline on a copy of *df*."""
    result = df.copy()
    result = FeatureBuilder().compute(result)
    result = Normalizer(window=4320).normalize(result)
    result = FactorCalculator().compute(result)
    result = StateStabilizer().stabilize(result)
    result = MetaCalculator().compute(result)
    result = FitnessEstimator().compute_dataframe(result)
    result = PolicyGenerator().generate_dataframe(result)
    return result


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------

def check_determinism(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Run the pipeline twice and assert bit-exact equality.

    Returns:
        (result_df, elapsed_run1, elapsed_run2)

    Raises:
        AssertionError if the two runs produce different outputs.
    """
    t0 = time.perf_counter()
    run1 = run_pipeline(df)
    elapsed1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    run2 = run_pipeline(df)
    elapsed2 = time.perf_counter() - t0

    try:
        pd.testing.assert_frame_equal(run1, run2, check_exact=True)
    except AssertionError as exc:
        raise AssertionError(
            f"DETERMINISM FAILURE: two pipeline runs on identical input diverged.\n{exc}"
        ) from exc

    return run1, elapsed1, elapsed2


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def analyse_signals(df: pd.DataFrame, pair: str) -> dict[str, Any]:
    """Extract signal metrics from a fully-processed DataFrame."""
    total_bars = len(df)
    warmup_mask = df.index < _WARMUP_BARS
    active_mask = ~warmup_mask
    active_bars = int(active_mask.sum())

    active_df = df.loc[active_mask]

    entry_mask = active_df["ohio_policy_enabled"] == True  # noqa: E712
    entry_count = int(entry_mask.sum())
    entry_rate = entry_count / active_bars if active_bars > 0 else 0.0

    # Mode distribution among entries
    entry_df = active_df.loc[entry_mask]
    mode_dist: dict[str, float] = {}
    if len(entry_df) > 0:
        counts = entry_df["ohio_active_mode"].value_counts(normalize=True)
        mode_dist = {str(k): round(float(v) * 100, 2) for k, v in counts.items()}

    # Average fitness of entries
    fitness_cols = [
        "ohio_fitness_trend_following",
        "ohio_fitness_mean_reversion",
        "ohio_fitness_breakout",
        "ohio_fitness_defensive",
    ]
    avg_fitness = float("nan")
    if len(entry_df) > 0:
        per_row_best = entry_df[fitness_cols].max(axis=1)
        avg_fitness = float(per_row_best.mean())

    avg_transition_risk = float("nan")
    if "ohio_meta_transition_risk" in active_df.columns:
        avg_transition_risk = float(
            active_df["ohio_meta_transition_risk"].mean()
        )

    return {
        "pair": pair,
        "total_bars": total_bars,
        "warmup_bars": int(warmup_mask.sum()),
        "active_bars": active_bars,
        "entry_count": entry_count,
        "entry_rate": entry_rate,
        "mode_distribution": mode_dist,
        "avg_fitness": avg_fitness,
        "avg_transition_risk": avg_transition_risk,
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    df: pd.DataFrame,
    metrics: dict[str, Any],
) -> list[str]:
    """Return a list of warning/error strings. Empty list means all OK."""
    warnings: list[str] = []
    pair = metrics["pair"]

    # 1. Entry rate sanity check
    # Note: ohio_policy_enabled is the first gate. In crypto markets,
    # mean_reversion conditions dominate, so near-100% rate is normal.
    # Actual entry filtering happens in confirm_trade_entry (risk gate).
    rate = metrics["entry_rate"]
    if rate < 0.05:
        warnings.append(
            f"[{pair}] WARN: entry rate too low ({rate:.1%}) — signals may be too sparse"
        )
    if rate == 0.0:
        warnings.append(
            f"[{pair}] WARN: entry rate is 0% — no signals generated at all"
        )

    # 2. At least 2 different modes used
    mode_count = len(metrics["mode_distribution"])
    if mode_count < 2:
        warnings.append(
            f"[{pair}] WARN: only {mode_count} mode(s) used in entries — "
            f"pipeline may not discriminate market states"
        )

    # 3. No NaN in ohio_stable_* columns after warmup
    active_df = df.iloc[_WARMUP_BARS:]
    for col in _STABLE_COLS:
        if col in active_df.columns:
            nan_count = int(active_df[col].isna().sum())
            if nan_count > 0:
                warnings.append(
                    f"[{pair}] WARN: {col} has {nan_count} NaN values after warmup"
                )

    # 4. ohio_meta_confidence > 0 for most post-warmup bars
    if "ohio_meta_confidence" in active_df.columns:
        conf = active_df["ohio_meta_confidence"]
        positive_ratio = float((conf > 0).mean())
        if positive_ratio < 0.80:
            warnings.append(
                f"[{pair}] WARN: ohio_meta_confidence > 0 only {positive_ratio:.1%} "
                f"of post-warmup bars (expected > 80%)"
            )

    return warnings


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    pair_results: list[tuple[dict[str, Any], list[str], float, float]],
) -> str:
    """Build a human-readable report string."""
    buf = StringIO()
    buf.write("=" * 72 + "\n")
    buf.write("  OHIO Market State Engine — Full Pipeline Backtest Report\n")
    buf.write("=" * 72 + "\n\n")

    all_warnings: list[str] = []

    for metrics, warnings, elapsed1, elapsed2 in pair_results:
        pair = metrics["pair"]
        buf.write(f"--- {pair} ---\n\n")
        buf.write(f"  Total bars:        {metrics['total_bars']:,}\n")
        buf.write(f"  Warmup bars:       {metrics['warmup_bars']:,}\n")
        buf.write(f"  Active bars:       {metrics['active_bars']:,}\n")
        buf.write(f"  Entry signals:     {metrics['entry_count']:,}\n")
        buf.write(f"  Entry rate:        {metrics['entry_rate']:.2%}\n")
        buf.write(f"  Avg fitness:       {metrics['avg_fitness']:.4f}\n")
        buf.write(f"  Avg trans. risk:   {metrics['avg_transition_risk']:.4f}\n")
        buf.write(f"  Pipeline run 1:    {elapsed1:.2f}s\n")
        buf.write(f"  Pipeline run 2:    {elapsed2:.2f}s\n")
        buf.write(f"  Determinism:       PASS\n\n")

        if metrics["mode_distribution"]:
            buf.write("  Mode distribution (entries):\n")
            for mode, pct in sorted(
                metrics["mode_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                buf.write(f"    {mode:<25s} {pct:6.2f}%\n")
            buf.write("\n")

        if warnings:
            buf.write("  Sanity warnings:\n")
            for w in warnings:
                buf.write(f"    {w}\n")
            buf.write("\n")
            all_warnings.extend(warnings)

    buf.write("-" * 72 + "\n")
    if all_warnings:
        buf.write(f"  Total warnings: {len(all_warnings)}\n")
    else:
        buf.write("  All sanity checks PASSED.\n")
    buf.write("=" * 72 + "\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OHIO full pipeline backtest — determinism + signal analysis",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="BTC_USDT,ETH_USDT",
        help="Comma-separated trading pairs (default: BTC_USDT,ETH_USDT)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "user_data" / "backtest_results"
        ),
        help="Directory for report and CSV output",
    )
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_results: list[tuple[dict[str, Any], list[str], float, float]] = []
    all_entry_dfs: list[pd.DataFrame] = []

    for pair in pairs:
        print(f"\n[{pair}] Loading candle data...")
        raw = load_candles(pair, _DATA_DIR)
        print(f"[{pair}] Loaded {len(raw):,} bars")

        print(f"[{pair}] Running determinism check (2 pipeline runs)...")
        result_df, elapsed1, elapsed2 = check_determinism(raw)
        print(f"[{pair}] Determinism PASS  (run1={elapsed1:.2f}s, run2={elapsed2:.2f}s)")

        metrics = analyse_signals(result_df, pair)
        warnings = run_sanity_checks(result_df, metrics)

        pair_results.append((metrics, warnings, elapsed1, elapsed2))

        # Collect entry signal rows for CSV export
        active_df = result_df.iloc[_WARMUP_BARS:]
        entry_mask = active_df["ohio_policy_enabled"] == True  # noqa: E712
        entry_rows = active_df.loc[entry_mask].copy()
        export_cols = [c for c in _SIGNAL_EXPORT_COLS if c in entry_rows.columns]
        entry_rows = entry_rows[export_cols]
        entry_rows.insert(0, "pair", pair)
        all_entry_dfs.append(entry_rows)

    # Build and print report
    report = format_report(pair_results)
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + report)

    # Write report file
    report_path = output_dir / "full_pipeline_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {report_path}")

    # Write CSV
    if all_entry_dfs:
        csv_df = pd.concat(all_entry_dfs, ignore_index=True)
        csv_path = output_dir / "pipeline_signals.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"Signals CSV written to: {csv_path}  ({len(csv_df):,} rows)")


if __name__ == "__main__":
    main()
