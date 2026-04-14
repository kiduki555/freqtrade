"""Regime Analysis: K-means clustering on OHIO 7-axis stabilized state vectors.

Loads candle data, runs the full OHIO pipeline, clusters the stabilized axes
via K-means (auto-selecting k by silhouette score), and outputs per-regime
performance metrics, a transition probability matrix, and regime-labeled data.

Usage::

    python scripts/regime_analysis.py --pair BTC_USDT
    python scripts/regime_analysis.py --pair ETH_USDT --k 4
    python scripts/regime_analysis.py --pair BTC_USDT --output-dir user_data/custom_output
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print(
        "ERROR: scikit-learn is required for regime analysis.\n"
        "Install it with: pip install scikit-learn"
    )
    sys.exit(1)

from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder
from freqtrade.ohio.core.market_state.normalizer import Normalizer
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator
from freqtrade.ohio.core.strategy_router.policy_generator import PolicyGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

STABLE_AXES: list[str] = [
    "ohio_stable_trend",
    "ohio_stable_volatility",
    "ohio_stable_downside",
    "ohio_stable_liquidity",
    "ohio_stable_relative_strength",
    "ohio_stable_correlation",
    "ohio_stable_breadth",
]

FITNESS_COLUMNS: list[str] = [
    "ohio_fitness_trend_following",
    "ohio_fitness_mean_reversion",
    "ohio_fitness_breakout",
    "ohio_fitness_defensive",
]

K_CANDIDATES: list[int] = [3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_candle_data(pair: str, data_dir: Path) -> pd.DataFrame:
    """Load a feather file for the given pair."""
    feather_path = data_dir / f"{pair}-1h.feather"
    if not feather_path.exists():
        raise FileNotFoundError(f"Candle data not found: {feather_path}")
    df = pd.read_feather(feather_path)
    logger.info("Loaded %d rows from %s", len(df), feather_path)
    return df


def run_ohio_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full OHIO pipeline (steps 1-7) on a DataFrame of candles."""
    df = df.copy()
    df = FeatureBuilder().compute(df)
    df = Normalizer(window=4320).normalize(df)
    df = FactorCalculator().compute(df)
    df = StateStabilizer().stabilize(df)
    df = MetaCalculator().compute(df)
    df = FitnessEstimator().compute_dataframe(df)
    df = PolicyGenerator().generate_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def select_best_k(
    features: np.ndarray,
    candidates: list[int] | None = None,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Try multiple k values and return the one with highest silhouette score.

    Returns:
        (best_k, scores_dict)  where scores_dict maps k → silhouette_score.
    """
    if candidates is None:
        candidates = K_CANDIDATES

    scores: dict[int, float] = {}
    for k in candidates:
        if k >= len(features):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(features)
        score = silhouette_score(features, labels)
        scores[k] = score
        logger.info("k=%d  silhouette=%.4f", k, score)

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores


def cluster_regimes(
    df: pd.DataFrame,
    k_override: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, int, dict[int, float]]:
    """Cluster rows on the 7 stabilized axes. Returns (df_with_regime, k, scores).

    Rows with NaN in any stable axis are excluded from clustering and labelled -1.
    """
    mask = df[STABLE_AXES].notna().all(axis=1)
    valid_idx = df.index[mask]

    df = df.copy()
    df["regime"] = -1

    if len(valid_idx) == 0:
        best_k = k_override if k_override is not None else 0
        logger.warning("No valid rows for clustering — all regimes set to -1")
        return df, best_k, {}

    raw_features = df.loc[valid_idx, STABLE_AXES].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw_features)

    if k_override is not None:
        best_k = k_override
        scores: dict[int, float] = {}
    else:
        best_k, scores = select_best_k(scaled, random_state=random_state)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = km.fit_predict(scaled)

    df.loc[valid_idx, "regime"] = labels

    logger.info("Assigned %d rows to %d regimes (%d NaN-excluded)", len(valid_idx), best_k, len(df) - len(valid_idx))
    return df, best_k, scores


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------


def compute_regime_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime summary metrics.

    Returns a DataFrame indexed by regime with columns:
        bar_count, pct_of_total, avg_return_1h, avg_volatility, avg_trend,
        dominant_mode, mode_distribution,
        avg_fitness_trend_following, avg_fitness_mean_reversion,
        avg_fitness_breakout, avg_fitness_defensive,
        avg_transition_risk
    """
    # Compute 1h return (close-to-close)
    if "close" in df.columns:
        df = df.copy()
        df["return_1h"] = df["close"].pct_change()

    clustered = df[df["regime"] >= 0]
    total_bars = len(clustered)
    records: list[dict] = []

    for regime_id, group in clustered.groupby("regime"):
        count = len(group)

        # Mode distribution
        mode_col = "ohio_active_mode"
        if mode_col in group.columns:
            mode_counts = group[mode_col].value_counts(normalize=True)
            dominant = mode_counts.index[0] if len(mode_counts) > 0 else "unknown"
            mode_dist = mode_counts.to_dict()
        else:
            dominant = "unknown"
            mode_dist = {}

        record: dict = {
            "regime": int(regime_id),
            "bar_count": count,
            "pct_of_total": round(count / total_bars * 100, 2) if total_bars > 0 else 0.0,
            "avg_return_1h": group["return_1h"].mean() if "return_1h" in group.columns else np.nan,
            "avg_volatility": group["ohio_stable_volatility"].mean() if "ohio_stable_volatility" in group.columns else np.nan,
            "avg_trend": group["ohio_stable_trend"].mean() if "ohio_stable_trend" in group.columns else np.nan,
            "dominant_mode": dominant,
            "mode_distribution": mode_dist,
            "avg_transition_risk": group["ohio_meta_transition_risk"].mean() if "ohio_meta_transition_risk" in group.columns else np.nan,
        }

        for fcol in FITNESS_COLUMNS:
            key = f"avg_{fcol.replace('ohio_', '')}"
            record[key] = group[fcol].mean() if fcol in group.columns else np.nan

        records.append(record)

    return pd.DataFrame(records).set_index("regime").sort_index()


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


def compute_transition_matrix(df: pd.DataFrame, n_regimes: int) -> pd.DataFrame:
    """Build an NxN transition probability matrix between regimes.

    Only considers consecutive rows where both have regime >= 0.
    Rows sum to ~1.0 (or 0 if a regime has no outgoing transitions).
    """
    regimes = df["regime"].values
    counts = np.zeros((n_regimes, n_regimes), dtype=np.int64)

    for i in range(len(regimes) - 1):
        src, dst = regimes[i], regimes[i + 1]
        if src >= 0 and dst >= 0:
            counts[src, dst] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    labels = [f"regime_{i}" for i in range(n_regimes)]
    return pd.DataFrame(probs, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(
    metrics_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write all CSVs to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "regime_analysis_summary.csv"
    transition_path = output_dir / "regime_transitions.csv"
    labeled_path = output_dir / "regime_labeled_data.csv"

    metrics_df.to_csv(summary_path)
    transition_df.to_csv(transition_path)

    # For labeled data, convert mode_distribution dicts in metrics are already saved;
    # the labeled_df just needs the regime column alongside OHLCV + ohio columns.
    labeled_df.to_csv(labeled_path, index=False)

    logger.info("Saved: %s", summary_path)
    logger.info("Saved: %s", transition_path)
    logger.info("Saved: %s", labeled_path)


def print_summary(
    metrics_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    best_k: int,
    scores: dict[int, float],
    pair: str,
) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  OHIO Regime Analysis - {pair}")
    print(f"{'=' * 70}")

    if scores:
        print("\nSilhouette scores:")
        for k, s in sorted(scores.items()):
            marker = " <-- best" if k == best_k else ""
            print(f"  k={k}: {s:.4f}{marker}")
    print(f"\nSelected k={best_k}")

    print(f"\n{'─' * 70}")
    print("Per-Regime Metrics:")
    print(f"{'─' * 70}")
    display_cols = [
        "bar_count", "pct_of_total", "avg_return_1h",
        "avg_volatility", "avg_trend", "dominant_mode", "avg_transition_risk",
    ]
    available = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[available].to_string())

    print(f"\n{'─' * 70}")
    print("Transition Probability Matrix:")
    print(f"{'─' * 70}")
    print(transition_df.round(3).to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OHIO Regime Analysis: cluster stabilized market states via K-means.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="BTC_USDT",
        help="Trading pair (default: BTC_USDT). Must match feather filename.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override cluster count (default: auto-select via silhouette score).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: user_data/backtest_results).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)
    pair: str = args.pair
    k_override: int | None = args.k

    data_dir = _PROJECT_ROOT / "user_data" / "data" / "binance"
    output_dir = Path(args.output_dir) if args.output_dir else _PROJECT_ROOT / "user_data" / "backtest_results"

    # Step 1: Load candle data
    print(f"Loading candle data for {pair} ...")
    df = load_candle_data(pair, data_dir)

    # Step 2: Run full OHIO pipeline
    print("Running OHIO pipeline (7 stages) ...")
    df = run_ohio_pipeline(df)

    # Step 3: K-means clustering
    print("Clustering stabilized state vectors ...")
    df, best_k, scores = cluster_regimes(df, k_override=k_override)

    # Step 4: Per-regime metrics
    print("Computing per-regime metrics ...")
    metrics_df = compute_regime_metrics(df)

    # Step 5: Transition matrix
    print("Computing transition matrix ...")
    transition_df = compute_transition_matrix(df, best_k)

    # Step 6: Output
    save_outputs(metrics_df, transition_df, df, output_dir)
    print_summary(metrics_df, transition_df, best_k, scores, pair)

    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
