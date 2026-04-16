"""Check regime distribution across different market periods.

Runs the OHIO pipeline (Feature → Normalize → Factor → Stabilize → Meta → Fitness)
on each pair and reports:
1. Mode distribution (% of time in each mode)
2. Mode distribution per market period (bull/bear/sideways)
3. Mode transition frequency
4. Fitness score distribution per mode
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from pathlib import Path


def load_pair_data(pair_file: str) -> pd.DataFrame:
    """Load a single pair's futures data."""
    return pd.read_feather(pair_file)


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full OHIO pipeline on a dataframe."""
    from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder
    from freqtrade.ohio.core.market_state.normalizer import Normalizer
    from freqtrade.ohio.core.market_state.factors import FactorCalculator
    from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
    from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
    from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator

    fb = FeatureBuilder()
    norm = Normalizer(window=4320)
    fc = FactorCalculator()
    stab = StateStabilizer()
    meta = MetaCalculator()
    fit = FitnessEstimator(hedge_eta=0.20, hedge_temperature=1.8)

    df = fb.compute(df)
    df = norm.normalize(df)
    df = fc.compute(df)
    df = stab.stabilize(df)
    df = meta.compute(df)
    df = fit.compute_dataframe(df)

    return df


def classify_market_period(df: pd.DataFrame, window: int = 720) -> pd.Series:
    """Classify each bar into bull/bear/sideways based on rolling return."""
    rolling_ret = df["close"].pct_change(window).fillna(0)
    conditions = [
        rolling_ret > 0.10,   # >10% over 30 days = bull
        rolling_ret < -0.10,  # <-10% over 30 days = bear
    ]
    choices = ["bull", "bear"]
    return pd.Series(
        np.select(conditions, choices, default="sideways"),
        index=df.index,
    )


def analyze_regime(df: pd.DataFrame, pair_name: str) -> dict:
    """Analyze regime distribution for a single pair."""
    if "ohio_active_mode" not in df.columns:
        return {}

    # Skip warmup period (first 4320 bars)
    df = df.iloc[4320:].copy()
    if len(df) == 0:
        return {}

    # Market period classification
    df["market_period"] = classify_market_period(df)

    # Overall mode distribution
    mode_dist = df["ohio_active_mode"].value_counts(normalize=True)

    # Mode distribution per market period
    mode_by_period = {}
    for period in ["bull", "bear", "sideways"]:
        mask = df["market_period"] == period
        if mask.sum() > 0:
            mode_by_period[period] = df.loc[mask, "ohio_active_mode"].value_counts(normalize=True)

    # Mode transition frequency
    transitions = (df["ohio_active_mode"] != df["ohio_active_mode"].shift(1)).sum()
    transition_rate = transitions / len(df) if len(df) > 0 else 0

    # Fitness distribution per mode
    fitness_by_mode = {}
    for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
        col = f"ohio_fitness_{mode}"
        if col in df.columns:
            mask = df["ohio_active_mode"] == mode
            if mask.sum() > 0:
                fitness_by_mode[mode] = {
                    "mean": df.loc[mask, col].mean(),
                    "std": df.loc[mask, col].std(),
                    "min": df.loc[mask, col].min(),
                    "max": df.loc[mask, col].max(),
                }

    # Market period distribution
    period_dist = df["market_period"].value_counts(normalize=True)

    return {
        "pair": pair_name,
        "total_bars": len(df),
        "mode_distribution": mode_dist.to_dict(),
        "mode_by_period": {k: v.to_dict() for k, v in mode_by_period.items()},
        "transition_rate": transition_rate,
        "transitions_per_day": transition_rate * 24,
        "fitness_by_mode": fitness_by_mode,
        "period_distribution": period_dist.to_dict(),
    }


def main():
    data_dir = Path("user_data/data/binance/futures")
    pairs = [
        "BTC_USDT_USDT-1h-futures.feather",
        "ETH_USDT_USDT-1h-futures.feather",
        "DOGE_USDT_USDT-1h-futures.feather",
        "DOT_USDT_USDT-1h-futures.feather",
        "ADA_USDT_USDT-1h-futures.feather",
        "1000PEPE_USDT_USDT-1h-futures.feather",
    ]

    print("=" * 80)
    print("OHIO REGIME DISTRIBUTION CHECK")
    print("=" * 80)

    all_results = []
    for pair_file in pairs:
        path = data_dir / pair_file
        if not path.exists():
            print(f"\n[SKIP] {pair_file} not found")
            continue

        pair_name = pair_file.split("-")[0].replace("_USDT_USDT", "/USDT")
        print(f"\n{'='*60}")
        print(f"Processing {pair_name}...")

        df = load_pair_data(str(path))
        print(f"  Data: {df['date'].min()} ~ {df['date'].max()} ({len(df)} bars)")

        df = run_pipeline(df)
        result = analyze_regime(df, pair_name)
        if not result:
            print("  [SKIP] Pipeline failed or insufficient data")
            continue

        all_results.append(result)

        # Print results
        print(f"\n  Market Period Distribution:")
        for period, pct in result["period_distribution"].items():
            print(f"    {period:>10}: {pct:6.1%}")

        print(f"\n  Overall Mode Distribution:")
        for mode, pct in sorted(result["mode_distribution"].items(), key=lambda x: -x[1]):
            print(f"    {mode:>20}: {pct:6.1%}")

        print(f"\n  Mode per Market Period:")
        for period in ["bull", "bear", "sideways"]:
            if period in result["mode_by_period"]:
                print(f"    [{period}]")
                for mode, pct in sorted(result["mode_by_period"][period].items(), key=lambda x: -x[1]):
                    print(f"      {mode:>20}: {pct:6.1%}")

        print(f"\n  Transitions: {result['transition_rate']:.4f}/bar ({result['transitions_per_day']:.1f}/day)")

        print(f"\n  Fitness per Active Mode:")
        for mode, stats in result["fitness_by_mode"].items():
            print(f"    {mode:>20}: mean={stats['mean']:.3f} std={stats['std']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]")

    # Aggregate summary
    if all_results:
        print(f"\n{'='*80}")
        print("AGGREGATE SUMMARY")
        print("=" * 80)

        # Average mode distribution across pairs
        all_modes = {}
        for r in all_results:
            for mode, pct in r["mode_distribution"].items():
                all_modes.setdefault(mode, []).append(pct)

        print("\n  Average Mode Distribution (across all pairs):")
        for mode, pcts in sorted(all_modes.items(), key=lambda x: -np.mean(x[1])):
            print(f"    {mode:>20}: {np.mean(pcts):6.1%} (std={np.std(pcts):.1%})")

        avg_transitions = np.mean([r["transitions_per_day"] for r in all_results])
        print(f"\n  Average transitions/day: {avg_transitions:.1f}")


if __name__ == "__main__":
    main()
