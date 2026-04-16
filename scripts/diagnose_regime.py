"""Diagnostic script: understand WHY defensive dominates regime detection.

Runs the full OHIO pipeline on BTC 1h data, then prints detailed diagnostics
for every stage: factor distributions, profile distances, fitness scores,
hedge weights, and pure-fitness vs hedged mode selection.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------------

from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder
from freqtrade.ohio.core.market_state.normalizer import Normalizer
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator
from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
from freqtrade.ohio.core.domain.models import StrategyMode


def load_data() -> pd.DataFrame:
    """Load BTC 1h futures data."""
    data_dir = Path("user_data/data/binance/futures")
    path = data_dir / "BTC_USDT_USDT-1h-futures.feather"
    if not path.exists():
        raise FileNotFoundError(f"BTC data not found at {path}")
    df = pd.read_feather(str(path))
    print(f"Loaded BTC/USDT 1h: {len(df)} bars")
    if "date" in df.columns:
        print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full OHIO pipeline."""
    fb = FeatureBuilder()
    norm = Normalizer(window=4320)
    fc = FactorCalculator()
    stab = StateStabilizer()
    meta = MetaCalculator()
    fit = FitnessEstimator(hedge_eta=0.20, hedge_temperature=1.8)

    df = fb.compute(df)
    print("  [1/6] FeatureBuilder done")
    df = norm.normalize(df)
    print("  [2/6] Normalizer done")
    df = fc.compute(df)
    print("  [3/6] FactorCalculator done")
    df = stab.stabilize(df)
    print("  [4/6] Stabilizer done")
    df = meta.compute(df)
    print("  [5/6] MetaCalculator done")
    df = fit.compute_dataframe(df)
    print("  [6/6] FitnessEstimator done")
    return df


def print_separator(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print("=" * 80)


def percentile_stats(series: pd.Series, name: str) -> None:
    """Print mean, std, and percentiles for a series."""
    s = series.dropna()
    if len(s) == 0:
        print(f"  {name:>30}: ALL NaN")
        return
    pcts = np.percentile(s, [10, 25, 50, 75, 90])
    print(
        f"  {name:>30}: mean={s.mean():.4f}  std={s.std():.4f}  "
        f"p10={pcts[0]:.4f}  p25={pcts[1]:.4f}  p50={pcts[2]:.4f}  "
        f"p75={pcts[3]:.4f}  p90={pcts[4]:.4f}"
    )


def diagnose_factor_distributions(df: pd.DataFrame) -> None:
    """Print distribution of stabilized factor values."""
    print_separator("SECTION 1: STABILIZED FACTOR DISTRIBUTIONS")

    stable_cols = [
        ("ohio_stable_trend", "trend_persistence"),
        ("ohio_stable_volatility", "volatility_level"),
        ("ohio_stable_downside", "downside_pressure"),
        ("ohio_stable_liquidity", "liquidity_stress"),
        ("ohio_stable_relative_strength", "relative_strength"),
        ("ohio_stable_correlation", "correlation_stress"),
        ("ohio_stable_breadth", "breadth_dispersion"),
    ]

    for col, axis in stable_cols:
        if col in df.columns:
            percentile_stats(df[col], col)
        else:
            print(f"  {col:>30}: MISSING")

    # Also show raw factor distributions for comparison
    print("\n  --- Raw Factor Distributions (before stabilization) ---")
    for col, axis in stable_cols:
        factor_col = col.replace("ohio_stable_", "ohio_factor_")
        if factor_col in df.columns:
            percentile_stats(df[factor_col], factor_col)


def diagnose_profile_distances(df: pd.DataFrame) -> None:
    """Compare actual factor medians to each profile's ideal values."""
    print_separator("SECTION 2: DISTANCE FROM IDEAL VALUES (median vs ideal)")

    profiles = load_default_profiles()

    stable_to_axis = {
        "ohio_stable_trend": "trend_persistence",
        "ohio_stable_volatility": "volatility_level",
        "ohio_stable_downside": "downside_pressure",
        "ohio_stable_liquidity": "liquidity_stress",
        "ohio_stable_relative_strength": "relative_strength",
        "ohio_stable_correlation": "correlation_stress",
        "ohio_stable_breadth": "breadth_dispersion",
    }

    # Get medians of stabilized values (skip warmup)
    analysis_df = df.iloc[4320:].copy()
    medians = {}
    for col, axis in stable_to_axis.items():
        if col in analysis_df.columns:
            medians[axis] = analysis_df[col].dropna().median()
        else:
            medians[axis] = float("nan")

    print("\n  Actual medians (post-warmup):")
    for axis, med in medians.items():
        print(f"    {axis:>25}: {med:.4f}")

    mode_order = [
        StrategyMode.TREND_FOLLOWING,
        StrategyMode.MEAN_REVERSION,
        StrategyMode.BREAKOUT,
        StrategyMode.DEFENSIVE,
    ]

    for mode in mode_order:
        profile = profiles[mode]
        print(f"\n  --- {mode.value} ---")
        total_weighted_dist = 0.0
        total_weight = 0.0
        for axis_name, pref in profile.preferences.items():
            actual = medians.get(axis_name, float("nan"))
            dist = abs(actual - pref.ideal)
            scale = 2.0 if axis_name == "trend_persistence" else 1.0
            normalized_dist = dist / scale
            proximity = (1.0 - normalized_dist) ** 2
            weighted_contrib = pref.weight * proximity
            total_weighted_dist += weighted_contrib
            total_weight += pref.weight
            print(
                f"    {axis_name:>25}: actual={actual:+.4f}  ideal={pref.ideal:+.4f}  "
                f"|diff|={dist:.4f}  prox^2={proximity:.4f}  "
                f"w={pref.weight:.2f}  contrib={weighted_contrib:.4f}"
            )
        print(f"    {'TOTAL axis_score':>25}: {total_weighted_dist:.4f}")
        meta_bonus_est = (
            analysis_df["ohio_meta_confidence"].dropna().median() * profile.meta_confidence_weight
            + analysis_df["ohio_meta_stability"].dropna().median() * profile.meta_stability_weight
        )
        print(f"    {'meta_bonus (est)':>25}: {meta_bonus_est:.4f}")
        print(f"    {'TOTAL fitness (est)':>25}: {min(total_weighted_dist + meta_bonus_est, 1.0):.4f}")


def diagnose_fitness_scores(df: pd.DataFrame) -> None:
    """Print fitness score distributions for all modes."""
    print_separator("SECTION 3: FITNESS SCORE DISTRIBUTIONS (all bars, post-warmup)")

    analysis_df = df.iloc[4320:].copy()

    mode_cols = [
        "ohio_fitness_trend_following",
        "ohio_fitness_mean_reversion",
        "ohio_fitness_breakout",
        "ohio_fitness_defensive",
    ]

    for col in mode_cols:
        if col in analysis_df.columns:
            percentile_stats(analysis_df[col], col)

    # Which mode wins by pure fitness (argmax of raw fitness)?
    print("\n  --- Pure Fitness Argmax (no hedge weights, no temperature) ---")
    fitness_matrix = analysis_df[mode_cols].to_numpy()
    pure_argmax = np.argmax(fitness_matrix, axis=1)
    mode_names = ["trend_following", "mean_reversion", "breakout", "defensive"]
    for i, name in enumerate(mode_names):
        count = (pure_argmax == i).sum()
        pct = count / len(pure_argmax) if len(pure_argmax) > 0 else 0
        print(f"    {name:>20}: {pct:6.1%}  ({count} bars)")

    # Mean fitness per mode across ALL bars
    print("\n  --- Mean Fitness Across All Bars ---")
    for col in mode_cols:
        mode_name = col.replace("ohio_fitness_", "")
        mean_val = analysis_df[col].dropna().mean()
        print(f"    {mode_name:>20}: {mean_val:.4f}")

    # Fitness gaps: how often is each mode within X% of the top?
    print("\n  --- Fitness Gap Analysis (how close each mode is to the winner) ---")
    max_fitness = fitness_matrix.max(axis=1, keepdims=True)
    gaps = max_fitness - fitness_matrix
    for i, name in enumerate(mode_names):
        gap_col = gaps[:, i]
        valid = ~np.isnan(gap_col)
        if valid.sum() > 0:
            mean_gap = np.nanmean(gap_col)
            within_005 = (gap_col[valid] < 0.05).sum() / valid.sum()
            within_01 = (gap_col[valid] < 0.10).sum() / valid.sum()
            print(
                f"    {name:>20}: mean_gap={mean_gap:.4f}  "
                f"within_0.05={within_005:6.1%}  within_0.10={within_01:6.1%}"
            )


def diagnose_hedge_weights(df: pd.DataFrame) -> None:
    """Print hedge weight distributions."""
    print_separator("SECTION 4: HEDGE WEIGHT DISTRIBUTIONS (post-warmup)")

    analysis_df = df.iloc[4320:].copy()

    hedge_cols = [
        "ohio_hedge_weight_trend_following",
        "ohio_hedge_weight_mean_reversion",
        "ohio_hedge_weight_breakout",
        "ohio_hedge_weight_defensive",
    ]

    for col in hedge_cols:
        if col in analysis_df.columns:
            percentile_stats(analysis_df[col], col)
        else:
            print(f"  {col}: MISSING")


def diagnose_temperature_effect(df: pd.DataFrame) -> None:
    """Show the effect of temperature flattening on fitness scores."""
    print_separator("SECTION 5: TEMPERATURE FLATTENING EFFECT (temp=1.8)")

    analysis_df = df.iloc[4320:].copy()

    mode_cols = [
        "ohio_fitness_trend_following",
        "ohio_fitness_mean_reversion",
        "ohio_fitness_breakout",
        "ohio_fitness_defensive",
    ]
    hedge_cols = [
        "ohio_hedge_weight_trend_following",
        "ohio_hedge_weight_mean_reversion",
        "ohio_hedge_weight_breakout",
        "ohio_hedge_weight_defensive",
    ]

    fitness_matrix = analysis_df[mode_cols].to_numpy()
    hedge_matrix = analysis_df[hedge_cols].to_numpy() if all(c in analysis_df.columns for c in hedge_cols) else None

    temp = 1.8
    flattened = np.power(np.maximum(fitness_matrix, 1e-8), 1.0 / temp)

    mode_names = ["trend_following", "mean_reversion", "breakout", "defensive"]

    print("\n  --- Raw fitness means ---")
    for i, name in enumerate(mode_names):
        print(f"    {name:>20}: {np.nanmean(fitness_matrix[:, i]):.4f}")

    print("\n  --- After temperature flattening (temp=1.8) ---")
    for i, name in enumerate(mode_names):
        print(f"    {name:>20}: {np.nanmean(flattened[:, i]):.4f}")

    if hedge_matrix is not None:
        adjusted = flattened * hedge_matrix
        print("\n  --- After hedge weight adjustment ---")
        for i, name in enumerate(mode_names):
            print(f"    {name:>20}: {np.nanmean(adjusted[:, i]):.4f}")

        # Argmax of adjusted = final mode selection
        final_argmax = np.argmax(adjusted, axis=1)
        print("\n  --- Final mode selection (adjusted argmax, pre-confidence gate) ---")
        for i, name in enumerate(mode_names):
            count = (final_argmax == i).sum()
            pct = count / len(final_argmax) if len(final_argmax) > 0 else 0
            print(f"    {name:>20}: {pct:6.1%}  ({count} bars)")


def diagnose_without_hedge(df: pd.DataFrame) -> None:
    """What would mode selection be with uniform hedge weights?"""
    print_separator("SECTION 6: MODE SELECTION WITHOUT HEDGE (uniform weights)")

    analysis_df = df.iloc[4320:].copy()

    mode_cols = [
        "ohio_fitness_trend_following",
        "ohio_fitness_mean_reversion",
        "ohio_fitness_breakout",
        "ohio_fitness_defensive",
    ]

    fitness_matrix = analysis_df[mode_cols].to_numpy()
    mode_names = ["trend_following", "mean_reversion", "breakout", "defensive"]

    # With temperature = 1.0 (no flattening) and uniform weights
    print("\n  --- Pure fitness argmax (temp=1.0, uniform weights) ---")
    pure_argmax = np.argmax(fitness_matrix, axis=1)
    for i, name in enumerate(mode_names):
        count = (pure_argmax == i).sum()
        pct = count / len(pure_argmax) if len(pure_argmax) > 0 else 0
        print(f"    {name:>20}: {pct:6.1%}  ({count} bars)")

    # With temperature = 1.8 but uniform weights
    print("\n  --- Flattened fitness argmax (temp=1.8, uniform weights) ---")
    flattened = np.power(np.maximum(fitness_matrix, 1e-8), 1.0 / 1.8)
    flat_argmax = np.argmax(flattened, axis=1)
    for i, name in enumerate(mode_names):
        count = (flat_argmax == i).sum()
        pct = count / len(flat_argmax) if len(flat_argmax) > 0 else 0
        print(f"    {name:>20}: {pct:6.1%}  ({count} bars)")


def diagnose_confidence_gate(df: pd.DataFrame) -> None:
    """Show the effect of the 15% confidence gate on transitions."""
    print_separator("SECTION 7: CONFIDENCE GATE ANALYSIS")

    analysis_df = df.iloc[4320:].copy()

    if "ohio_active_mode" not in analysis_df.columns:
        print("  ohio_active_mode not available")
        return

    mode_dist = analysis_df["ohio_active_mode"].value_counts(normalize=True)
    print("\n  --- Final mode distribution (with confidence gate) ---")
    for mode, pct in sorted(mode_dist.items(), key=lambda x: -x[1]):
        print(f"    {mode:>20}: {pct:6.1%}")

    # Transition count
    transitions = (analysis_df["ohio_active_mode"] != analysis_df["ohio_active_mode"].shift(1)).sum()
    print(f"\n  Transitions: {transitions} ({transitions/len(analysis_df)*24:.1f}/day)")

    # Dwell time per mode
    print("\n  --- Average dwell time per mode (bars) ---")
    modes = analysis_df["ohio_active_mode"].values
    current_mode = modes[0]
    dwell = 1
    dwells = {m: [] for m in ["trend_following", "mean_reversion", "breakout", "defensive"]}
    for i in range(1, len(modes)):
        if modes[i] == current_mode:
            dwell += 1
        else:
            if current_mode in dwells:
                dwells[current_mode].append(dwell)
            current_mode = modes[i]
            dwell = 1
    if current_mode in dwells:
        dwells[current_mode].append(dwell)

    for mode, dwell_list in dwells.items():
        if dwell_list:
            arr = np.array(dwell_list)
            print(
                f"    {mode:>20}: mean={arr.mean():.1f}  "
                f"median={np.median(arr):.1f}  "
                f"min={arr.min()}  max={arr.max()}  "
                f"count={len(arr)}"
            )
        else:
            print(f"    {mode:>20}: never selected")


def diagnose_normalized_features(df: pd.DataFrame) -> None:
    """Show the distribution of normalized features feeding into factors."""
    print_separator("SECTION 8: KEY NORMALIZED FEATURE DISTRIBUTIONS (post-warmup)")

    analysis_df = df.iloc[4320:].copy()

    key_norm_cols = [
        # Trend inputs
        "ohio_norm_log_return_24",
        "ohio_norm_ma_slope_20",
        "ohio_norm_adx_14",
        "ohio_norm_efficiency_ratio_24",
        "ohio_norm_hurst_168",
        # Volatility inputs
        "ohio_norm_realized_vol_24",
        "ohio_norm_atr_ratio_14",
        "ohio_norm_parkinson_vol_24",
        # Downside inputs
        "ohio_norm_sortino_downside_24",
        "ohio_norm_max_drawdown_24",
        "ohio_norm_cvar_24",
        "ohio_norm_lower_shadow_ratio_24",
        "ohio_norm_negative_return_ratio_24",
        # Liquidity inputs
        "ohio_norm_bid_ask_approx",
        "ohio_norm_volume_ratio_24",
        "ohio_norm_obv_slope_24",
    ]

    for col in key_norm_cols:
        if col in analysis_df.columns:
            percentile_stats(analysis_df[col], col)
        else:
            print(f"  {col:>40}: MISSING")


def diagnose_meta_values(df: pd.DataFrame) -> None:
    """Show meta value distributions."""
    print_separator("SECTION 9: META VALUE DISTRIBUTIONS (post-warmup)")

    analysis_df = df.iloc[4320:].copy()

    meta_cols = [
        "ohio_meta_transition_risk",
        "ohio_meta_confidence",
        "ohio_meta_stability",
    ]

    for col in meta_cols:
        if col in analysis_df.columns:
            percentile_stats(analysis_df[col], col)


def diagnose_defensive_advantage(df: pd.DataFrame) -> None:
    """Deep dive: why does defensive win so often?"""
    print_separator("SECTION 10: DEFENSIVE DOMINANCE ROOT CAUSE ANALYSIS")

    analysis_df = df.iloc[4320:].copy()
    profiles = load_default_profiles()

    mode_cols = {
        "trend_following": "ohio_fitness_trend_following",
        "mean_reversion": "ohio_fitness_mean_reversion",
        "breakout": "ohio_fitness_breakout",
        "defensive": "ohio_fitness_defensive",
    }

    # For each bar, compute per-axis contributions for defensive vs the runner-up
    mode_order = [
        StrategyMode.TREND_FOLLOWING,
        StrategyMode.MEAN_REVERSION,
        StrategyMode.BREAKOUT,
        StrategyMode.DEFENSIVE,
    ]

    stable_to_axis = {
        "ohio_stable_trend": "trend_persistence",
        "ohio_stable_volatility": "volatility_level",
        "ohio_stable_downside": "downside_pressure",
        "ohio_stable_liquidity": "liquidity_stress",
        "ohio_stable_relative_strength": "relative_strength",
        "ohio_stable_correlation": "correlation_stress",
        "ohio_stable_breadth": "breadth_dispersion",
    }

    # Count: when defensive wins by pure fitness, what's the breakdown?
    fitness_matrix = analysis_df[[mode_cols[m.value] for m in mode_order]].to_numpy()
    pure_argmax = np.argmax(fitness_matrix, axis=1)
    def_wins_mask = pure_argmax == 3  # defensive is index 3

    n_def_wins = def_wins_mask.sum()
    print(f"\n  Defensive wins by pure fitness: {n_def_wins}/{len(pure_argmax)} ({n_def_wins/len(pure_argmax):.1%})")

    if n_def_wins == 0:
        print("  Defensive never wins by pure fitness -- problem is entirely in hedge/temperature")
        return

    # When defensive wins, what's the runner-up?
    def_win_scores = fitness_matrix[def_wins_mask]
    def_win_scores_copy = def_win_scores.copy()
    def_win_scores_copy[:, 3] = -1  # mask out defensive
    runner_up = np.argmax(def_win_scores_copy, axis=1)
    mode_names = ["trend_following", "mean_reversion", "breakout", "defensive"]
    print("\n  Runner-up when defensive wins:")
    for i, name in enumerate(mode_names[:3]):
        count = (runner_up == i).sum()
        print(f"    {name:>20}: {count} ({count/n_def_wins:.1%})")

    # Average per-axis contribution for each mode (across all bars)
    print("\n  --- Average per-axis fitness contribution (all bars) ---")
    print(f"  {'axis':>25} | {'TF':>8} | {'MR':>8} | {'BO':>8} | {'DEF':>8} | {'actual_median':>14}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*14}")

    for col, axis_name in stable_to_axis.items():
        if col not in analysis_df.columns:
            continue
        arr = analysis_df[col].to_numpy()
        scale = 2.0 if axis_name == "trend_persistence" else 1.0
        contribs = []
        for mode in mode_order:
            profile = profiles[mode]
            pref = profile.preferences[axis_name]
            proximity = (1.0 - np.abs(arr - pref.ideal) / scale) ** 2
            proximity = np.nan_to_num(proximity, nan=0.0)
            contrib = pref.weight * np.maximum(0.0, proximity)
            contribs.append(np.nanmean(contrib))
        actual_med = np.nanmedian(arr)
        print(
            f"  {axis_name:>25} | {contribs[0]:8.4f} | {contribs[1]:8.4f} | "
            f"{contribs[2]:8.4f} | {contribs[3]:8.4f} | {actual_med:>14.4f}"
        )

    # Meta bonus comparison
    print(f"\n  {'meta_bonus':>25} | ", end="")
    conf_med = analysis_df["ohio_meta_confidence"].dropna().median()
    stab_med = analysis_df["ohio_meta_stability"].dropna().median()
    for mode in mode_order:
        profile = profiles[mode]
        bonus = conf_med * profile.meta_confidence_weight + stab_med * profile.meta_stability_weight
        print(f"{bonus:8.4f} | ", end="")
    print()


def main() -> None:
    print("=" * 80)
    print("  OHIO REGIME DETECTION DIAGNOSTIC")
    print("  Goal: Understand why defensive=43%, breakout=30%, MR=17%, TF=10%")
    print("=" * 80)

    df = load_data()
    print("\nRunning full pipeline...")
    df = run_pipeline(df)

    # Run all diagnostic sections
    diagnose_normalized_features(df)
    diagnose_factor_distributions(df)
    diagnose_meta_values(df)
    diagnose_profile_distances(df)
    diagnose_fitness_scores(df)
    diagnose_hedge_weights(df)
    diagnose_temperature_effect(df)
    diagnose_without_hedge(df)
    diagnose_confidence_gate(df)
    diagnose_defensive_advantage(df)

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
