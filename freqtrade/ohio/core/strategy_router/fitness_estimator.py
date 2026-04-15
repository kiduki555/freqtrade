"""FT-011: Fitness Estimator for the OHIO Market State Engine.

Computes a fitness score [0, 1] for each of the 4 strategy modes by comparing
a stabilised StateVector against each mode's StrategyProfile preferences.

No Freqtrade framework imports — only domain models, strategy_profile, stdlib,
numpy, and pandas.
"""
from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

from freqtrade.ohio.core.domain.models import (
    StateMeta,
    StateVector,
    StrategyFitness,
    StrategyMode,
)
from freqtrade.ohio.core.strategy_router.strategy_profile import (
    StrategyProfile,
    load_default_profiles,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# trend_persistence spans [-1, 1] → normalisation range is 2.0; all others [0, 1] → 1.0
_TREND_AXIS: Final[str] = "trend_persistence"
_TREND_SCALE: Final[float] = 2.0
_DEFAULT_SCALE: Final[float] = 1.0

# Mapping from ohio_stable_* column suffixes to StateVector field names.
_STABLE_COL_TO_AXIS: Final[dict[str, str]] = {
    "trend": "trend_persistence",
    "volatility": "volatility_level",
    "downside": "downside_pressure",
    "liquidity": "liquidity_stress",
    "relative_strength": "relative_strength",
    "correlation": "correlation_stress",
    "breadth": "breadth_dispersion",
}

# Mapping from StateMeta fields to ohio_meta_* column suffixes.
_META_COL_TO_FIELD: Final[dict[str, str]] = {
    "transition_risk": "transition_risk",
    "confidence": "confidence",
    "stability": "stability",
}


# ---------------------------------------------------------------------------
# Reward computation for Hedge mode selection
# ---------------------------------------------------------------------------


def _compute_rewards(
    ret: np.ndarray,
    trend: np.ndarray,
    atr: np.ndarray,
    lookback: int = 168,
) -> np.ndarray:
    """Compute per-mode rewards from 1-bar return, trend, and ATR.

    TF/MR rewards: directional alignment with trend, normalised by ATR.
    BO/DEF rewards: |return| vs rolling median of |return| — centred so
    neither mode is systematically favoured (ATR is too large as threshold:
    88%+ of bars have |ret| < ATR, which would bias DEF permanently).

    After computing raw per-mode rewards, a **cross-sectional z-score** is
    applied per row (bar): subtract the row mean and divide by the row
    standard deviation.  This guarantees mean-zero rewards across modes for
    every bar, eliminating systematic bias from fat-tailed return
    distributions (e.g. BO previously had +0.08 mean due to |ret| skew).

    Args:
        ret:      1-bar return array (close.pct_change()). NaN → 0 reward.
        trend:    ohio_stable_trend array.
        atr:      ohio_feat_atr_ratio_14 array. Used as normalizer for TF/MR.
        lookback: Rolling window for BO/DEF median. Default 168 (7 days @ 1h).

    Returns:
        (N, 4) array: columns = [TF, MR, BO, DEF], cross-sectionally
        z-scored and clipped to [-1, 1].
    """
    safe_ret = np.nan_to_num(ret, nan=0.0)
    safe_trend = np.nan_to_num(trend, nan=0.0)
    safe_atr = np.maximum(np.nan_to_num(atr, nan=0.01), 1e-8)

    trend_sign = np.sign(safe_trend)

    # TF/MR: directional — trend continuation vs reversal
    reward_tf = np.clip((safe_ret * trend_sign) / safe_atr, -1.0, 1.0)
    reward_mr = np.clip((-safe_ret * trend_sign) / safe_atr, -1.0, 1.0)

    # BO/DEF: volatility regime — compare |ret| to its rolling median
    # This centres rewards: ~50% positive for each (median is the midpoint)
    abs_ret = np.abs(safe_ret)
    median_abs_ret = (
        pd.Series(abs_ret)
        .rolling(lookback, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )
    median_abs_ret = np.maximum(median_abs_ret, 1e-8)

    reward_bo = np.clip((abs_ret - median_abs_ret) / median_abs_ret, -1.0, 1.0)
    reward_df = np.clip((median_abs_ret - abs_ret) / median_abs_ret, -1.0, 1.0)

    raw = np.column_stack([reward_tf, reward_mr, reward_bo, reward_df])

    # Cross-sectional z-score: eliminate distributional bias per bar
    row_mean = raw.mean(axis=1, keepdims=True)
    row_std = np.maximum(raw.std(axis=1, keepdims=True), 1e-8)
    normalized = (raw - row_mean) / row_std

    return np.clip(normalized, -1.0, 1.0)


def _compute_hedge_weights(
    rewards: np.ndarray,
    eta: float,
    weight_floor: float,
    lookback: int = 168,
) -> np.ndarray:
    """Compute Hedge algorithm weights from rolling cumulative rewards.

    w_i(t) ∝ exp(eta * rolling_sum(reward_i, lookback)), with per-mode floor.

    Uses a rolling window (default 168 bars = 7 days) instead of all-history
    cumsum to prevent a single mode from dominating over long horizons.
    Log-sum-exp stabilisation prevents overflow; weight floor is applied
    *after* normalisation to provide relative uplift (not a hard minimum —
    re-normalisation shrinks effective floor slightly below the stated value).

    Args:
        rewards:      (N, 4) reward array from _compute_rewards.
        eta:          Learning rate. 0.0 → uniform weights.
        weight_floor: Minimum weight per mode (prevents mode death).
        lookback:     Rolling window size in bars. Default 168 (7 days @ 1h).

    Returns:
        (N, 4) array of normalized weights per row, each row sums to 1.0.
    """
    # Rolling sum over lookback window (min_periods=1 for warmup)
    rolling_rewards = (
        pd.DataFrame(rewards)
        .rolling(lookback, min_periods=1)
        .sum()
        .to_numpy(dtype=np.float64)
    )
    scaled = eta * rolling_rewards

    # Log-sum-exp stabilisation: subtract row max to prevent overflow
    max_scaled = scaled.max(axis=1, keepdims=True)
    raw_weights = np.exp(scaled - max_scaled)

    # Normalise → apply floor → re-normalise
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    normalised = raw_weights / row_sums
    floored = np.maximum(normalised, weight_floor)
    floored_sums = floored.sum(axis=1, keepdims=True)
    return floored / floored_sums


# ---------------------------------------------------------------------------
# FitnessEstimator
# ---------------------------------------------------------------------------


class FitnessEstimator:
    """Compute strategy fitness scores from a StateVector and StrategyProfiles.

    Fitness is a scalar in [0, 1] that reflects how well the current market
    state matches a strategy mode's ideal conditions.

    Scoring formula for each axis:
        scale     = 2.0  if axis == "trend_persistence" else 1.0
        proximity = (1.0 - |axis_value - pref.ideal| / scale) ** 2
        axis_score += pref.weight * max(0.0, proximity)

    Meta bonus:
        meta_bonus = meta.confidence * profile.meta_confidence_weight
                   + meta.stability  * profile.meta_stability_weight

    Final score:
        fitness = clip(axis_score + meta_bonus, 0.0, 1.0)
    """

    def __init__(
        self,
        profiles: dict[StrategyMode, StrategyProfile] | None = None,
        hedge_eta: float = 0.25,
        hedge_temperature: float = 1.0,
        hedge_weight_floor: float = 0.05,
    ) -> None:
        """Initialise the estimator.

        Args:
            profiles: Pre-loaded strategy profiles.  If *None*, the four
                      built-in YAML profiles are loaded automatically.
            hedge_eta: Hedge algorithm learning rate. 0.0 → uniform weights.
            hedge_temperature: Temperature for softening fitness scores before
                               Hedge adjustment. Higher values flatten scores.
            hedge_weight_floor: Minimum weight per mode to prevent mode death.
        """
        if hedge_temperature <= 0.0:
            raise ValueError(f"hedge_temperature must be > 0, got {hedge_temperature}")
        if hedge_eta < 0.0:
            raise ValueError(f"hedge_eta must be >= 0, got {hedge_eta}")
        if not (0.0 <= hedge_weight_floor < 0.25):
            raise ValueError(f"hedge_weight_floor must be in [0, 0.25), got {hedge_weight_floor}")

        if profiles is None:
            profiles = load_default_profiles()
        self._profiles: dict[StrategyMode, StrategyProfile] = profiles
        self._hedge_eta = hedge_eta
        self._hedge_temperature = hedge_temperature
        self._hedge_weight_floor = hedge_weight_floor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_fitness(
        self,
        state: StateVector,
        meta: StateMeta,
        profile: StrategyProfile,
    ) -> float:
        """Compute fitness score for a single strategy profile.

        Args:
            state: Stabilised 7-axis market state vector.
            meta:  Quality metadata accompanying the state vector.
            profile: Strategy profile defining axis preferences.

        Returns:
            Fitness score in [0.0, 1.0].
        """
        axis_score = 0.0

        for axis_name, pref in profile.preferences.items():
            axis_value: float = getattr(state, axis_name)
            scale = _TREND_SCALE if axis_name == _TREND_AXIS else _DEFAULT_SCALE
            proximity = (1.0 - abs(axis_value - pref.ideal) / scale) ** 2
            axis_score += pref.weight * max(0.0, proximity)

        meta_bonus = (
            meta.confidence * profile.meta_confidence_weight
            + meta.stability * profile.meta_stability_weight
        )

        raw = axis_score + meta_bonus
        return float(np.clip(raw, 0.0, 1.0))

    def compute_all(
        self,
        state: StateVector,
        meta: StateMeta,
    ) -> StrategyFitness:
        """Compute fitness scores for all four strategy modes.

        Args:
            state: Stabilised market state vector.
            meta:  Quality metadata for the state.

        Returns:
            StrategyFitness with scores for each of the four modes.

        Raises:
            KeyError: If any StrategyMode profile is not loaded.
        """
        return StrategyFitness(
            trend_following=self.compute_fitness(
                state, meta, self._profiles[StrategyMode.TREND_FOLLOWING]
            ),
            mean_reversion=self.compute_fitness(
                state, meta, self._profiles[StrategyMode.MEAN_REVERSION]
            ),
            breakout=self.compute_fitness(
                state, meta, self._profiles[StrategyMode.BREAKOUT]
            ),
            defensive=self.compute_fitness(
                state, meta, self._profiles[StrategyMode.DEFENSIVE]
            ),
        )

    def compute_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Vectorised computation: add fitness columns from ohio_stable_* columns.

        Reads:
            ohio_stable_trend, ohio_stable_volatility, ohio_stable_downside,
            ohio_stable_liquidity, ohio_stable_relative_strength, ohio_stable_correlation,
            ohio_stable_breadth
            ohio_meta_confidence, ohio_meta_stability

        Writes (returns new DataFrame, does not mutate input):
            ohio_fitness_trend_following
            ohio_fitness_mean_reversion
            ohio_fitness_breakout
            ohio_fitness_defensive
            ohio_active_mode  (StrategyMode.value string of the best-fit mode)

        Args:
            dataframe: Candle dataframe that already has ohio_stable_* and
                       ohio_meta_* columns populated.

        Returns:
            New DataFrame with the five additional columns appended.
        """
        df = dataframe.copy()

        # ------------------------------------------------------------------
        # Build axis arrays from ohio_stable_* columns
        # ------------------------------------------------------------------
        axis_arrays: dict[str, np.ndarray] = {}
        for col_suffix, axis_name in _STABLE_COL_TO_AXIS.items():
            col = f"ohio_stable_{col_suffix}"
            if col not in df.columns:
                logger.warning("Missing column %r — defaulting to 0.0", col)
                axis_arrays[axis_name] = np.zeros(len(df), dtype=np.float64)
            else:
                axis_arrays[axis_name] = df[col].to_numpy(dtype=np.float64)

        # ------------------------------------------------------------------
        # Meta arrays
        # ------------------------------------------------------------------
        def _meta_array(suffix: str) -> np.ndarray:
            col = f"ohio_meta_{suffix}"
            if col not in df.columns:
                logger.warning("Missing column %r — defaulting to 0.0", col)
                return np.zeros(len(df), dtype=np.float64)
            return df[col].to_numpy(dtype=np.float64)

        # NaN within existing columns → 0.0 (matches scalar path: no bonus).
        confidence_arr = np.nan_to_num(_meta_array("confidence"), nan=0.0)
        stability_arr = np.nan_to_num(_meta_array("stability"), nan=0.0)

        # ------------------------------------------------------------------
        # Compute per-mode fitness vectors
        # ------------------------------------------------------------------
        mode_scores: dict[str, np.ndarray] = {}
        for mode, profile in self._profiles.items():
            axis_score = np.zeros(len(df), dtype=np.float64)

            for axis_name, pref in profile.preferences.items():
                arr = axis_arrays.get(axis_name, np.zeros(len(df), dtype=np.float64))
                scale = _TREND_SCALE if axis_name == _TREND_AXIS else _DEFAULT_SCALE
                proximity = (1.0 - np.abs(arr - pref.ideal) / scale) ** 2
                # NaN in arr → NaN proximity; treat as zero contribution
                # (matches scalar path where max(0.0, NaN) → 0.0).
                proximity = np.nan_to_num(proximity, nan=0.0)
                axis_score += pref.weight * np.maximum(0.0, proximity)

            meta_bonus = (
                confidence_arr * profile.meta_confidence_weight
                + stability_arr * profile.meta_stability_weight
            )

            raw = axis_score + meta_bonus
            mode_scores[mode.value] = np.clip(raw, 0.0, 1.0)

        # ------------------------------------------------------------------
        # Write fitness output columns (raw, pre-Hedge)
        # ------------------------------------------------------------------
        mode_order = [
            StrategyMode.TREND_FOLLOWING,
            StrategyMode.MEAN_REVERSION,
            StrategyMode.BREAKOUT,
            StrategyMode.DEFENSIVE,
        ]

        score_matrix = np.column_stack(
            [mode_scores[m.value] for m in mode_order]
        )

        for mode in mode_order:
            df[f"ohio_fitness_{mode.value}"] = mode_scores[mode.value]

        # ------------------------------------------------------------------
        # Hedge-based mode selection
        # ------------------------------------------------------------------
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column for Hedge rewards")
        ret = df["close"].pct_change().to_numpy(dtype=np.float64)
        trend = (
            df.get("ohio_stable_trend", pd.Series(0.0, index=df.index))
            .to_numpy(dtype=np.float64)
        )
        atr = (
            df.get("ohio_feat_atr_ratio_14", pd.Series(0.01, index=df.index))
            .fillna(0.01)
            .to_numpy(dtype=np.float64)
        )

        rewards = _compute_rewards(ret, trend, atr)
        hedge_weights = _compute_hedge_weights(
            rewards, self._hedge_eta, self._hedge_weight_floor,
        )

        # Temperature flatten + Hedge adjust → argmax
        flattened = np.power(
            np.maximum(score_matrix, 1e-8),
            1.0 / self._hedge_temperature,
        )
        adjusted = flattened * hedge_weights
        best_indices = np.argmax(adjusted, axis=1)

        mode_values = np.array([m.value for m in mode_order])
        df["ohio_active_mode"] = mode_values[best_indices]

        # Hedge weight columns for monitoring
        for i, mode in enumerate(mode_order):
            df[f"ohio_hedge_weight_{mode.value}"] = hedge_weights[:, i]

        return df
