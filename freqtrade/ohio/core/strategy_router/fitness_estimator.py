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
) -> np.ndarray:
    """Compute per-mode rewards from 1-bar return, trend, and ATR.

    Args:
        ret:   1-bar return array (close.pct_change()). NaN → 0 reward.
        trend: ohio_sv_trend_persistence array.
        atr:   ohio_feat_atr_ratio_14 array. Used as normalizer.

    Returns:
        (N, 4) array: columns = [TF, MR, BO, DEF], values in [-1, 1].
    """
    safe_ret = np.nan_to_num(ret, nan=0.0)
    safe_trend = np.nan_to_num(trend, nan=0.0)
    safe_atr = np.maximum(np.nan_to_num(atr, nan=0.01), 1e-8)

    trend_sign = np.sign(safe_trend)

    reward_tf = np.clip((safe_ret * trend_sign) / safe_atr, -1.0, 1.0)
    reward_mr = np.clip((-safe_ret * trend_sign) / safe_atr, -1.0, 1.0)
    reward_bo = np.clip((np.abs(safe_ret) - safe_atr) / safe_atr, -1.0, 1.0)
    reward_df = np.clip((safe_atr - np.abs(safe_ret)) / safe_atr, -1.0, 1.0)

    return np.column_stack([reward_tf, reward_mr, reward_bo, reward_df])


def _compute_hedge_weights(
    rewards: np.ndarray,
    eta: float,
    weight_floor: float,
) -> np.ndarray:
    """Compute Hedge algorithm weights from cumulative rewards.

    w_i(t) ∝ exp(eta * cumsum(reward_i(1..t))), with per-mode floor.

    Args:
        rewards:      (N, 4) reward array from _compute_rewards.
        eta:          Learning rate. 0.0 → uniform weights.
        weight_floor: Minimum weight per mode (prevents mode death).

    Returns:
        (N, 4) array of normalized weights per row, each row sums to 1.0.
    """
    cum_rewards = np.cumsum(rewards, axis=0)
    raw_weights = np.exp(eta * cum_rewards)
    raw_weights = np.maximum(raw_weights, weight_floor)
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    return raw_weights / row_sums


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
    ) -> None:
        """Initialise the estimator.

        Args:
            profiles: Pre-loaded strategy profiles.  If *None*, the four
                      built-in YAML profiles are loaded automatically.
        """
        if profiles is None:
            profiles = load_default_profiles()
        self._profiles: dict[StrategyMode, StrategyProfile] = profiles

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
        # Write output columns
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
        best_indices = np.argmax(score_matrix, axis=1)
        mode_values = np.array([m.value for m in mode_order])
        active_modes = mode_values[best_indices]

        for mode in mode_order:
            df[f"ohio_fitness_{mode.value}"] = mode_scores[mode.value]

        df["ohio_active_mode"] = active_modes

        return df
