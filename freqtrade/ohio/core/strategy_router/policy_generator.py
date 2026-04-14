"""FT-012: Policy Generator — converts fitness scores into ExecutionPolicy.

Translates a StrategyMode + fitness score + StateMeta into an ExecutionPolicy
that controls position sizing, entry thresholds, max positions, and stoploss
adjustments.

No Freqtrade framework imports — only domain models, pandas, and stdlib.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from freqtrade.ohio.core.domain.models import (
    DataMode,
    ExecutionPolicy,
    StateMeta,
    StrategyFitness,
    StrategyMode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — documented output ranges
# ---------------------------------------------------------------------------

_SIZE_MIN: float = 0.3
_SIZE_MAX: float = 1.5

_ENTRY_ADJ_HIGH_FITNESS: float = -0.05   # high fitness → easier entry (lower bar)
_ENTRY_ADJ_LOW_FITNESS: float = 0.15     # low fitness → harder entry (higher bar)

_MAX_POS_MIN: int = 1
_MAX_POS_MAX: int = 6

_SL_ADJ_MIN: float = -0.01
_SL_ADJ_MAX: float = 0.01

# Meta-adjustment factors
_TRANSITION_RISK_THRESHOLD: float = 0.70
_TRANSITION_RISK_REDUCTION: float = 0.30   # reduce size by 30%

_CONFIDENCE_THRESHOLD: float = 0.50
_CONFIDENCE_REDUCTION: float = 0.20        # reduce size by 20%

_FALLBACK_MAX_POSITIONS: int = 3


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + (b - a) * t, with t clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


class PolicyGenerator:
    """Generate ExecutionPolicy from fitness scores.

    Converts a (StrategyMode, fitness, StateMeta) triple into a frozen
    ExecutionPolicy dataclass using linear interpolation across all output
    dimensions, then applies meta quality adjustments.
    """

    def __init__(self, disabled_threshold: float = 0.40) -> None:
        """Initialise the generator.

        Args:
            disabled_threshold: Fitness score below which enabled=False.
                                 Default matches OhioConfig.fitness_disabled_threshold.
        """
        self._disabled_threshold = disabled_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        mode: StrategyMode,
        fitness: float,
        meta: StateMeta,
    ) -> ExecutionPolicy:
        """Generate ExecutionPolicy for a single strategy mode.

        Args:
            mode:    Which strategy mode to generate a policy for.
            fitness: Fitness score [0, 1] from FitnessEstimator.
            meta:    StateMeta for quality adjustments.

        Returns:
            ExecutionPolicy frozen dataclass.

        Logic:
            enabled = fitness >= disabled_threshold AND data_mode != DEGRADED

            Base values via lerp(fitness):
                size_multiplier      = lerp(0.3, 1.5, fitness)
                entry_threshold_adj  = lerp(0.15, -0.05, fitness)   # inverted
                max_positions        = round(lerp(1, 6, fitness))
                stoploss_width_adj   = lerp(-0.01, 0.01, fitness)

            Meta adjustments:
                transition_risk > 0.7  → size_multiplier * 0.70
                confidence < 0.5       → size_multiplier * 0.80
                data_mode == FALLBACK  → cap max_positions at 3
        """
        enabled = (
            fitness >= self._disabled_threshold
            and meta.data_mode != DataMode.DEGRADED
        )

        # Base lerp values
        raw_size = _lerp(_SIZE_MIN, _SIZE_MAX, fitness)
        entry_adj = _lerp(_ENTRY_ADJ_LOW_FITNESS, _ENTRY_ADJ_HIGH_FITNESS, fitness)
        raw_max_pos = _lerp(float(_MAX_POS_MIN), float(_MAX_POS_MAX), fitness)
        sl_adj = _lerp(_SL_ADJ_MIN, _SL_ADJ_MAX, fitness)

        # Meta adjustments to size_multiplier (order: transition_risk first)
        size = raw_size
        if meta.transition_risk > _TRANSITION_RISK_THRESHOLD:
            size *= (1.0 - _TRANSITION_RISK_REDUCTION)
        if meta.confidence < _CONFIDENCE_THRESHOLD:
            size *= (1.0 - _CONFIDENCE_REDUCTION)

        # Clip size to documented range
        size = max(_SIZE_MIN, min(_SIZE_MAX, size))

        # max_positions rounding + FALLBACK cap
        max_pos = round(raw_max_pos)
        max_pos = max(_MAX_POS_MIN, min(_MAX_POS_MAX, max_pos))
        if meta.data_mode == DataMode.FALLBACK:
            max_pos = min(max_pos, _FALLBACK_MAX_POSITIONS)

        # Clip other outputs to their documented ranges
        entry_adj = max(_ENTRY_ADJ_HIGH_FITNESS, min(_ENTRY_ADJ_LOW_FITNESS, entry_adj))
        sl_adj = max(_SL_ADJ_MIN, min(_SL_ADJ_MAX, sl_adj))

        return ExecutionPolicy(
            strategy_mode=mode,
            enabled=enabled,
            size_multiplier=size,
            entry_threshold_adj=entry_adj,
            max_positions=max_pos,
            stoploss_width_adj=sl_adj,
        )

    def generate_best(
        self,
        fitness: StrategyFitness,
        meta: StateMeta,
    ) -> ExecutionPolicy:
        """Generate policy for the best (highest fitness) strategy mode.

        Args:
            fitness: StrategyFitness with all four mode scores.
            meta:    StateMeta for quality adjustments.

        Returns:
            ExecutionPolicy for fitness.best_mode.
        """
        best = fitness.best_mode
        return self.generate(best, fitness.score_for(best), meta)

    def generate_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Vectorized: add ohio_policy_* columns from ohio_fitness_* + ohio_meta_*.

        Reads per-row:
            ohio_fitness_trend_following, ohio_fitness_mean_reversion,
            ohio_fitness_breakout, ohio_fitness_defensive
            ohio_active_mode
            ohio_meta_transition_risk, ohio_meta_confidence, ohio_meta_data_mode

        Writes (in place on a copy):
            ohio_policy_enabled           (bool)
            ohio_policy_size_multiplier   (float)
            ohio_policy_entry_threshold_adj (float)
            ohio_policy_max_positions     (int)
            ohio_policy_stoploss_width_adj (float)

        NaN in ohio_meta_* columns is treated as worst-case:
            transition_risk NaN → 1.0
            confidence NaN      → 0.0
            data_mode NaN       → "degraded"

        Args:
            dataframe: DataFrame with ohio_fitness_* and ohio_meta_* columns.

        Returns:
            New DataFrame (copy) with ohio_policy_* columns added.
        """
        df = dataframe.copy()

        # --- fitness: select per-row score based on ohio_active_mode ----------
        mode_col = df.get("ohio_active_mode", pd.Series("", index=df.index))

        fitness_tf = df.get(
            "ohio_fitness_trend_following",
            pd.Series(0.0, index=df.index),
        ).fillna(0.0)
        fitness_mr = df.get(
            "ohio_fitness_mean_reversion",
            pd.Series(0.0, index=df.index),
        ).fillna(0.0)
        fitness_bo = df.get(
            "ohio_fitness_breakout",
            pd.Series(0.0, index=df.index),
        ).fillna(0.0)
        fitness_def = df.get(
            "ohio_fitness_defensive",
            pd.Series(0.0, index=df.index),
        ).fillna(0.0)

        # Map mode string → fitness score
        fitness_series = pd.Series(0.0, index=df.index)
        fitness_series = np.where(
            mode_col == StrategyMode.TREND_FOLLOWING.value,
            fitness_tf,
            np.where(
                mode_col == StrategyMode.MEAN_REVERSION.value,
                fitness_mr,
                np.where(
                    mode_col == StrategyMode.BREAKOUT.value,
                    fitness_bo,
                    np.where(
                        mode_col == StrategyMode.DEFENSIVE.value,
                        fitness_def,
                        0.0,  # unknown mode → 0
                    ),
                ),
            ),
        )
        fitness_series = np.clip(fitness_series, 0.0, 1.0)

        # --- meta columns — NaN → worst case --------------------------------
        transition_risk = (
            df.get("ohio_meta_transition_risk", pd.Series(np.nan, index=df.index))
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
        confidence = (
            df.get("ohio_meta_confidence", pd.Series(np.nan, index=df.index))
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        data_mode_raw = (
            df.get("ohio_meta_data_mode", pd.Series(DataMode.DEGRADED.value, index=df.index))
            .fillna(DataMode.DEGRADED.value)
        )

        is_degraded = data_mode_raw == DataMode.DEGRADED.value
        is_fallback = data_mode_raw == DataMode.FALLBACK.value

        # --- vectorized lerp -------------------------------------------------
        f = np.asarray(fitness_series, dtype=float)

        raw_size = _SIZE_MIN + (_SIZE_MAX - _SIZE_MIN) * f
        entry_adj = (
            _ENTRY_ADJ_LOW_FITNESS
            + (_ENTRY_ADJ_HIGH_FITNESS - _ENTRY_ADJ_LOW_FITNESS) * f
        )
        raw_max_pos = _MAX_POS_MIN + (_MAX_POS_MAX - _MAX_POS_MIN) * f
        sl_adj = _SL_ADJ_MIN + (_SL_ADJ_MAX - _SL_ADJ_MIN) * f

        # --- meta adjustments ------------------------------------------------
        size = raw_size.copy()
        size = np.where(
            transition_risk > _TRANSITION_RISK_THRESHOLD,
            size * (1.0 - _TRANSITION_RISK_REDUCTION),
            size,
        )
        size = np.where(
            confidence < _CONFIDENCE_THRESHOLD,
            size * (1.0 - _CONFIDENCE_REDUCTION),
            size,
        )
        size = np.clip(size, _SIZE_MIN, _SIZE_MAX)

        max_pos = np.round(raw_max_pos).astype(int)
        max_pos = np.clip(max_pos, _MAX_POS_MIN, _MAX_POS_MAX)
        max_pos = np.where(is_fallback, np.minimum(max_pos, _FALLBACK_MAX_POSITIONS), max_pos)

        entry_adj = np.clip(entry_adj, _ENTRY_ADJ_HIGH_FITNESS, _ENTRY_ADJ_LOW_FITNESS)
        sl_adj = np.clip(sl_adj, _SL_ADJ_MIN, _SL_ADJ_MAX)

        # --- enabled: fitness >= threshold AND not degraded ------------------
        enabled = (f >= self._disabled_threshold) & (~is_degraded)

        # --- write outputs ---------------------------------------------------
        df["ohio_policy_enabled"] = enabled
        df["ohio_policy_size_multiplier"] = size
        df["ohio_policy_entry_threshold_adj"] = entry_adj
        df["ohio_policy_max_positions"] = max_pos
        df["ohio_policy_stoploss_width_adj"] = sl_adj

        return df
