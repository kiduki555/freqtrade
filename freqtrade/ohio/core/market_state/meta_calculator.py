"""MetaCalculator: inference quality metadata from stabilized state.

Pipeline position: Stabilizer → **MetaCalculator** → Fitness

Computes StateMeta (transition_risk, confidence, stability, data_mode)
from ohio_stable_* and ohio_factor_* columns.

Usage::

    from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator

    calc = MetaCalculator()
    dataframe = calc.compute(dataframe)
    meta = calc.compute_meta(row_dict)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from freqtrade.ohio.core.domain.models import DataMode, StateMeta


# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

_AXES = [
    "trend",
    "volatility",
    "downside",
    "liquidity",
    "relative_strength",
    "correlation",
    "breadth",
]

_FACTOR_COLS = [f"ohio_factor_{a}" for a in _AXES]
_STABLE_COLS = [f"ohio_stable_{a}" for a in _AXES]

_JUMP_THRESHOLD = 0.15   # normalisation constant for mean_delta
_MAX_DELTA_NORM = 0.30   # normalisation constant for max_delta
_VOL_SCALE = 5.0         # volatility amplifier
_FLIP_SCALE = 3.0        # flip-rate amplifier
_FLIP_WINDOW = 12        # bars for flip-rate computation
_VOL_WINDOW = 6          # bars for factor volatility
_STABLE_VAR_WINDOW = 12  # bars for stable-variance computation


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, value))


def _get_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return *col* from *df*, or a NaN series if absent."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------------------
# MetaCalculator
# ---------------------------------------------------------------------------


class MetaCalculator:
    """Compute StateMeta from stabilized state and raw features.

    Reads ohio_stable_* and ohio_factor_* columns; writes ohio_meta_* columns.
    """

    # ------------------------------------------------------------------
    # Vectorized DataFrame path
    # ------------------------------------------------------------------

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add ohio_meta_* columns to *dataframe*.

        Args:
            dataframe: DataFrame with ohio_stable_* and ohio_factor_* columns.

        Returns:
            Same DataFrame (copy) with four ohio_meta_* columns appended.
        """
        if dataframe.empty:
            result = dataframe.copy()
            for col in (
                "ohio_meta_transition_risk",
                "ohio_meta_confidence",
                "ohio_meta_stability",
                "ohio_meta_data_mode",
            ):
                result[col] = np.nan if col != "ohio_meta_data_mode" else "degraded"
            return result

        result = dataframe.copy()

        # ---- axis deltas: |factor - stable| --------------------------------
        delta_parts: list[pd.Series] = []
        for factor_col, stable_col in zip(_FACTOR_COLS, _STABLE_COLS):
            f = _get_col(result, factor_col)
            s = _get_col(result, stable_col)
            delta_parts.append((f - s).abs())

        delta_df = pd.concat(delta_parts, axis=1)
        mean_delta = delta_df.mean(axis=1)
        max_delta = delta_df.max(axis=1)

        # ---- factor volatility: rolling(6).std per axis, then mean ----------
        vol_parts: list[pd.Series] = []
        for factor_col in _FACTOR_COLS:
            col = _get_col(result, factor_col)
            vol_parts.append(col.rolling(_VOL_WINDOW, min_periods=2).std())

        factor_volatility = pd.concat(vol_parts, axis=1).mean(axis=1)

        # ---- flip rate: fraction of axes that crossed midpoint in last 12 bars
        flip_parts: list[pd.Series] = []
        for factor_col in _FACTOR_COLS:
            col = _get_col(result, factor_col)
            midpoint = 0.0 if "trend" in factor_col else 0.5
            crossed = (col > midpoint).astype(float).diff().abs()
            flip_parts.append(
                crossed.rolling(_FLIP_WINDOW, min_periods=1).sum()
            )

        flip_count = pd.concat(flip_parts, axis=1).sum(axis=1)
        flip_rate = flip_count / (_FLIP_WINDOW * len(_AXES))

        # ---- transition_risk -----------------------------------------------
        transition_risk = (
            0.35 * (mean_delta / _JUMP_THRESHOLD).clip(upper=1.0)
            + 0.25 * (max_delta / _MAX_DELTA_NORM).clip(upper=1.0)
            + 0.20 * (factor_volatility * _VOL_SCALE).clip(upper=1.0)
            + 0.20 * (flip_rate * _FLIP_SCALE).clip(upper=1.0)
        ).clip(0.0, 1.0)

        # ---- NaN ratio per row (ohio_norm_* columns) -----------------------
        norm_cols = [c for c in result.columns if c.startswith("ohio_norm_")]
        if norm_cols:
            nan_ratio = result[norm_cols].isna().mean(axis=1)
        else:
            nan_ratio = pd.Series(1.0, index=result.index)

        # ---- stable variance: rolling(12).std per axis, then mean ----------
        stable_var_parts: list[pd.Series] = []
        for stable_col in _STABLE_COLS:
            col = _get_col(result, stable_col)
            stable_var_parts.append(
                col.rolling(_STABLE_VAR_WINDOW, min_periods=2).std()
            )

        stable_variance = pd.concat(stable_var_parts, axis=1).mean(axis=1)

        # ---- confidence ----------------------------------------------------
        confidence = (
            1.0
            - 0.35 * nan_ratio
            - 0.25 * (stable_variance * 3.0).clip(upper=1.0)
            - 0.20 * transition_risk
            - 0.20 * (factor_volatility * _VOL_SCALE).clip(upper=1.0)
        ).clip(0.0, 1.0)

        # ---- stability -----------------------------------------------------
        stability = (
            (1.0 - transition_risk) * 0.70
            + confidence * 0.20
            + (1.0 - (factor_volatility * 3.0).clip(upper=1.0)) * 0.10
        ).clip(0.0, 1.0)

        # ---- data_mode -----------------------------------------------------
        def _mode_from_ratio(r: float) -> str:
            if r == 0.0:
                return DataMode.FULL.value
            if r < 0.3:
                return DataMode.FALLBACK.value
            return DataMode.DEGRADED.value

        data_mode = nan_ratio.map(_mode_from_ratio)

        # ---- write columns -------------------------------------------------
        result["ohio_meta_transition_risk"] = transition_risk
        result["ohio_meta_confidence"] = confidence
        result["ohio_meta_stability"] = stability
        result["ohio_meta_data_mode"] = data_mode

        return result

    # ------------------------------------------------------------------
    # Single-row path
    # ------------------------------------------------------------------

    def compute_meta(self, row: dict[str, float]) -> StateMeta:
        """Compute StateMeta for a single row (live trading).

        Args:
            row: Dict of column values for the current bar.

        Returns:
            StateMeta frozen dataclass.
        """
        # ---- axis deltas ---------------------------------------------------
        axis_deltas = [
            abs(row.get(f"ohio_factor_{a}", 0.0) - row.get(f"ohio_stable_{a}", 0.0))
            for a in _AXES
        ]
        mean_delta = sum(axis_deltas) / len(axis_deltas)
        max_delta = max(axis_deltas)

        # ---- factor volatility (rolling not available; use 0 as fallback) --
        factor_volatility = row.get("_meta_factor_volatility", 0.0)

        # ---- flip rate (rolling not available; use 0 as fallback) ----------
        flip_rate = row.get("_meta_flip_rate", 0.0)

        # ---- transition_risk -----------------------------------------------
        transition_risk = _clamp01(
            0.35 * _clamp01(mean_delta / _JUMP_THRESHOLD)
            + 0.25 * _clamp01(max_delta / _MAX_DELTA_NORM)
            + 0.20 * _clamp01(factor_volatility * _VOL_SCALE)
            + 0.20 * _clamp01(flip_rate * _FLIP_SCALE)
        )

        # ---- NaN ratio -----------------------------------------------------
        norm_keys = [k for k in row if k.startswith("ohio_norm_")]
        if norm_keys:
            nan_count = sum(
                1 for k in norm_keys if _is_nan(row[k])
            )
            nan_ratio = nan_count / len(norm_keys)
        else:
            nan_ratio = 1.0

        # ---- stable variance (rolling not available; use 0 as fallback) ----
        stable_variance = row.get("_meta_stable_variance", 0.0)

        # ---- confidence ----------------------------------------------------
        confidence = _clamp01(
            1.0
            - 0.35 * nan_ratio
            - 0.25 * _clamp01(stable_variance * 3.0)
            - 0.20 * transition_risk
            - 0.20 * _clamp01(factor_volatility * _VOL_SCALE)
        )

        # ---- stability -----------------------------------------------------
        stability = _clamp01(
            (1.0 - transition_risk) * 0.70
            + confidence * 0.20
            + (1.0 - _clamp01(factor_volatility * 3.0)) * 0.10
        )

        # ---- data_mode -----------------------------------------------------
        if nan_ratio == 0.0:
            data_mode = DataMode.FULL
        elif nan_ratio < 0.3:
            data_mode = DataMode.FALLBACK
        else:
            data_mode = DataMode.DEGRADED

        return StateMeta(
            transition_risk=transition_risk,
            confidence=confidence,
            stability=stability,
            data_mode=data_mode,
        )


def _is_nan(value: object) -> bool:
    """Return True if *value* is NaN (float or numpy scalar)."""
    try:
        return math.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
