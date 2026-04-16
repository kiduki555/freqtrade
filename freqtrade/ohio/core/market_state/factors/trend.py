"""Trend persistence factor — axis 1 of the StateVector."""
from __future__ import annotations

import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_trend(row: dict[str, float]) -> float:
    """Compute trend_persistence axis from normalized features.

    Returns: float in [-1, 1]. Positive = uptrend, negative = downtrend.
    """
    raw = (
        0.30 * row.get("ohio_norm_log_return_24", 0.5)
        + 0.22 * row.get("ohio_norm_ma_slope_20", 0.5)
        + 0.17 * row.get("ohio_norm_adx_14", 0.5)
        + 0.16 * row.get("ohio_norm_efficiency_ratio_24", 0.5)
        + 0.15 * row.get("ohio_norm_hurst_168", 0.5)
    )
    log_ret = row.get("ohio_norm_log_return_24", 0.5)
    real_vol = max(row.get("ohio_norm_realized_vol_24", 0.5), 0.15)
    centered_ret = log_ret - 0.5
    vol_adj = centered_ret / real_vol
    vol_adj_norm = max(0.0, min(1.0, vol_adj + 0.5))
    blended = 0.75 * raw + 0.25 * vol_adj_norm
    return math.tanh(2.0 * (blended - 0.5))
