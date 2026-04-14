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
        0.35 * row.get("ohio_norm_log_return_24", 0.5)
        + 0.25 * row.get("ohio_norm_ma_slope_20", 0.5)
        + 0.20 * row.get("ohio_norm_adx_14", 0.5)
        + 0.20 * row.get("ohio_norm_efficiency_ratio_24", 0.5)
    )
    return math.tanh(2.0 * (raw - 0.5))
