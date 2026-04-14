"""Relative strength factor — axis 5 of the StateVector."""
from __future__ import annotations

import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_relative_strength(row: dict[str, float]) -> float:
    """Compute relative_strength axis from normalized cross-asset features.

    Returns: float in [0, 1]. 0.5 when cross-asset data is unavailable.
    """
    rs = row.get("ohio_norm_rs_raw", float("nan"))
    if math.isnan(rs):
        return 0.5
    return clamp01(rs)
