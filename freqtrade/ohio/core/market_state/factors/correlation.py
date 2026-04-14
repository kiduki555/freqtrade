"""Correlation stress factor — axis 6 of the StateVector (shared).

TODO(FT-019): Replace placeholder with actual cross-asset correlation feature.
"""
from __future__ import annotations

import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_correlation(row: dict[str, float]) -> float:
    """Compute correlation_stress axis from normalized cross-asset features.

    Shared axis — computed once per market snapshot, not per symbol.
    Returns: float in [0, 1]. 0.5 when cross-asset data is unavailable.
    """
    corr = row.get("ohio_norm_cross_asset_placeholder", float("nan"))
    if math.isnan(corr):
        return 0.5
    return clamp01(corr)
