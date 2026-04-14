"""Breadth dispersion factor — axis 7 of the StateVector (shared).

TODO(FT-019): Replace placeholder with actual cross-asset breadth feature.
"""
from __future__ import annotations

import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_breadth(row: dict[str, float]) -> float:
    """Compute breadth_dispersion axis from normalized cross-asset features.

    Shared axis — computed once per market snapshot, not per symbol.
    Returns: float in [0, 1]. 0.5 when cross-asset data is unavailable.
    """
    breadth = row.get("ohio_norm_cross_asset_placeholder", float("nan"))
    if math.isnan(breadth):
        return 0.5
    return clamp01(breadth)
