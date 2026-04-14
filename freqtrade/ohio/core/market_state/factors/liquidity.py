"""Liquidity stress factor — axis 4 of the StateVector."""
from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_liquidity(row: dict[str, float]) -> float:
    """Compute liquidity_stress axis from normalized features.

    Higher spread = more stress; lower volume/OBV slope = more stress.
    Returns: float in [0, 1]. Higher = worse liquidity conditions.
    """
    spread = row.get("ohio_norm_bid_ask_approx", 0.5)
    vol_inv = 1.0 - row.get("ohio_norm_volume_ratio_24", 0.5)
    obv_inv = 1.0 - row.get("ohio_norm_obv_slope_24", 0.5)
    return clamp01(0.40 * spread + 0.35 * vol_inv + 0.25 * obv_inv)
