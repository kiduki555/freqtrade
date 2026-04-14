"""Volatility level factor — axis 2 of the StateVector."""
from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_volatility(row: dict[str, float]) -> float:
    """Compute volatility_level axis from normalized features.

    Returns: float in [0, 1]. Higher = more volatile.
    """
    return clamp01(
        0.50 * row.get("ohio_norm_realized_vol_24", 0.5)
        + 0.30 * row.get("ohio_norm_atr_ratio_14", 0.5)
        + 0.20 * row.get("ohio_norm_parkinson_vol_24", 0.5)
    )
