"""Downside pressure factor — axis 3 of the StateVector."""
from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_downside(row: dict[str, float]) -> float:
    """Compute downside_pressure axis from normalized features.

    Returns: float in [0, 1]. Higher = more downside risk.
    """
    return clamp01(
        0.25 * row.get("ohio_norm_sortino_downside_24", 0.5)
        + 0.25 * row.get("ohio_norm_max_drawdown_24", 0.5)
        + 0.20 * row.get("ohio_norm_cvar_24", 0.5)
        + 0.15 * row.get("ohio_norm_lower_shadow_ratio_24", 0.5)
        + 0.15 * row.get("ohio_norm_negative_return_ratio_24", 0.5)
    )
