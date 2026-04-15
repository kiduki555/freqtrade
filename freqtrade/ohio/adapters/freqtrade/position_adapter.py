"""FT-013: PositionAdapter for the OHIO Market State Engine.

Pure computation module that converts fitness scores and execution policies
into position-sizing parameters (stake amount, leverage).

No Freqtrade framework imports — only ohio/core domain types.
"""
from __future__ import annotations

from freqtrade.ohio.core.domain.models import ExecutionPolicy
from freqtrade.ohio.core.strategy_router.strategy_profile import StrategyProfile


# ---------------------------------------------------------------------------
# Internal: Fractional Kelly
# ---------------------------------------------------------------------------

def fractional_kelly(
    win_prob: float,
    risk_reward: float = 2.5,
    fraction: float = 0.25,
) -> float:
    """Compute a fractional Kelly criterion bet fraction.

    Args:
        win_prob: Estimated win probability [0, 1].
        risk_reward: Average win / average loss ratio (must be > 0).
        fraction: Kelly fraction to use (0.25 = quarter-Kelly).

    Returns:
        Non-negative Kelly fraction, floored at 0.0.

    Raises:
        ValueError: If *risk_reward* is <= 0 or *win_prob* is outside [0, 1].
    """
    if risk_reward <= 0:
        raise ValueError(f"risk_reward must be > 0, got {risk_reward}")
    if not (0.0 <= win_prob <= 1.0):
        raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")

    kelly_full = win_prob - (1.0 - win_prob) / risk_reward
    return max(0.0, kelly_full * fraction)


# ---------------------------------------------------------------------------
# Internal: Fitness → Win Probability Calibration
# ---------------------------------------------------------------------------

def _calibrate_win_prob(fitness_score: float) -> float:
    """Map fitness score to conservative win probability estimate.

    Fitness measures state-profile alignment, not actual win probability.
    This linear mapping produces a conservative estimate in [0.50, 0.60],
    reflecting that even ideal market conditions provide only a modest edge.
    """
    return 0.50 + 0.10 * fitness_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_stake(
    policy: ExecutionPolicy,
    fitness_score: float,
    base_stake: float,
    min_stake: float,
    max_stake: float,
    dd_scale: float = 1.0,
) -> float:
    """Compute the position stake amount.

    Args:
        policy: Execution policy from the pipeline.
        fitness_score: Best-strategy fitness [0, 1].
        base_stake: Baseline stake amount (e.g. account balance fraction).
        min_stake: Minimum allowed stake.
        max_stake: Maximum allowed stake.
        dd_scale: Drawdown scale factor [0, 1] from DrawdownController.

    Returns:
        Clamped stake amount in [min_stake, max_stake].
    """
    if not policy.enabled:
        return min_stake

    calibrated_wp = _calibrate_win_prob(fitness_score)
    kelly = fractional_kelly(calibrated_wp, risk_reward=2.0, fraction=0.15)
    stake = base_stake * kelly * policy.size_multiplier * dd_scale

    return max(min_stake, min(stake, max_stake))


def compute_leverage(
    profile: StrategyProfile,
    fitness_score: float,
    max_leverage: float,
) -> float:
    """Compute leverage by interpolating within the profile's leverage range.

    Args:
        profile: Strategy profile with leverage_range (min, max).
        fitness_score: Best-strategy fitness [0, 1], used to lerp.
        max_leverage: Hard ceiling on leverage.

    Returns:
        Leverage value rounded to 1 decimal place, clamped to [1.0, max_leverage].
    """
    lev_min, lev_max = profile.leverage_range
    lev = lev_min + (lev_max - lev_min) * fitness_score
    lev = max(1.0, min(lev, max_leverage))
    return round(lev, 1)
