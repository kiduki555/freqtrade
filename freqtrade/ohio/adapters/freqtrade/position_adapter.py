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
    """Map fitness score to win probability estimate.

    Range [0.55, 0.75] — backtest shows 69% actual win rate.
    Previous cap at 0.65 was too conservative and limited position size.
    """
    return 0.55 + 0.20 * fitness_score


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
    correlation_stress: float = 0.5,
    asset_volatility: float = 0.0,
) -> float:
    """Compute the position stake amount.

    Args:
        policy: Execution policy from the pipeline.
        fitness_score: Best-strategy fitness [0, 1].
        base_stake: Baseline stake amount (e.g. account balance fraction).
        min_stake: Minimum allowed stake.
        max_stake: Maximum allowed stake.
        dd_scale: Drawdown scale factor [0, 1] from DrawdownController.
        correlation_stress: Current cross-asset correlation [0, 1].
            Above 0.7, position size is reduced to prevent correlated drawdowns.
        asset_volatility: Current ATR/close ratio for the asset.
            Higher volatility assets get smaller positions (volatility targeting).

    Returns:
        Clamped stake amount in [min_stake, max_stake].
    """
    if not policy.enabled:
        return min_stake

    # Simple aggressive sizing: use 40-70% of proposed base_stake
    # base_stake is typically balance/max_open_trades (e.g. 10000/10 = 1000)
    # With 10 max trades, 70% = 7% of total balance per trade
    # Max aggregate exposure: 10 × 7% = 70% of capital
    size_factor = 0.40 + 0.30 * fitness_score  # [0.40, 0.70] based on fitness
    stake = base_stake * size_factor * policy.size_multiplier * dd_scale

    # Correlation: reduce when whole market moves together
    if correlation_stress > 0.7:
        corr_scale = 1.0 - 0.4 * (correlation_stress - 0.7) / 0.3
        corr_scale = max(0.6, corr_scale)  # floor 60%
        stake *= corr_scale

    # Per-asset vol scaling (high vol assets get smaller size)
    if asset_volatility > 0.03:  # only scale for high-vol assets
        vol_scale = 0.03 / asset_volatility
        vol_scale = max(0.5, vol_scale)  # floor 50%
        stake *= vol_scale

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
