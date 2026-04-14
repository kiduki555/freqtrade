"""FT-015: ExitAdapter — stoploss & exit-reason computation.

Translates StrategyProfile + ExecutionPolicy into concrete stoploss
values and exit signals.  No Freqtrade framework imports.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HARD_FLOOR: float = -0.20       # Freqtrade hard ceiling (self.stoploss)
_TIGHTEST: float = -0.01         # Never tighter than -1%
_TRANSITION_TIGHTEN_THRESHOLD: float = 0.70   # tighten stoploss above this
_TRANSITION_TIGHTEN_FACTOR: float = 0.70      # multiply adjusted by this (30% narrower)
_TRAILING_PROFIT_THRESHOLD: float = 0.02      # lock profit above 2%
_TRAILING_PROFIT_RATIO: float = 0.50          # lock half of unrealized profit


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from *a* to *b* by *t* in [0, 1]."""
    t = _clamp(t, 0.0, 1.0)
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_stoploss(
    profile: object,
    policy: object,
    current_profit: float,
    transition_risk: float,
    fitness_score: float = 0.5,
) -> float:
    """Compute the dynamic stoploss ratio for an open position.

    Args:
        profile: A ``StrategyProfile`` with ``stoploss_range``
                 (sl_wide, sl_tight — both negative).
        policy: An ``ExecutionPolicy`` with ``enabled`` and
                ``stoploss_width_adj``.
        current_profit: Unrealised profit ratio of the position.
        transition_risk: Regime-transition probability in [0, 1].
            Note: entry is blocked at >0.85 (risk_gate), but stoploss
            tightening starts at >0.70 — intentional graduated response.
        fitness_score: Current fitness [0, 1] for lerping within stoploss_range.

    Returns:
        Negative float representing the stoploss ratio.
    """
    if not policy.enabled:  # type: ignore[union-attr]
        return _HARD_FLOOR

    sl_wide, sl_tight = profile.stoploss_range  # type: ignore[union-attr]
    base_sl = _lerp(sl_wide, sl_tight, fitness_score)

    adjusted = base_sl + policy.stoploss_width_adj  # type: ignore[union-attr]

    # Regime-transition tightening (graduated: fires before entry block at 0.85)
    if transition_risk > _TRANSITION_TIGHTEN_THRESHOLD:
        adjusted *= _TRANSITION_TIGHTEN_FACTOR  # closer to zero = tighter

    # Trailing profit lock
    if current_profit > _TRAILING_PROFIT_THRESHOLD:
        adjusted = max(adjusted, -(current_profit * _TRAILING_PROFIT_RATIO))

    return _clamp(adjusted, _HARD_FLOOR, _TIGHTEST)


def compute_exit(
    bars_since_entry: int,
    current_profit: float,
    fitness_score: float,
    transition_risk: float,
    is_kill_switch: bool,
) -> str | None:
    """Decide whether the position should be exited and why.

    Args:
        bars_since_entry: Number of completed bars since entry.
        current_profit: Unrealised profit ratio.
        fitness_score: Current fitness of the active strategy mode.
        transition_risk: Regime-transition probability in [0, 1].
        is_kill_switch: Whether the global kill switch is active.

    Returns:
        A string exit-reason tag, or ``None`` if no exit is warranted.
    """
    if is_kill_switch:
        return "ohio_kill_switch"

    if bars_since_entry >= 12 and fitness_score < 0.15:
        return "ohio_time_exit"

    if transition_risk > 0.80:
        return "ohio_regime_exit"

    if current_profit > 0.03 and fitness_score < 0.30:
        return "ohio_profit_preserve"

    return None
