"""FT-015: ExitAdapter — stoploss & exit-reason computation.

Translates StrategyProfile + ExecutionPolicy into concrete stoploss
values and exit signals.  No Freqtrade framework imports.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Defaults (overridden by HyperoptParams when optimising)
# ---------------------------------------------------------------------------

_HARD_FLOOR: float = -0.20       # Freqtrade hard ceiling (self.stoploss)
_TIGHTEST: float = -0.01         # Never tighter than -1%
_TRANSITION_TIGHTEN_THRESHOLD: float = 0.70   # tighten stoploss above this
_TRANSITION_TIGHTEN_FACTOR: float = 0.70      # multiply adjusted by this (30% narrower)
_TRAILING_PROFIT_THRESHOLD: float = 0.02      # lock profit above 2%
_TRAILING_PROFIT_RATIO: float = 0.50          # lock half of unrealized profit


@dataclass(frozen=True)
class ExitParams:
    """Tuneable exit/stoploss parameters — passed from strategy hyperopt."""

    # Stoploss
    hard_floor: float = _HARD_FLOOR
    transition_tighten_factor: float = _TRANSITION_TIGHTEN_FACTOR
    trailing_profit_threshold: float = _TRAILING_PROFIT_THRESHOLD
    trailing_profit_ratio: float = _TRAILING_PROFIT_RATIO

    # Exit reasons — thresholds calibrated to observed MR fitness (0.5+)
    time_exit_bars: int = 12
    time_exit_fitness: float = 0.40       # was 0.15 — fitness stays 0.5+ with MR
    regime_exit_risk: float = 0.65        # was 0.80 — MetaCalculator caps make 0.80 unreachable
    profit_preserve_profit: float = 0.03
    profit_preserve_fitness: float = 0.50  # was 0.30 — needs to be above typical MR fitness


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
    atr_scale: float = 1.0,
    atr_ratio: float = 0.0,
    params: ExitParams | None = None,
) -> float:
    """Compute the dynamic stoploss ratio for an open position.

    Args:
        profile: A ``StrategyProfile`` with ``stoploss_range`` and optional
                 ``atr_stop_direction``, ``atr_scale_cap``, ``chandelier_*``.
        policy: An ``ExecutionPolicy`` with ``enabled`` and
                ``stoploss_width_adj``.
        current_profit: Unrealised profit ratio of the position.
        transition_risk: Regime-transition probability in [0, 1].
        fitness_score: Current fitness [0, 1] for lerping within stoploss_range.
        atr_scale: Ratio of current ATR to baseline ATR. 1.0 = average vol.
        atr_ratio: Current ATR/close ratio for Chandelier calculation.
        params: Tuneable parameters. Uses defaults when None.

    Returns:
        Negative float representing the stoploss ratio.
    """
    p = params or ExitParams()

    if not policy.enabled:  # type: ignore[union-attr]
        return p.hard_floor

    sl_wide, sl_tight = profile.stoploss_range  # type: ignore[union-attr]
    base_sl = _lerp(sl_wide, sl_tight, fitness_score)

    adjusted = base_sl + policy.stoploss_width_adj  # type: ignore[union-attr]

    # --- ATR scaling ---
    atr_cap = getattr(profile, "atr_scale_cap", 2.0)
    clamped_scale = _clamp(atr_scale, 1.0 / atr_cap, atr_cap)
    direction = getattr(profile, "atr_stop_direction", "tighten")

    if direction == "widen":
        # MR: vol up → stop wider (more negative)
        adjusted *= clamped_scale
    else:
        # tighten: vol up → stop tighter (closer to zero)
        adjusted *= (1.0 / clamped_scale)

    # Regime-transition tightening (graduated: fires before entry block at 0.85)
    # Only tighten on transition risk if the trade is not in significant profit.
    # Otherwise the tightening triggers premature stop-outs on volatile profitable trades.
    if transition_risk > _TRANSITION_TIGHTEN_THRESHOLD and current_profit < 0.02:
        adjusted *= p.transition_tighten_factor  # closer to zero = tighter

    # --- Trailing: Chandelier or profit lock ---
    chandelier_on = getattr(profile, "chandelier_enabled", False)
    chandelier_mult = getattr(profile, "chandelier_multiplier", 2.5)
    chandelier_act = getattr(profile, "chandelier_activation", 0.02)

    if chandelier_on and atr_ratio > 0 and current_profit > chandelier_act:
        chandelier_sl = -(chandelier_mult * atr_ratio)
        adjusted = max(adjusted, chandelier_sl)  # tighter wins (both negative)
    elif current_profit > p.trailing_profit_threshold:
        adjusted = max(adjusted, -(current_profit * p.trailing_profit_ratio))

    # Use mode-specific hard floor if available from profile, else global
    mode_floor = getattr(profile, "hard_floor", p.hard_floor)
    return _clamp(adjusted, mode_floor, _TIGHTEST)


def compute_exit(
    bars_since_entry: int,
    current_profit: float,
    fitness_score: float,
    transition_risk: float,
    is_kill_switch: bool,
    active_mode: str | None = None,
    params: ExitParams | None = None,
) -> str | None:
    """Decide whether the position should be exited and why.

    Args:
        bars_since_entry: Number of completed bars since entry.
        current_profit: Unrealised profit ratio.
        fitness_score: Current fitness of the active strategy mode.
        transition_risk: Regime-transition probability in [0, 1].
        is_kill_switch: Whether the global kill switch is active.
        active_mode: Current strategy mode string (e.g. "mean_reversion",
                     "trend_following"). Used to scale the time-exit window.
                     ``None`` or unrecognised values use the base bars.
        params: Tuneable parameters. Uses defaults when None.

    Returns:
        A string exit-reason tag, or ``None`` if no exit is warranted.
    """
    p = params or ExitParams()

    if is_kill_switch:
        return "ohio_kill_switch"

    # Regime-conditional holding period.
    # Mean reversion has shorter edge decay (5-10 bars); trend following longer (30-50).
    effective_bars = p.time_exit_bars
    if active_mode == "mean_reversion":
        effective_bars = max(6, p.time_exit_bars // 2)    # shorter hold for MR
    elif active_mode == "trend_following":
        effective_bars = int(p.time_exit_bars * 2.5)       # longer hold for TF

    if bars_since_entry >= effective_bars and fitness_score < p.time_exit_fitness:
        return "ohio_time_exit"

    # Regime exit: only for losing trades during regime transitions.
    # Prevents small losses from becoming trailing_stop losses (-1.5% avg).
    if transition_risk > p.regime_exit_risk and current_profit < 0.005:
        return "ohio_regime_exit"

    if current_profit > p.profit_preserve_profit and fitness_score < p.profit_preserve_fitness:
        return "ohio_profit_preserve"

    return None
