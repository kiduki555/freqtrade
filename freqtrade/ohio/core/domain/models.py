"""OHIO Market State Engine — Domain Objects.

These frozen dataclasses define the system's core contracts.
All pipeline phases communicate through these types.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal


class DataMode(str, Enum):
    """Indicates data availability quality."""
    FULL = "full"           # All data sources available
    FALLBACK = "fallback"   # Some sources missing, using fallback
    DEGRADED = "degraded"   # Significant data missing


class StrategyMode(str, Enum):
    """Available trading strategy modes."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    DEFENSIVE = "defensive"


@dataclass(frozen=True)
class StateVector:
    """7-axis market state representation.

    Per-symbol axes: [-1,1] for trend, [0,1] for others.
    Shared axes: correlation_stress, breadth_dispersion.
    """
    trend_persistence: float     # [-1, 1] directional trend strength
    volatility_level: float      # [0, 1] normalized volatility
    downside_pressure: float     # [0, 1] downside risk intensity
    liquidity_stress: float      # [0, 1] liquidity deterioration
    relative_strength: float     # [0, 1] vs peer basket
    correlation_stress: float    # [0, 1] shared — cross-asset correlation
    breadth_dispersion: float    # [0, 1] shared — return dispersion


@dataclass(frozen=True)
class StateMeta:
    """Inference quality metadata for a StateVector."""
    transition_risk: float    # [0, 1] probability of regime change
    confidence: float         # [0, 1] data quality confidence
    stability: float          # [0, 1] state persistence measure
    data_mode: DataMode       # data availability indicator


@dataclass(frozen=True)
class StrategyFitness:
    """Per-strategy fitness scores from the Strategy Router."""
    trend_following: float    # [0, 1]
    mean_reversion: float     # [0, 1]
    breakout: float           # [0, 1]
    defensive: float          # [0, 1]

    @property
    def best_mode(self) -> StrategyMode:
        """Return the strategy mode with highest fitness."""
        scores = {
            StrategyMode.TREND_FOLLOWING: self.trend_following,
            StrategyMode.MEAN_REVERSION: self.mean_reversion,
            StrategyMode.BREAKOUT: self.breakout,
            StrategyMode.DEFENSIVE: self.defensive,
        }
        return max(scores, key=lambda m: scores[m])

    def score_for(self, mode: StrategyMode) -> float:
        """Get fitness score for a specific strategy mode."""
        return {
            StrategyMode.TREND_FOLLOWING: self.trend_following,
            StrategyMode.MEAN_REVERSION: self.mean_reversion,
            StrategyMode.BREAKOUT: self.breakout,
            StrategyMode.DEFENSIVE: self.defensive,
        }[mode]


@dataclass(frozen=True)
class ExecutionPolicy:
    """Execution parameters derived from fitness score."""
    strategy_mode: StrategyMode
    enabled: bool                    # False if fitness < threshold
    size_multiplier: float           # [0.3, 1.5] position size scaling
    entry_threshold_adj: float       # [-0.05, 0.15] entry signal adjustment
    max_positions: int               # [1, 6] max concurrent positions for this mode
    stoploss_width_adj: float        # [-0.01, 0.01] stoploss width adjustment


@dataclass(frozen=True)
class MarketStateSnapshot:
    """Complete market state at a point in time."""
    symbol: str
    timestamp: datetime
    timeframe: str
    state_vector: StateVector
    meta: StateMeta
    fitness: StrategyFitness
    active_mode: StrategyMode
    policy: ExecutionPolicy


@dataclass(frozen=True)
class ExecutionIntent:
    """What the strategy wants to do — passed to Freqtrade callbacks."""
    symbol: str
    mode: StrategyMode
    direction: Literal["long", "short"]
    size_multiplier: float
    leverage: float
    stoploss: float
    take_profit: float | None = None
    max_hold_bars: int | None = None
    entry_reason: str = ""
