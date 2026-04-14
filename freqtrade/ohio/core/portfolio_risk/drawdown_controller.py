"""FT-016: Drawdown Controller for the OHIO Market State Engine.

Tracks portfolio equity drawdown from peak and returns a tier that governs
position sizing and entry permission.

No Freqtrade framework imports — pure ohio/core module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class DrawdownTier(str, Enum):
    """Severity level of the current drawdown."""

    NORMAL = "normal"    # No meaningful drawdown
    REDUCE = "reduce"    # Reduce position sizes
    BLOCK = "block"      # Block new entries
    KILL = "kill"        # Emergency — flatten everything


# ---------------------------------------------------------------------------
# Config (frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawdownConfig:
    """Threshold percentages that map drawdown depth to a tier.

    All values are expressed as positive fractions (e.g. 0.05 = 5 %).
    """

    reduce_pct: float = 0.05
    block_pct: float = 0.10
    kill_pct: float = 0.15


# ---------------------------------------------------------------------------
# Tier → behaviour mapping
# ---------------------------------------------------------------------------

_SIZE_SCALE: Final[dict[DrawdownTier, float]] = {
    DrawdownTier.NORMAL: 1.0,
    DrawdownTier.REDUCE: 0.5,
    DrawdownTier.BLOCK: 0.0,
    DrawdownTier.KILL: 0.0,
}

_ENTRY_ALLOWED: Final[dict[DrawdownTier, bool]] = {
    DrawdownTier.NORMAL: True,
    DrawdownTier.REDUCE: True,
    DrawdownTier.BLOCK: False,
    DrawdownTier.KILL: False,
}


# ---------------------------------------------------------------------------
# Controller (stateful)
# ---------------------------------------------------------------------------

class DrawdownController:
    """Track equity drawdown and expose risk-tier properties.

    Usage::

        ctrl = DrawdownController()
        tier = ctrl.update(current_equity=98_000)
        if ctrl.entry_allowed:
            size *= ctrl.size_scale
    """

    def __init__(self, config: DrawdownConfig | None = None) -> None:
        self._config = config or DrawdownConfig()
        self._peak: float = 0.0
        self._drawdown: float = 0.0
        self._tier: DrawdownTier = DrawdownTier.NORMAL

    # -- public mutating methods --------------------------------------------

    def update(self, current_equity: float) -> DrawdownTier:
        """Record *current_equity* and return the resulting tier.

        Raises ``ValueError`` for negative equity.
        """
        if current_equity < 0:
            raise ValueError(
                f"current_equity must be >= 0, got {current_equity}"
            )

        # Update peak (first call or new high)
        if current_equity > self._peak:
            self._peak = current_equity

        # Compute drawdown ratio
        if self._peak == 0.0:
            self._drawdown = 0.0
        else:
            self._drawdown = (self._peak - current_equity) / self._peak

        # Classify tier (highest matching threshold wins)
        cfg = self._config
        if self._drawdown >= cfg.kill_pct:
            self._tier = DrawdownTier.KILL
        elif self._drawdown >= cfg.block_pct:
            self._tier = DrawdownTier.BLOCK
        elif self._drawdown >= cfg.reduce_pct:
            self._tier = DrawdownTier.REDUCE
        else:
            self._tier = DrawdownTier.NORMAL

        return self._tier

    def reset(self, equity: float) -> None:
        """Reset peak to *equity* for manual recovery."""
        if equity < 0:
            raise ValueError(f"equity must be >= 0, got {equity}")
        self._peak = equity
        self._drawdown = 0.0
        self._tier = DrawdownTier.NORMAL

    # -- read-only properties -----------------------------------------------

    @property
    def drawdown(self) -> float:
        """Current drawdown ratio (0.0 = no drawdown)."""
        return self._drawdown

    @property
    def peak_equity(self) -> float:
        """Highest equity observed."""
        return self._peak

    @property
    def tier(self) -> DrawdownTier:
        """Current drawdown tier."""
        return self._tier

    @property
    def size_scale(self) -> float:
        """Position-size multiplier for the current tier."""
        return _SIZE_SCALE[self._tier]

    @property
    def entry_allowed(self) -> bool:
        """Whether new entries are permitted under the current tier."""
        return _ENTRY_ALLOWED[self._tier]
