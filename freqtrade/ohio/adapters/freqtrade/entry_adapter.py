"""EntryAdapter — dispatch entry signals to mode-specific strategies.

Each mode has a dedicated entry strategy implementing vectorized
computation. Unknown modes fall back to trend-sign logic.
"""
from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class EntryStrategy(Protocol):
    """Protocol for mode-specific entry strategy."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        """Add enter_long/enter_short columns. Must be vectorized."""
        ...


class EntryAdapter:
    """Dispatch entry signal computation to mode-specific strategies."""

    def __init__(self) -> None:
        from freqtrade.ohio.adapters.freqtrade.entries.trend_following import (
            TrendFollowingEntry,
        )
        from freqtrade.ohio.adapters.freqtrade.entries.mean_reversion import (
            MeanReversionEntry,
        )
        from freqtrade.ohio.adapters.freqtrade.entries.breakout import (
            BreakoutEntry,
        )
        from freqtrade.ohio.adapters.freqtrade.entries.defensive import (
            DefensiveEntry,
        )

        self._strategies: dict[str, EntryStrategy] = {
            "trend_following": TrendFollowingEntry(),
            "mean_reversion": MeanReversionEntry(),
            "breakout": BreakoutEntry(),
            "defensive": DefensiveEntry(),
        }

    def compute_entries(
        self,
        dataframe: pd.DataFrame,
        mode: str,
        params: dict[str, float],
    ) -> pd.DataFrame:
        """Dispatch to mode-specific strategy. Unknown modes use fallback."""
        strategy = self._strategies.get(mode)
        if strategy is None:
            logger.warning("Unknown mode %r, using trend-sign fallback", mode)
            return self._fallback_entries(dataframe)
        return strategy.compute_entries(dataframe, params)

    @staticmethod
    def _fallback_entries(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fallback: use trend sign as entry signal."""
        trend = dataframe.get(
            "ohio_stable_trend", pd.Series(0.0, index=dataframe.index)
        )
        dataframe["enter_long"] = (trend >= 0).astype(int)
        dataframe["enter_short"] = (trend < 0).astype(int)
        return dataframe
