"""Trend-following entry strategy stub."""
from __future__ import annotations

import pandas as pd

DEFAULT_PARAMS: dict[str, float] = {}


class TrendFollowingEntry:
    """Trend-following entry: go with the trend direction."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        trend = dataframe.get(
            "ohio_stable_trend", pd.Series(0.0, index=dataframe.index),
        )
        dataframe["enter_long"] = (trend > 0).astype(int)
        dataframe["enter_short"] = (trend < 0).astype(int)
        return dataframe
