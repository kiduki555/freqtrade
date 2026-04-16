"""Breakout entry strategy — Bollinger Squeeze release + Donchian break.

Enter on the first bar after a Bollinger Squeeze ends, confirmed by a
Donchian channel breakout.  Previous filters (volume gate, candle
strength, BB expansion) were removed because they suppress true
positives more than false positives in crypto markets.

Sources:
- Arda (SSRN 5775962): Volatility contraction then expansion
- Wen et al. (SSRN 4080253): Crypto technical analysis efficiency
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "bo_squeeze_min_bars": 6.0,
    "bo_donchian_window": 20.0,
}


class BreakoutEntry:
    """Enter on Donchian channel break when a Bollinger Squeeze releases."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        sq_min = int(params.get("bo_squeeze_min_bars", DEFAULT_PARAMS["bo_squeeze_min_bars"]))

        close = dataframe["close"]
        squeeze = dataframe.get("ohio_feat_squeeze_count", pd.Series(0.0, index=dataframe.index))
        don_upper = dataframe.get(
            "ohio_feat_donchian_upper_20",
            pd.Series(float("inf"), index=dataframe.index),
        )
        don_lower = dataframe.get(
            "ohio_feat_donchian_lower_20",
            pd.Series(0.0, index=dataframe.index),
        )

        # Recent squeeze (within last 3 bars)
        had_recent_squeeze = (
            (squeeze.shift(1).fillna(0) >= sq_min)
            | (squeeze.shift(2).fillna(0) >= sq_min)
            | (squeeze.shift(3).fillna(0) >= sq_min)
        )

        # Fresh Donchian break
        don_break_up = (close > don_upper.shift(1)) & (close.shift(1) <= don_upper.shift(1))
        don_break_down = (close < don_lower.shift(1)) & (close.shift(1) >= don_lower.shift(1))

        # Volume confirmation: current volume > 1.5x 20-bar average
        volume = dataframe.get("volume", pd.Series(0.0, index=dataframe.index))
        vol_sma = volume.rolling(20).mean()
        vol_spike = volume > (1.5 * vol_sma)

        long_mask = had_recent_squeeze & don_break_up & vol_spike
        short_mask = had_recent_squeeze & don_break_down & vol_spike

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
