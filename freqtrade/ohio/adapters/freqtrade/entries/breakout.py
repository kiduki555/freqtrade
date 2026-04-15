"""Breakout entry strategy — Bollinger Squeeze + Donchian + Volume.

Sources:
- Arda (SSRN 5775962): Volatility contraction then expansion
- Wen et al. (SSRN 4080253): Crypto technical analysis efficiency
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "bo_squeeze_min_bars": 6.0,
    "bo_volume_mult": 1.5,
    "bo_donchian_window": 20.0,
}


class BreakoutEntry:
    """Enter on Donchian channel break after Bollinger Squeeze with volume confirmation."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        sq_min = int(params.get("bo_squeeze_min_bars", DEFAULT_PARAMS["bo_squeeze_min_bars"]))
        vol_mult = params.get("bo_volume_mult", DEFAULT_PARAMS["bo_volume_mult"])

        close = dataframe["close"]
        volume = dataframe.get("volume", pd.Series(0.0, index=dataframe.index))
        squeeze = dataframe.get("ohio_feat_squeeze_count", pd.Series(0.0, index=dataframe.index))
        don_upper = dataframe.get(
            "ohio_feat_donchian_upper_20",
            pd.Series(float("inf"), index=dataframe.index),
        )
        don_lower = dataframe.get(
            "ohio_feat_donchian_lower_20",
            pd.Series(0.0, index=dataframe.index),
        )
        vol_sma = dataframe.get(
            "ohio_feat_volume_sma_20",
            pd.Series(1.0, index=dataframe.index),
        )

        # Squeeze must have been active recently (current bar or previous bar)
        squeeze_gate = (squeeze >= sq_min) | (squeeze.shift(1).fillna(0) >= sq_min)
        volume_gate = volume > (vol_mult * vol_sma)

        long_mask = squeeze_gate & volume_gate & (close > don_upper)
        short_mask = squeeze_gate & volume_gate & (close < don_lower)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
