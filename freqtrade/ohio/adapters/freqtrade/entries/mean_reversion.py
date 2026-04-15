"""Mean Reversion entry strategy — Z-score + Hurst gate + BB + RSI.

Sources:
- Avellaneda & Lee 2010: Statistical arbitrage z-score framework
- Beluska & Vojtko (SSRN 4955617): Hurst exponent as MR filter
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "mr_zscore_entry": 2.0,
    "mr_rsi_oversold": 30.0,
    "mr_rsi_overbought": 70.0,
    "mr_hurst_max": 0.45,
}


class MeanReversionEntry:
    """Enter when price is extreme (z-score) in a mean-reverting regime (Hurst < 0.45)."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        zs_th = params.get("mr_zscore_entry", DEFAULT_PARAMS["mr_zscore_entry"])
        rsi_os = params.get("mr_rsi_oversold", DEFAULT_PARAMS["mr_rsi_oversold"])
        rsi_ob = params.get("mr_rsi_overbought", DEFAULT_PARAMS["mr_rsi_overbought"])
        hurst_max = params.get("mr_hurst_max", DEFAULT_PARAMS["mr_hurst_max"])

        zscore = dataframe.get("ohio_feat_zscore_20", pd.Series(0.0, index=dataframe.index))
        rsi = dataframe.get("ohio_feat_rsi_14", pd.Series(50.0, index=dataframe.index))
        hurst = dataframe.get("ohio_feat_hurst_168", pd.Series(0.5, index=dataframe.index))
        close = dataframe["close"]
        bb_lower = dataframe.get("ohio_feat_bb_lower_20", pd.Series(0.0, index=dataframe.index))
        bb_upper = dataframe.get("ohio_feat_bb_upper_20", pd.Series(float("inf"), index=dataframe.index))

        # Hurst gate: only enter MR when market is mean-reverting
        hurst_gate = hurst < hurst_max

        long_mask = hurst_gate & (zscore < -zs_th) & (rsi < rsi_os) & (close <= bb_lower)
        short_mask = hurst_gate & (zscore > zs_th) & (rsi > rsi_ob) & (close >= bb_upper)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
