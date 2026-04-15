"""Trend Following entry strategy — TSMOM + KAMA + ADX + Hurst.

Sources:
- Huang et al. 2024 (SSRN 4825389): Vol-weighted TSMOM
- Sepp (SSRN 3167787): Volatility scaling
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "tf_adx_threshold": 25.0,
    "tf_hurst_threshold": 0.55,
    "tf_kama_slope_threshold": 0.001,
    "tf_vol_scale_target": 1.0,
}


class TrendFollowingEntry:
    """Enter on strong persistent trends confirmed by KAMA + ADX + Hurst."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        adx_th = params.get("tf_adx_threshold", DEFAULT_PARAMS["tf_adx_threshold"])
        hurst_th = params.get("tf_hurst_threshold", DEFAULT_PARAMS["tf_hurst_threshold"])
        kama_th = params.get("tf_kama_slope_threshold", DEFAULT_PARAMS["tf_kama_slope_threshold"])

        trend = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index))
        adx = dataframe.get("ohio_feat_adx_14", pd.Series(0.0, index=dataframe.index))
        hurst = dataframe.get("ohio_feat_hurst_168", pd.Series(0.5, index=dataframe.index))
        kama_slope = dataframe.get("ohio_feat_kama_slope", pd.Series(0.0, index=dataframe.index))

        # Base gate: strong trend (ADX) + persistent (Hurst)
        base_gate = (adx > adx_th) & (hurst > hurst_th)

        long_mask = base_gate & (kama_slope > kama_th) & (trend > 0)
        short_mask = base_gate & (kama_slope < -kama_th) & (trend < 0)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
