"""Defensive entry strategy — vol-targeting + low-ADX gate.

Sources:
- Moreira & Muir 2017: Volatility targeting
- Faber 2007: Tactical asset allocation
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "def_adx_max": 20.0,
    "def_vol_scale": 0.15,
    "def_min_trend_abs": 0.03,
}


class DefensiveEntry:
    """Enter only in calm markets with small positions, following mild trend direction."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        adx_max = params.get("def_adx_max", DEFAULT_PARAMS["def_adx_max"])
        min_trend = params.get("def_min_trend_abs", DEFAULT_PARAMS["def_min_trend_abs"])

        trend = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index))
        adx = dataframe.get("ohio_feat_adx_14", pd.Series(50.0, index=dataframe.index))

        # Only enter in calm (low ADX) markets with minimum directional conviction
        calm_gate = adx < adx_max

        long_mask = calm_gate & (trend > min_trend)
        short_mask = calm_gate & (trend < -min_trend)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
