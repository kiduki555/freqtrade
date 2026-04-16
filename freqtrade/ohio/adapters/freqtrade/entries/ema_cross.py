"""EMA Crossover entry — Signal 1 of independent entry system.

Enters on EMA(13)/EMA(48) crossover, confirmed by 4H trend direction.
Academic basis: Mesicek & Vojtko (SSRN 5748642), Bitwise Trendwise ETFs.
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "ema_fast_span": 13.0,
    "ema_slow_span": 48.0,
}


class EMACrossEntry:
    """Enter on EMA fast/slow crossover confirmed by higher-timeframe trend."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        fast_span = int(params.get("ema_fast_span", DEFAULT_PARAMS["ema_fast_span"]))
        slow_span = int(params.get("ema_slow_span", DEFAULT_PARAMS["ema_slow_span"]))

        close = dataframe["close"]
        ema_fast = close.ewm(span=fast_span, adjust=False).mean()
        ema_slow = close.ewm(span=slow_span, adjust=False).mean()

        # Crossover detection
        cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        # 4H trend confirmation
        sma_slope_4h = dataframe.get(
            "ohio_4h_sma_slope", pd.Series(0.0, index=dataframe.index),
        )
        bullish_4h = sma_slope_4h > 0
        bearish_4h = sma_slope_4h < 0

        # RSI momentum alignment — avoid crosses while momentum is still weak
        rsi = dataframe.get(
            "ohio_feat_rsi_14", pd.Series(50.0, index=dataframe.index),
        )
        rsi_long_ok = rsi > 45
        rsi_short_ok = rsi < 55

        # ADX > 15 — very soft trend confirmation
        adx = dataframe.get(
            "ohio_feat_adx_14", pd.Series(0.0, index=dataframe.index),
        )
        trending = adx > 15

        # Volume filter removed — vol_confirm was suppressing too many signals
        # RSI+ADX+4H slope combination is sufficient quality filter

        dataframe["enter_long"] = (
            cross_up & bullish_4h & rsi_long_ok & trending
        ).astype(int)
        dataframe["enter_short"] = (
            cross_down & bearish_4h & rsi_short_ok & trending
        ).astype(int)
        return dataframe
